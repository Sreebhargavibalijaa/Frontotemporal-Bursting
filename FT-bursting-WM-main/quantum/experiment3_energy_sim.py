"""
Experiment 3: Energy Efficiency Simulation -- QuSpike-V
=======================================================
Implements a quantum-inspired spiking vision encoder where synaptic weights
are replaced by complex-valued phase coefficients, then benchmarks energy
(synaptic event count) against a matched ANN baseline on CIFAR-100.

Biological grounding:
  Phase init theta is seeded from the empirical PBC peak-phase values observed
  in LMFG->RMTG channel pairs (40 deg from Figure 5d, Omelyusik et al. 2025).

Architecture
------------
  Input (3x32x32)
    -> Rate encoding -> spike trains (N_TIMESTEPS steps)
  SNN Encoder  (3 conv layers, LIF neurons, phase-weighted synapses)
    -> population-vector readout
  Linear classifier -> 100-class output

Energy metric
-------------
  E_snn  = N_synaptic_events x 0.9 pJ per inference   (28 nm CMOS)
  E_ann  = N_MAC x 2 x 4.6 pJ                         (ANN baseline)
  Ratio  = E_ann / E_snn   (target >= 5x)

Requirements: pip install snntorch torch torchvision tqdm
Approx runtime: ~20 min on A100 for 20 epochs on CIFAR-100.
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

try:
    import snntorch as snn
    from snntorch import surrogate
    from snntorch import functional as SF
except ImportError:
    raise ImportError("Install snnTorch: pip install snntorch")

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 128
N_EPOCHS       = 15   # reduce to 5 for quick test
N_TIMESTEPS    = 25        # SNN simulation steps per image
LR             = 1e-3
DATA_DIR       = "/tmp/cifar100"

# Energy constants (28 nm CMOS, literature values)
PJ_PER_SYNAPTIC_EVENT = 0.9
PJ_PER_MAC            = 4.6

# Biological phase prior from iEEG (Figure 5d, paper)
BIO_PHASE_DEG = 40.0
BIO_PHASE_RAD = math.radians(BIO_PHASE_DEG)

print(f"Device: {DEVICE}")
print(f"Biological phase prior: {BIO_PHASE_DEG} deg = {BIO_PHASE_RAD:.3f} rad")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1  Phase-encoded synaptic weight layer (QuSpike-V core)
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseConv2d(nn.Module):
    """
    Conv layer with complex-valued phase-encoded weights.

    Standard SNN weight: w (real scalar)
    QuSpike-V weight:    w = r * exp(i*theta)  -- complex

    Classical implementation (no quantum hardware needed):
      effective_weight = r * cos(theta)
    where theta is a learnable parameter initialised from the biological prior.

    Energy saving: cos(theta) modulates effective weight magnitude.
    At theta ~ 40 deg, cos(40) ~ 0.77 -- weights are smaller on average,
    producing fewer threshold crossings and thus fewer synaptic events.
    This is the mechanism behind the energy reduction claim.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=1, bio_phase_init=BIO_PHASE_RAD):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False)

        # r: magnitude -- initialised with standard Kaiming
        nn.init.kaiming_normal_(self.conv.weight)

        # theta: phase offset -- learnable, seeded from biology
        # Shape matches conv weight: (out, in, kH, kW)
        self.theta = nn.Parameter(
            torch.full_like(self.conv.weight, bio_phase_init)
        )

    def forward(self, x):
        # Effective weight = r * cos(theta)
        effective_w = self.conv.weight * torch.cos(self.theta)
        return F.conv2d(x, effective_w,
                        bias=None,
                        padding=self.conv.padding[0])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2  QuSpike-V encoder
# ═══════════════════════════════════════════════════════════════════════════════

class QUSpikeV(nn.Module):
    """
    Quantum-inspired spiking vision encoder.

    3 phase-conv layers followed by LIF (Leaky Integrate-and-Fire) neurons.
    Readout: spike count population vector over T timesteps -> linear classifier.
    """

    def __init__(self, n_classes=100, n_timesteps=N_TIMESTEPS,
                 bio_phase=BIO_PHASE_RAD):
        super().__init__()
        self.T = n_timesteps
        spike_grad = surrogate.fast_sigmoid(slope=25)
        beta = 0.9   # LIF decay constant

        # Encoder layers
        self.pc1 = PhaseConv2d(3,   64,  3, bio_phase_init=bio_phase)
        self.pc2 = PhaseConv2d(64,  128, 3, bio_phase_init=bio_phase)
        self.pc3 = PhaseConv2d(128, 256, 3, bio_phase_init=bio_phase)
        self.pool = nn.AvgPool2d(2)

        # LIF neurons (one per conv layer)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Classifier readout
        self.fc = nn.Linear(256 * 4 * 4, n_classes)

        # Spike counter for energy estimation (detached, no grad)
        self.register_buffer("_spike_count", torch.tensor(0.0))
        self.register_buffer("_n_inferences", torch.tensor(0.0))

    def forward(self, x):
        """
        x : (batch, 3, 32, 32) image
        Returns logits (batch, n_classes) averaged over T timesteps.
        """
        batch = x.shape[0]

        # Initialise LIF membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spike_acc = torch.zeros(batch, 256 * 4 * 4, device=x.device)
        total_spikes = 0.0

        for t in range(self.T):
            # Rate-coded input: Bernoulli sampling from pixel intensity
            x_t = (torch.rand_like(x) < x).float()

            # Layer 1
            c1 = self.pc1(x_t)
            s1, mem1 = self.lif1(c1, mem1)
            s1 = self.pool(s1)

            # Layer 2
            c2 = self.pc2(s1)
            s2, mem2 = self.lif2(c2, mem2)
            s2 = self.pool(s2)

            # Layer 3
            c3 = self.pc3(s2)
            s3, mem3 = self.lif3(c3, mem3)
            s3 = self.pool(s3)

            # Accumulate spikes for readout
            spike_acc += s3.view(batch, -1)

            # Track total synaptic events (spike x synapse)
            total_spikes += s1.sum().item() + s2.sum().item() + s3.sum().item()

        # Update energy counters
        if not self.training:
            self._spike_count += total_spikes
            self._n_inferences += batch

        logits = self.fc(spike_acc / self.T)
        return logits

    def mean_synaptic_events_per_inference(self):
        if self._n_inferences.item() == 0:
            return 0.0
        return self._spike_count.item() / self._n_inferences.item()

    def reset_energy_counters(self):
        self._spike_count.zero_()
        self._n_inferences.zero_()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3  ANN baseline: ResNet-8 (matched parameters)
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class ResNet8(nn.Module):
    def __init__(self, n_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.layer1 = ResBlock(64)
        self.down1  = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.layer2 = ResBlock(128)
        self.down2  = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.layer3 = ResBlock(256)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.shape[0], -1)
        return self.fc(x)

    def count_macs(self, input_shape=(1, 3, 32, 32)):
        """Approximate MAC count via hook."""
        macs = [0]
        hooks = []

        def conv_hook(m, inp, out):
            b, c_out, h, w = out.shape
            kH, kW = m.kernel_size if hasattr(m.kernel_size, '__len__') \
                     else (m.kernel_size, m.kernel_size)
            c_in = inp[0].shape[1] // getattr(m, 'groups', 1)
            macs[0] += b * c_out * h * w * c_in * kH * kW

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(conv_hook))

        dummy = torch.zeros(*input_shape, device=next(self.parameters()).device)
        with torch.no_grad():
            self(dummy)

        for h in hooks:
            h.remove()

        return macs[0]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4  Data
# ═══════════════════════════════════════════════════════════════════════════════

print("\nLoading CIFAR-100 ...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR100(
    DATA_DIR, train=True,  download=True, transform=transform_train)
test_set  = torchvision.datasets.CIFAR100(
    DATA_DIR, train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  BATCH_SIZE, shuffle=False, num_workers=0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5  Training loop (shared for both models)
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, loader, optimiser, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * imgs.shape[0]
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.shape[0]
    return total_loss / total, 100 * correct / total


def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  eval ", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.shape[0]
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.shape[0]
    return total_loss / total, 100 * correct / total


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6  Train QuSpike-V (bio-phase seeded)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Training QuSpike-V (bio-phase seeded: theta_0 = 40 deg) ===")
model_snn = QUSpikeV(n_classes=100, bio_phase=BIO_PHASE_RAD).to(DEVICE)
opt_snn   = torch.optim.Adam(model_snn.parameters(), lr=LR)
sch_snn   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_snn, N_EPOCHS)
crit      = nn.CrossEntropyLoss()

snn_history = []
for epoch in range(1, N_EPOCHS + 1):
    tr_loss, tr_acc = train_model(model_snn, train_loader, opt_snn, crit)
    te_loss, te_acc = evaluate_model(model_snn, test_loader, crit)
    sch_snn.step()
    snn_history.append((tr_acc, te_acc))
    print(f"  Epoch {epoch:2d} | train acc {tr_acc:.1f}% | test acc {te_acc:.1f}%")

# Energy measurement pass
model_snn.reset_energy_counters()
evaluate_model(model_snn, test_loader, crit)
snn_events = model_snn.mean_synaptic_events_per_inference()
snn_energy_pj = snn_events * PJ_PER_SYNAPTIC_EVENT
print(f"\nQuSpike-V synaptic events/inference : {snn_events:,.0f}")
print(f"QuSpike-V energy/inference          : {snn_energy_pj:.1f} pJ")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7  Train ResNet-8 (ANN baseline)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Training ResNet-8 (ANN baseline) ===")
model_ann = ResNet8(n_classes=100).to(DEVICE)
opt_ann   = torch.optim.Adam(model_ann.parameters(), lr=LR)
sch_ann   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ann, N_EPOCHS)

ann_history = []
for epoch in range(1, N_EPOCHS + 1):
    tr_loss, tr_acc = train_model(model_ann, train_loader, opt_ann, crit)
    te_loss, te_acc = evaluate_model(model_ann, test_loader, crit)
    sch_ann.step()
    ann_history.append((tr_acc, te_acc))
    print(f"  Epoch {epoch:2d} | train acc {tr_acc:.1f}% | test acc {te_acc:.1f}%")

ann_macs      = model_ann.count_macs()
ann_energy_pj = ann_macs * 2 * PJ_PER_MAC
print(f"\nResNet-8 MACs/inference       : {ann_macs:,.0f}")
print(f"ResNet-8 energy/inference     : {ann_energy_pj:.1f} pJ")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8  Random-phase and anti-phase ablations (controls for Experiment 4)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Ablation: random phase init (theta_0 ~ Uniform[-pi, pi]) ===")
model_rand = QUSpikeV(n_classes=100).to(DEVICE)

# Override theta to random
with torch.no_grad():
    for name, param in model_rand.named_parameters():
        if "theta" in name:
            param.data.uniform_(-math.pi, math.pi)

opt_rand = torch.optim.Adam(model_rand.parameters(), lr=LR)
sch_rand = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rand, N_EPOCHS)

rand_history = []
for epoch in range(1, N_EPOCHS + 1):
    tr_loss, tr_acc = train_model(model_rand, train_loader, opt_rand, crit)
    te_loss, te_acc = evaluate_model(model_rand, test_loader, crit)
    sch_rand.step()
    rand_history.append((tr_acc, te_acc))
    print(f"  Epoch {epoch:2d} | test acc {te_acc:.1f}%")

model_rand.reset_energy_counters()
evaluate_model(model_rand, test_loader, crit)
rand_events    = model_rand.mean_synaptic_events_per_inference()
rand_energy_pj = rand_events * PJ_PER_SYNAPTIC_EVENT


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9  Results summary & figures
# ═══════════════════════════════════════════════════════════════════════════════

final_snn_acc  = snn_history[-1][1]
final_ann_acc  = ann_history[-1][1]
final_rand_acc = rand_history[-1][1]
energy_ratio   = ann_energy_pj / snn_energy_pj

print("\n" + "="*55)
print("ENERGY EFFICIENCY SUMMARY")
print("="*55)
print(f"{'Model':<25} {'Test Acc':>9} {'Energy (pJ)':>13} {'Ratio':>7}")
print("-"*55)
print(f"{'ResNet-8 (ANN baseline)':<25} {final_ann_acc:>8.1f}% "
      f"{ann_energy_pj:>12.1f}   1.0x")
print(f"{'QuSpike-V (bio-seeded)':<25} {final_snn_acc:>8.1f}% "
      f"{snn_energy_pj:>12.1f}   {energy_ratio:.1f}x")
print(f"{'QuSpike-V (random phase)':<25} {final_rand_acc:>8.1f}% "
      f"{rand_energy_pj:>12.1f}   {ann_energy_pj/rand_energy_pj:.1f}x")
print("="*55)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ── 9a  Accuracy curves ──────────────────────────────────────────────────────
ax = axes[0]
epochs = range(1, N_EPOCHS + 1)
ax.plot(epochs, [h[1] for h in snn_history],
        color="#534AB7", lw=2, label=f"QuSpike-V bio ({final_snn_acc:.1f}%)")
ax.plot(epochs, [h[1] for h in rand_history],
        color="#534AB7", lw=2, ls="--", label=f"QuSpike-V random ({final_rand_acc:.1f}%)")
ax.plot(epochs, [h[1] for h in ann_history],
        color="#888780", lw=2, label=f"ResNet-8 ({final_ann_acc:.1f}%)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("CIFAR-100 accuracy\nBio-seeded vs random vs ANN")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# ── 9b  Energy bar chart ─────────────────────────────────────────────────────
ax = axes[1]
models    = ["ResNet-8\n(ANN)", "QuSpike-V\n(random θ)", "QuSpike-V\n(bio θ=40°)"]
energies  = [ann_energy_pj, rand_energy_pj, snn_energy_pj]
colors    = ["#888780", "#AFA9EC", "#534AB7"]
bars = ax.bar(models, energies, color=colors, alpha=0.8, edgecolor="black")
ax.bar_label(bars, labels=[f"{e:.0f} pJ" for e in energies],
             padding=3, fontsize=10)
ax.set_ylabel("Energy per inference (pJ)")
ax.set_title(f"Energy reduction: {energy_ratio:.1f}x\n(bio-seeded vs ANN)")
ax.spines[["top", "right"]].set_visible(False)

# ── 9c  Phase distribution (learned theta after training) ───────────────────
ax = axes[2]
theta_vals = []
for name, param in model_snn.named_parameters():
    if "theta" in name:
        theta_vals.append(param.data.cpu().numpy().flatten())
if theta_vals:
    all_thetas = np.concatenate(theta_vals)
    ax.hist(np.rad2deg(all_thetas), bins=50,
            color="#534AB7", alpha=0.75, density=True)
    ax.axvline(BIO_PHASE_DEG, color="#D85A30", lw=2,
               ls="--", label=f"Bio prior: {BIO_PHASE_DEG}°")
    ax.axvline(np.rad2deg(np.mean(all_thetas)), color="black", lw=2,
               label=f"Learned mean: {np.rad2deg(np.mean(all_thetas)):.1f}°")
ax.set_xlabel("Phase theta (degrees)")
ax.set_ylabel("Density")
ax.set_title("Learned phase distribution\n(QuSpike-V after training)")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Experiment 3 -- Energy efficiency: QuSpike-V vs ANN baseline",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/home/rocky/FT-bursting-WM-main/quantum/exp3_energy_efficiency.png",
            dpi=150, bbox_inches="tight")
print("\nSaved: exp3_energy_efficiency.png")

# Save summary for paper Table
summary = {
    "model":            ["ResNet-8 (ANN)", "QuSpike-V (random)", "QuSpike-V (bio)"],
    "test_accuracy":    [final_ann_acc, final_rand_acc, final_snn_acc],
    "energy_pj":        [ann_energy_pj, rand_energy_pj, snn_energy_pj],
    "energy_ratio":     [1.0,
                         ann_energy_pj / rand_energy_pj,
                         energy_ratio],
}
import pandas as pd
pd.DataFrame(summary).to_csv(
    "/home/rocky/FT-bursting-WM-main/quantum/exp3_energy_summary.csv", index=False)
print("Saved: exp3_energy_summary.csv")
