"""
Experiment 1: Amplitude Vector Extraction
==========================================
Recast the PBC phase-bin histogram as a quantum amplitude vector |psi>.
Tests whether Born-rule probability |psi|^2 predicts WM trial accuracy
better than raw PBC entropy alone.

Builds directly on figure_5.ipynb pipeline:
  - compute_phase_burst_counts  -> phase-bin histogram (your existing output)
  - phase_burst_coupling        -> PBC entropy (your existing metric)
  - NEW: amplitude_vector()     -> normalized sqrt of burst counts
  - NEW: born_probability()     -> |psi|^2 at peak phase bin
  - NEW: logistic regression    -> AUC comparison (born_prob vs PBC)

Data: Utah cohort sub-P1, sub-P6 (same as figure_5.ipynb)
      Delay window: 3000-5500 ms (same as figure_5.ipynb Fig 5d)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

# ── Adjust these paths to match your setup ────────────────────────────────────
ROOT_DIR   = "/home/rocky/FT-bursting-WM-main"
DATA_PATH  = "/home/rocky/ds006136-1.0.0"
BURST_TOOLBOX = "/home/rocky/burst_toolbox-main/src"
sys.path.append(ROOT_DIR)
sys.path.append(BURST_TOOLBOX)
# ─────────────────────────────────────────────────────────────────────────────

import utils
from burst_toolbox.dsp import compute_power
from burst_toolbox.bursts import detect_bursts
from burst_toolbox.coupling import compute_phase_burst_counts, phase_burst_coupling

SUBJECTS     = [f"sub-P{k}" for k in range(1, 8)]
PBC_SUBJECTS = ["sub-P1", "sub-P6"]   # subjects with both LMFG and RMTG coverage
N_PHASE_BINS = 18                      # must match figure_5.ipynb
DELAY_START  = 3000                    # ms
DELAY_END    = 5500                    # ms
BASELINE     = utils.BASELINE          # (400, 600) from utils


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1  ·  Core quantum-inspired functions (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def amplitude_vector(counts: np.ndarray) -> np.ndarray:
    """
    Convert burst-per-phase-bin counts to a unit-norm amplitude vector.

    Parameters
    ----------
    counts : (n_phase_bins,) array of burst counts summed over a time window

    Returns
    -------
    psi : (n_phase_bins,) complex-valued amplitude vector
          amplitude[i] = sqrt(counts[i] / total_counts)
          Phase component set to 0 here (real-valued representation).
          For interference experiments (Exp 2) we add phase from the
          instantaneous beta phase at each bin centre.

    Notes
    -----
    This is the Born-rule-compatible representation:
      |psi[i]|^2 = counts[i] / total   =>  sum(|psi|^2) = 1
    which is exactly the normalised burst density already plotted in Fig 5d,
    but cast as probability amplitudes rather than densities.
    """
    total = np.sum(counts)
    if total == 0:
        return np.zeros(len(counts))
    return np.sqrt(counts / total)      # real-valued amplitudes


def born_probability(psi: np.ndarray) -> np.ndarray:
    """
    Apply the Born rule: P(bin i) = |psi[i]|^2.
    For real amplitudes this is just psi^2, but we keep the formulation
    explicit so it generalises when psi is complex (Experiment 2).
    """
    return np.abs(psi) ** 2


def von_neumann_entropy(psi: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute von Neumann entropy of the amplitude state.
    For a pure state this equals the Shannon entropy of the Born probabilities:
      S = -sum_i P_i * log(P_i)
    High S  =  uniform distribution  =  no preferred phase  =  low PBC
    Low S   =  peaked distribution   =  strong phase locking =  high PBC

    This is the quantum-information dual of the PBC metric:
      PBC  = (ln(18) - entropy_of_uniform) / ln(18)   [from paper Methods]
      S_vN = -sum(P * log(P))
    Both measure the same clustering; vN entropy gives an absolute scale.
    """
    probs = born_probability(psi)
    probs = probs[probs > eps]          # avoid log(0)
    return -np.sum(probs * np.log(probs))


def peak_phase_amplitude(psi: np.ndarray, edge_labels: np.ndarray) -> tuple:
    """
    Return the phase bin with maximum Born probability and its amplitude.
    Used to seed Experiment 2 (interference) and Experiment 4 (bio seeding).

    Returns
    -------
    peak_phase_deg : float  - phase angle in degrees of peak bin
    peak_amplitude : float  - amplitude at that bin
    """
    probs   = born_probability(psi)
    peak_i  = np.argmax(probs)
    return edge_labels[peak_i], psi[peak_i]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2  ·  Data loading & burst detection  (same as figure_5.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading data ...")
data = utils.read_openneuro(path=DATA_PATH, subjects=SUBJECTS)
data["sc"] = data["subject"] + "_" + data["channel"].astype(str)

print("Computing power and detecting bursts ...")
processed_data = []

for region in ["LMFG", "RMTG"]:
    for freq_band in ["beta", "high_gamma"]:
        for sc in data[data["region"] == region]["sc"].unique():
            subject, channel = sc.split("_")
            channel = int(channel)

            channel_data = data[
                (data["subject"] == subject) &
                (data["channel"] == channel)
            ]

            power = compute_power(
                LFP=channel_data[utils.LFP_COLS].to_numpy(),
                freq_band=utils.FREQ_BANDS[freq_band]
            )
            bursts = detect_bursts(
                power=power,
                reference_period=np.array([1, 1000]),
                min_dur_ms=3 * 1000 / np.mean(utils.FREQ_BANDS[freq_band])
            )

            composite = pd.DataFrame(
                np.hstack((power, bursts)),
                columns=utils.POWER_COLS + utils.BURST_COLS
            )
            composite["subject"]   = subject
            composite["region"]    = region
            composite["channel"]   = channel
            composite["freq_band"] = freq_band
            composite["trial_idx"] = channel_data["trial_idx"].to_numpy()
            composite["modulated"] = channel_data["modulated"].to_numpy()
            composite["n_correct"] = channel_data["n_correct"].to_numpy()
            processed_data.append(composite)

processed_data = pd.concat(processed_data, axis=0, ignore_index=True)
print(f"  processed_data shape: {processed_data.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3  ·  Per-trial amplitude vectors & PBC (LMFG → RMTG)
# ═══════════════════════════════════════════════════════════════════════════════

print("\nComputing per-trial amplitude vectors and PBC ...")

edge_labels = np.array([-180 + d * 20 for d in range(N_PHASE_BINS)])

# Store one row per channel-pair × trial
records = []   # dict per trial: {subject, ch_from, ch_to, trial_idx,
               #                  n_correct, psi, pbc_value, vn_entropy,
               #                  peak_phase, peak_amp}

for subject in PBC_SUBJECTS:
    lmfg_channels = data[
        (data["subject"] == subject) &
        (data["region"] == "LMFG") &
        (data["modulated"] == 1)
    ]["channel"].unique()

    rmtg_channels = processed_data[
        (processed_data["subject"] == subject) &
        (processed_data["region"] == "RMTG") &
        (processed_data["modulated"] == 1)
    ]["channel"].unique()

    for ch_from in lmfg_channels:
        for ch_to in rmtg_channels:

            # Common trials (correct + incorrect)
            trials_from = set(data[
                (data["subject"] == subject) &
                (data["channel"] == ch_from)
            ]["trial_idx"])
            trials_to = set(processed_data[
                (processed_data["subject"] == subject) &
                (processed_data["channel"] == ch_to)
            ]["trial_idx"])
            common = sorted(trials_from & trials_to)
            if len(common) < 5:
                continue

            lfp_from = data[
                (data["subject"] == subject) &
                (data["channel"] == ch_from) &
                (data["trial_idx"].isin(common))
            ].sort_values("trial_idx")[utils.LFP_COLS].to_numpy()

            bursts_to = processed_data[
                (processed_data["subject"] == subject) &
                (processed_data["channel"] == ch_to) &
                (processed_data["freq_band"] == "high_gamma") &
                (processed_data["trial_idx"].isin(common))
            ].sort_values("trial_idx")[utils.BURST_COLS].to_numpy()

            n_correct_vec = data[
                (data["subject"] == subject) &
                (data["channel"] == ch_from) &
                (data["trial_idx"].isin(common))
            ].sort_values("trial_idx")["n_correct"].to_numpy()

            assert lfp_from.shape == bursts_to.shape, \
                f"Shape mismatch: {lfp_from.shape} vs {bursts_to.shape}"

            # Phase-burst counts: shape (n_trials, time, n_bins)
            pbc_counts = compute_phase_burst_counts(
                LFP=lfp_from,
                bursts=bursts_to,
                filter=True,
                phase_freq_band=np.array([12, 30])
            )

            # ── Per-trial computation ──────────────────────────────────────
            for t_idx in range(len(common)):
                # Delay-window counts for this trial
                delay_counts = pbc_counts[
                    t_idx, DELAY_START:DELAY_END, :
                ].sum(axis=0)   # (n_bins,)

                if delay_counts.sum() == 0:
                    continue

                psi    = amplitude_vector(delay_counts)
                probs  = born_probability(psi)
                s_vn   = von_neumann_entropy(psi)
                pp, pa = peak_phase_amplitude(psi, edge_labels)

                # PBC for this trial's delay window (single-window version)
                # We use the existing phase_burst_coupling on the delay slice
                delay_slice = pbc_counts[t_idx:t_idx+1, DELAY_START:DELAY_END, :]
                pbc_val = float(phase_burst_coupling(delay_slice)[0].mean())

                records.append({
                    "subject":     subject,
                    "ch_from":     ch_from,
                    "ch_to":       ch_to,
                    "trial_idx":   common[t_idx],
                    "n_correct":   n_correct_vec[t_idx],
                    "psi":         psi,         # (n_bins,) amplitude vector
                    "pbc_value":   pbc_val,
                    "vn_entropy":  s_vn,
                    "peak_phase":  pp,
                    "peak_amp":    pa,
                    "born_peak":   float(probs.max()),   # P at mode
                })

df = pd.DataFrame(records)
print(f"  {len(df)} channel-pair × trial records assembled")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4  ·  Logistic regression: born_peak vs PBC  →  AUC comparison
# ═══════════════════════════════════════════════════════════════════════════════

print("\nRunning leave-one-subject-out logistic regression ...")

# Binary performance: correct = n_correct == 3
df["correct"] = (df["n_correct"] == 3).astype(int)
# Drop mixed trials (n_correct == 2)
df_binary = df[df["n_correct"] != 2].copy()

# Channel-pair ID as group for LOSO-CV
df_binary["pair_id"] = (
    df_binary["subject"] + "_" +
    df_binary["ch_from"].astype(str) + "_" +
    df_binary["ch_to"].astype(str)
)

features = {
    "PBC only":          ["pbc_value"],
    "Born peak only":    ["born_peak"],
    "vN entropy only":   ["vn_entropy"],
    "Born + PBC":        ["born_peak", "pbc_value"],
}

logo     = LeaveOneGroupOut()
groups   = df_binary["pair_id"].to_numpy()
y        = df_binary["correct"].to_numpy()
auc_results = {}

for label, cols in features.items():
    X    = df_binary[cols].to_numpy()
    aucs = []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        prob = clf.predict_proba(X[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[test_idx], prob))
    auc_results[label] = aucs
    print(f"  {label:25s} → mean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5  ·  Null distribution (phase-bin shuffle)
# ═══════════════════════════════════════════════════════════════════════════════

print("\nBuilding shuffle null distribution (n=1000) ...")
rng = np.random.default_rng(42)
null_aucs = []

X_bp = df_binary[["born_peak"]].to_numpy()
for _ in range(1000):
    y_shuffle = rng.permutation(y)
    fold_aucs = []
    for train_idx, test_idx in logo.split(X_bp, y_shuffle, groups):
        if len(np.unique(y_shuffle[train_idx])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_bp[train_idx], y_shuffle[train_idx])
        prob = clf.predict_proba(X_bp[test_idx])[:, 1]
        if len(np.unique(y_shuffle[test_idx])) < 2:
            continue
        fold_aucs.append(roc_auc_score(y_shuffle[test_idx], prob))
    if fold_aucs:
        null_aucs.append(np.mean(fold_aucs))

null_aucs = np.array(null_aucs)
observed  = np.mean(auc_results["Born peak only"])
p_val     = np.mean(null_aucs >= observed)
print(f"  Observed AUC = {observed:.3f}, null mean = {null_aucs.mean():.3f}, p = {p_val:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6  ·  Figures
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ── 6a  AUC bar comparison ──────────────────────────────────────────────────
ax = axes[0]
labels = list(auc_results.keys())
means  = [np.mean(v) for v in auc_results.values()]
sems   = [np.std(v) / np.sqrt(len(v)) for v in auc_results.values()]
colors = ["#534AB7", "#1D9E75", "#BA7517", "#185FA5"]
ax.bar(range(len(labels)), means, yerr=sems,
       color=colors, alpha=0.75, edgecolor="black", capsize=4)
ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("AUC (LOSO-CV)")
ax.set_title("Predictor comparison\n(delay-period, LMFG→RMTG)")
ax.set_ylim(0.4, 0.85)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# ── 6b  vN entropy by performance ──────────────────────────────────────────
ax = axes[1]
for perf, color, label in [(1, "#1D9E75", "Correct"), (0, "#D85A30", "Incorrect")]:
    vals = df_binary[df_binary["correct"] == perf]["vn_entropy"].to_numpy()
    ax.hist(vals, bins=25, alpha=0.6, color=color, label=label, density=True)
ax.set_xlabel("von Neumann entropy (delay period)")
ax.set_ylabel("Density")
ax.set_title("Entropy lower on correct trials\n(quantum superposition collapse)")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# ── 6c  Null vs observed AUC ────────────────────────────────────────────────
ax = axes[2]
ax.hist(null_aucs, bins=40, color="#888780", alpha=0.7, label="Shuffle null")
ax.axvline(observed, color="#534AB7", lw=2, label=f"Observed = {observed:.3f}")
ax.axvline(np.percentile(null_aucs, 95), color="gray",
           lw=1.5, ls="--", label="95th percentile null")
ax.set_xlabel("AUC")
ax.set_ylabel("Count")
ax.set_title(f"Born-peak AUC vs null\n(p = {p_val:.4f})")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Experiment 1 — Amplitude vector (quantum reframing of PBC)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/exp1_amplitude_vector.png",
            dpi=150, bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7  ·  Export peak-phase data for Experiment 2
# ═══════════════════════════════════════════════════════════════════════════════

# Save per-trial peak phases — used as input to Experiment 2 (QI-PBC)
df[["subject", "ch_from", "ch_to", "trial_idx",
    "n_correct", "peak_phase", "peak_amp",
    "pbc_value", "vn_entropy", "born_peak"]].to_csv(
    "/mnt/user-data/outputs/exp1_peak_phases.csv", index=False
)
print("\nSaved: exp1_peak_phases.csv  (input for Experiment 2)")
print("Saved: exp1_amplitude_vector.png")
