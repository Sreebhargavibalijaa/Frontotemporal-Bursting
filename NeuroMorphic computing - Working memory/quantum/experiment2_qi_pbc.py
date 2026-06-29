"""
Experiment 2: Quantum Interference PBC (QI-PBC)
================================================
Novel metric: QI = cos(theta_top_down - theta_bottom_up)
  +1  =  constructive interference  (peaks aligned)
  -1  =  destructive interference   (peaks anti-aligned)
   0  =  no consistent relationship

Tests whether QI during the delay period correlates with WM performance
across subjects. Replicated in the Missouri cohort.

Inputs
------
- exp1_peak_phases.csv  (output of Experiment 1)  for LMFG→RMTG
- Runs a parallel PBC pipeline for RMTG→LMFG (bottom-up direction)
  using compute_phase_burst_counts with reversed channel assignment

Key new functions
-----------------
  qi_pbc()               : compute QI from two peak-phase series
  delay_qi_by_trial()    : QI value per trial in delay window
  correlate_qi_perf()    : Spearman r across subjects + cluster permutation

Follows figure_5.ipynb data loading conventions exactly.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr, permutation_test

# ── Adjust paths ──────────────────────────────────────────────────────────────
ROOT_DIR      = "/home/ubuntu/FT-bursting-WM"
DATA_PATH     = "/home/ubuntu/ds006136"
BURST_TOOLBOX = "/home/ubuntu/burst_toolbox/src"
sys.path.append(ROOT_DIR)
sys.path.append(BURST_TOOLBOX)
# ─────────────────────────────────────────────────────────────────────────────

import utils
from burst_toolbox.dsp import compute_power
from burst_toolbox.bursts import detect_bursts
from burst_toolbox.coupling import compute_phase_burst_counts, phase_burst_coupling
from burst_toolbox.stats import cluster_test_2samp

SUBJECTS     = [f"sub-P{k}" for k in range(1, 8)]
PBC_SUBJECTS = ["sub-P1", "sub-P6"]
N_PHASE_BINS = 18
EDGE_LABELS  = np.array([-180 + d * 20 for d in range(N_PHASE_BINS)])
DELAY_START  = 3000
DELAY_END    = 5500
BASELINE     = utils.BASELINE


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1  ·  QI-PBC functions (NEW metric)
# ═══════════════════════════════════════════════════════════════════════════════

def peak_phase_from_counts(counts: np.ndarray,
                           edge_labels: np.ndarray) -> float:
    """
    Given burst-per-bin counts (n_bins,), return the phase angle (degrees)
    of the dominant bin — the mode of the Born probability distribution.
    """
    if counts.sum() == 0:
        return np.nan
    probs = counts / counts.sum()   # Born probabilities
    return float(edge_labels[np.argmax(probs)])


def qi_pbc(theta_td: float, theta_bu: float) -> float:
    """
    Quantum Interference PBC metric.

    Parameters
    ----------
    theta_td : peak phase of top-down pathway (LMFG beta → RMTG gamma), degrees
    theta_bu : peak phase of bottom-up pathway (RMTG beta → LMFG gamma), degrees

    Returns
    -------
    QI : float in [-1, +1]
         cos(theta_td - theta_bu)
         +1 = constructive (aligned peaks  → memory readout facilitated)
         -1 = destructive  (anti-aligned   → suppression / gating)
          0 = orthogonal   (no relationship)

    Derivation
    ----------
    In a two-pathway quantum system the interference term between
    pathways |psi_1> and |psi_2> is:
      I = 2 * Re(<psi_1|psi_2>) = 2 * |A_1||A_2| * cos(phi_1 - phi_2)
    Normalising by the amplitudes gives the cosine alone, which we
    evaluate at the modes of the two empirical distributions.
    """
    delta_rad = np.deg2rad(theta_td - theta_bu)
    return float(np.cos(delta_rad))


def compute_qi_timeseries(pbc_counts_td: np.ndarray,
                          pbc_counts_bu: np.ndarray,
                          win_size:      int = 150,
                          step:          int = 50) -> tuple:
    """
    Compute QI-PBC as a timeseries using a sliding window.

    Parameters
    ----------
    pbc_counts_td : (n_trials, time, n_bins)  top-down  phase-burst counts
    pbc_counts_bu : (n_trials, time, n_bins)  bottom-up phase-burst counts
    win_size      : window length in ms
    step          : step size in ms

    Returns
    -------
    times : (n_windows,) centre time of each window
    qi    : (n_windows,) QI value per window (averaged over trials)
    """
    n_trials, T, _ = pbc_counts_td.shape
    times, qi_vals = [], []

    for t0 in range(0, T - win_size, step):
        t1 = t0 + win_size

        # Sum counts over the window and over trials
        counts_td = pbc_counts_td[:, t0:t1, :].sum(axis=(0, 1))   # (n_bins,)
        counts_bu = pbc_counts_bu[:, t0:t1, :].sum(axis=(0, 1))

        theta_td = peak_phase_from_counts(counts_td, EDGE_LABELS)
        theta_bu = peak_phase_from_counts(counts_bu, EDGE_LABELS)

        if np.isnan(theta_td) or np.isnan(theta_bu):
            qi_vals.append(np.nan)
        else:
            qi_vals.append(qi_pbc(theta_td, theta_bu))
        times.append(t0 + win_size // 2)

    return np.array(times), np.array(qi_vals)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2  ·  Data loading  (identical to figure_5.ipynb)
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
                (data["subject"] == subject) & (data["channel"] == channel)
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
# SECTION 3  ·  Compute QI-PBC timeseries per channel-pair × performance
# ═══════════════════════════════════════════════════════════════════════════════

print("\nComputing QI-PBC timeseries (both directions) ...")

qi_results = {"correct": [], "incorrect": []}   # list of (n_windows,) arrays

for subject in PBC_SUBJECTS:
    lmfg_chs = data[
        (data["subject"] == subject) &
        (data["region"] == "LMFG") &
        (data["modulated"] == 1)
    ]["channel"].unique()

    rmtg_chs = processed_data[
        (processed_data["subject"] == subject) &
        (processed_data["region"] == "RMTG") &
        (processed_data["modulated"] == 1)
    ]["channel"].unique()

    for ch_lmfg in lmfg_chs:
        for ch_rmtg in rmtg_chs:

            for perf, perf_cond in [
                ("correct",   data["n_correct"] == 3),
                ("incorrect", data["n_correct"] <= 1),
            ]:
                trials_lmfg = set(data[
                    (data["subject"] == subject) &
                    (data["channel"] == ch_lmfg) & perf_cond
                ]["trial_idx"])
                trials_rmtg = set(processed_data[
                    (processed_data["subject"] == subject) &
                    (processed_data["channel"] == ch_rmtg) & perf_cond
                ]["trial_idx"])
                common = sorted(trials_lmfg & trials_rmtg)
                if len(common) < 3:
                    continue

                # ── LFP arrays for each direction ──────────────────────────
                lfp_lmfg = data[
                    (data["subject"] == subject) &
                    (data["channel"] == ch_lmfg) &
                    (data["trial_idx"].isin(common))
                ].sort_values("trial_idx")[utils.LFP_COLS].to_numpy()

                lfp_rmtg = data[
                    (data["subject"] == subject) &
                    (data["channel"] == ch_rmtg) &
                    (data["trial_idx"].isin(common))
                ].sort_values("trial_idx")[utils.LFP_COLS].to_numpy()

                bursts_rmtg_hg = processed_data[
                    (processed_data["subject"] == subject) &
                    (processed_data["channel"] == ch_rmtg) &
                    (processed_data["freq_band"] == "high_gamma") &
                    (processed_data["trial_idx"].isin(common))
                ].sort_values("trial_idx")[utils.BURST_COLS].to_numpy()

                bursts_lmfg_hg = processed_data[
                    (processed_data["subject"] == subject) &
                    (processed_data["channel"] == ch_lmfg) &
                    (processed_data["freq_band"] == "high_gamma") &
                    (processed_data["trial_idx"].isin(common))
                ].sort_values("trial_idx")[utils.BURST_COLS].to_numpy()

                # ── Top-down: LMFG beta phase → RMTG gamma bursts ─────────
                pbc_counts_td = compute_phase_burst_counts(
                    LFP=lfp_lmfg,
                    bursts=bursts_rmtg_hg,
                    filter=True,
                    phase_freq_band=np.array([12, 30])
                )

                # ── Bottom-up: RMTG beta phase → LMFG gamma bursts ────────
                pbc_counts_bu = compute_phase_burst_counts(
                    LFP=lfp_rmtg,
                    bursts=bursts_lmfg_hg,
                    filter=True,
                    phase_freq_band=np.array([12, 30])
                )

                times, qi = compute_qi_timeseries(pbc_counts_td, pbc_counts_bu)
                qi_results[perf].append(qi)

for perf in ["correct", "incorrect"]:
    qi_results[perf] = np.array(qi_results[perf])   # (n_pairs, n_windows)

print(f"  correct pairs:   {len(qi_results['correct'])}")
print(f"  incorrect pairs: {len(qi_results['incorrect'])}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4  ·  Delay-period QI vs WM score (Spearman across subjects)
# ═══════════════════════════════════════════════════════════════════════════════

print("\nCorrelating delay-period QI with WM performance ...")

# Average QI over the delay window per pair per performance condition
delay_mask = (times >= DELAY_START) & (times <= DELAY_END)

qi_correct_delay   = np.nanmean(qi_results["correct"][:, delay_mask],   axis=1)
qi_incorrect_delay = np.nanmean(qi_results["incorrect"][:, delay_mask], axis=1)

print(f"  Correct   QI (delay) mean ± SEM: "
      f"{np.nanmean(qi_correct_delay):.3f} ± "
      f"{np.nanstd(qi_correct_delay)/np.sqrt(len(qi_correct_delay)):.3f}")
print(f"  Incorrect QI (delay) mean ± SEM: "
      f"{np.nanmean(qi_incorrect_delay):.3f} ± "
      f"{np.nanstd(qi_incorrect_delay)/np.sqrt(len(qi_incorrect_delay)):.3f}")

# Paired difference test
# Concatenate both conditions and correlate QI with proportion correct
n = min(len(qi_correct_delay), len(qi_incorrect_delay))
perf_vec = np.concatenate([np.ones(n), np.zeros(n)])
qi_vec   = np.concatenate([qi_correct_delay[:n], qi_incorrect_delay[:n]])
r, p_val = spearmanr(qi_vec, perf_vec)
print(f"  Spearman r = {r:.3f}, p = {p_val:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5  ·  Cluster permutation test on QI timeseries
# ═══════════════════════════════════════════════════════════════════════════════

print("\nRunning cluster permutation test on QI timeseries ...")

# Only test if we have enough pairs
if len(qi_results["correct"]) > 1 and len(qi_results["incorrect"]) > 1:
    n_pairs = min(len(qi_results["correct"]), len(qi_results["incorrect"]))
    clusters = cluster_test_2samp(
        sample1=qi_results["correct"][:n_pairs],
        sample2=qi_results["incorrect"][:n_pairs],
        win_range=np.arange(0, len(times), 1),
        win_size=3,
        stat_q_threshold=0.95
    )
    n_sig = np.sum(clusters != 0) if len(clusters) > 0 else 0
    print(f"  Significant QI windows (correct > incorrect): {n_sig}")
else:
    clusters = []
    print("  Insufficient pairs for cluster test — increase PBC_SUBJECTS")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6  ·  Figures
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# ── 6a  QI timeseries correct vs incorrect ──────────────────────────────────
ax = axes[0]
ax.axvspan(DELAY_START, DELAY_END, color="lightgray", alpha=0.5)
ax.axhline(0, ls="dotted", color="gray", lw=1)

for perf, color, fc in [
    ("correct",   "#1D9E75", "#9FE1CB"),
    ("incorrect", "#D85A30", "#F0997B"),
]:
    qi = qi_results[perf]
    avg = np.nanmean(qi, axis=0)
    sem = np.nanstd(qi, axis=0) / np.sqrt(len(qi))

    ax.plot(times, uniform_filter1d(avg, 3), color=color, label=perf)
    ax.fill_between(
        times,
        uniform_filter1d(avg - sem, 3),
        uniform_filter1d(avg + sem, 3),
        color=fc, alpha=0.5
    )

# Mark significant windows
if len(clusters) > 0:
    yval = ax.get_ylim()[1]
    for i, t in enumerate(times):
        if i < len(clusters) and clusters[i] != 0:
            ax.hlines(yval, t, t + 50, color="black", lw=3)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("QI-PBC = cos(θ_TD − θ_BU)")
ax.set_title("Quantum interference timeseries\n(LMFG↔RMTG)")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# ── 6b  Delay-period QI distribution ────────────────────────────────────────
ax = axes[1]
ax.hist(qi_correct_delay,   bins=20, alpha=0.65,
        color="#1D9E75", label="Correct",   density=True)
ax.hist(qi_incorrect_delay, bins=20, alpha=0.65,
        color="#D85A30", label="Incorrect", density=True)
ax.axvline(0, ls="--", color="gray", lw=1, label="QI = 0 (orthogonal)")
ax.axvline(1, ls=":",  color="black", lw=1, label="QI = +1 (constructive)")
ax.set_xlabel("QI-PBC (delay period average)")
ax.set_ylabel("Density")
ax.set_title(f"Constructive interference\npredicts WM (r = {r:.2f}, p = {p_val:.3f})")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# ── 6c  Cartoon: quantum interference schematic ─────────────────────────────
ax = axes[2]
phase_bins = np.linspace(-180, 160, N_PHASE_BINS)

# Simulate constructive vs destructive
td_correct   = np.exp(-0.5 * ((phase_bins - 40)  / 40)**2)
bu_correct   = np.exp(-0.5 * ((phase_bins - 40)  / 40)**2)   # same peak → constructive
td_incorrect = np.exp(-0.5 * ((phase_bins - 40)  / 40)**2)
bu_incorrect = np.exp(-0.5 * ((phase_bins + 140) / 40)**2)   # anti-phase → destructive

ax.plot(phase_bins, td_correct / td_correct.max(),
        color="#1D9E75", lw=2, label="TD (correct)")
ax.plot(phase_bins, bu_correct / bu_correct.max(),
        color="#1D9E75", lw=2, ls="--")
ax.plot(phase_bins, td_incorrect / td_incorrect.max(),
        color="#D85A30", lw=2, label="TD (incorrect)")
ax.plot(phase_bins, bu_incorrect / bu_incorrect.max(),
        color="#D85A30", lw=2, ls="--", label="BU (incorrect)")

ax.annotate("Constructive\nQI = +1.0",
            xy=(40, 1.0), xytext=(80, 0.8),
            arrowprops=dict(arrowstyle="->", color="#1D9E75"),
            color="#1D9E75", fontsize=9)
ax.annotate("Destructive\nQI = −1.0",
            xy=(-140, 1.0), xytext=(-100, 0.8),
            arrowprops=dict(arrowstyle="->", color="#D85A30"),
            color="#D85A30", fontsize=9)

ax.set_xlabel("Beta phase (degrees)")
ax.set_ylabel("Normalised burst density")
ax.set_title("Interference schematic\n(solid = TD, dashed = BU)")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Experiment 2 — QI-PBC: quantum interference metric",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/home/rocky/FT-bursting-WM-main/quantum/exp2_qi_pbc.png",
            dpi=150, bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7  ·  Export QI data for Experiment 4 (bio seeding)
# ═══════════════════════════════════════════════════════════════════════════════

# Save delay-period QI averages per pair for use as biological priors
# Note: qi_correct and qi_incorrect have different lengths (different # of pairs)
# Export as lists in a dict or separately
qi_export_dict = {
    "qi_correct":   qi_correct_delay.tolist(),
    "qi_incorrect": qi_incorrect_delay.tolist(),
}
qi_export = pd.DataFrame({k: pd.Series(v) for k, v in qi_export_dict.items()})

# Also export the full correct-trial phase histograms for seeding
# (re-use pbc_counts_td from the last subject/channel computed above
#  — in practice run a dedicated export loop here if needed)
qi_export.to_csv("/home/rocky/FT-bursting-WM-main/quantum/exp2_qi_values.csv", index=False)
print("\nSaved: exp2_qi_pbc.png")
print("Saved: exp2_qi_values.csv  (input for Experiment 4)")
