"""
anticipatory_dbs.py
====================
Anticipatory Closed-Loop Deep Brain Stimulation
via Phase-Burst Coupling (PBC) Trajectory Forecasting

Pipeline:
  1. Load data via utils.read_openneuro (same as figure_4)
  2. Compute power + detect bursts per channel/region/freq_band (same as figure_4)
  3. Extract PBC frequency signatures per trial/channel
  4. Build a healthy-state cluster manifold (PCA + GMM)
  5. Fit a temporal forecasting model (AR fallback or TCN if PyTorch available)
  6. At inference time: predict future PBC state; if predicted state
     falls inside the healthy cluster → no stimulation needed;
     if it falls outside → trigger DBS and compute LFP drive
     to recover PBC at the target time-point.

Data schema (OpenNeuro ds006136, 7 subjects):
  - utils.read_openneuro  → DataFrame with LFP_0..LFP_6497, subject, region,
                            channel, modulated, trial_idx, n_correct
  - utils.LFP_COLS        → list of LFP column names
  - utils.BURST_COLS      → list of burst-rate column names
  - utils.FREQ_BANDS      → dict {"beta": (lo, hi), "high_gamma": (lo, hi)}
  - utils.CUTOFF          → int, last valid timepoint
  - utils.PLOTTING        → dict with colour/label info
  - burst_toolbox.dsp.compute_power
  - burst_toolbox.bursts.detect_bursts
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# Optional heavy deps — imported lazily so the rest of the module loads cleanly
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("[WARNING] umap-learn not installed. Falling back to PCA for manifold.")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. Using numpy AR fallback forecaster.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PBC Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

class PBCFeatureExtractor:
    """
    Extracts a compact PBC frequency signature vector from raw LFP.

    For each (beta_channel, gamma_channel) pair the signature is:
        [preferred_phase, resultant_length, mean_burst_density, peak_phase_bin]

    Parameters
    ----------
    fs : float
        Sampling rate in Hz (1000 Hz for ds006136).
    beta_band : tuple
        (low, high) Hz for beta.
    gamma_band : tuple
        (low, high) Hz for high gamma.
    n_phase_bins : int
        Number of phase bins for PBC histogram (default 18 → 20° resolution).
    delay_window : tuple
        (start_ms, end_ms) of the delay period used for PBC estimation.
    min_burst_dur_cycles : float
        Minimum burst duration in cycles (default 3, matching figure_5).
    """

    def __init__(
        self,
        fs: float = 1000.0,
        beta_band: tuple = (13, 30),
        gamma_band: tuple = (70, 150),
        n_phase_bins: int = 18,
        delay_window: tuple = (3000, 5500),
        min_burst_dur_cycles: float = 3.0,
    ):
        self.fs = fs
        self.beta_band = beta_band
        self.gamma_band = gamma_band
        self.n_phase_bins = n_phase_bins
        self.delay_window = delay_window
        self.min_burst_dur_cycles = min_burst_dur_cycles
        self.phase_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
        self.phase_centres = (self.phase_edges[:-1] + self.phase_edges[1:]) / 2

    def _bandpass(self, signal: np.ndarray, band: tuple) -> np.ndarray:
        nyq = self.fs / 2.0
        b, a = butter(4, [band[0] / nyq, band[1] / nyq], btype="band")
        return filtfilt(b, a, signal, axis=-1)

    def _instantaneous_phase(self, signal: np.ndarray) -> np.ndarray:
        return np.angle(hilbert(signal, axis=-1))

    def _instantaneous_amplitude(self, signal: np.ndarray) -> np.ndarray:
        return np.abs(hilbert(signal, axis=-1))

    def _detect_bursts(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Threshold-based burst detection (≥ 75th-percentile of first second),
        requiring a minimum duration of min_burst_dur_cycles cycles.
        Returns a binary mask of shape (n_trials, T).
        """
        ref_s, ref_e = 0, int(self.fs)
        threshold = np.nanpercentile(amplitude[:, ref_s:ref_e], 75, axis=1, keepdims=True)
        above = amplitude > threshold
        min_dur = int(self.min_burst_dur_cycles * self.fs / np.mean(self.gamma_band))

        burst_mask = np.zeros_like(above, dtype=float)
        for i in range(above.shape[0]):
            idx = np.where(np.diff(above[i].astype(int)) != 0)[0] + 1
            if above[i, 0]:
                starts = np.concatenate([[0], idx])[::2]
                ends = idx[1::2] if len(idx) > 1 else np.array([above.shape[1]])
            else:
                starts = idx[::2]
                ends = idx[1::2] if len(idx) > 1 else np.array([above.shape[1]])
            for s, e in zip(starts, ends):
                if (e - s) >= min_dur:
                    burst_mask[i, s:e] = 1.0
        return burst_mask

    def _pbc_histogram(
        self,
        beta_phase: np.ndarray,
        gamma_burst: np.ndarray,
        window: tuple,
    ) -> np.ndarray:
        s, e = window
        phase_w = beta_phase[:, s:e].ravel()
        burst_w = gamma_burst[:, s:e].ravel()

        counts = np.zeros(self.n_phase_bins)
        for k, (lo, hi) in enumerate(zip(self.phase_edges[:-1], self.phase_edges[1:])):
            mask = (phase_w >= lo) & (phase_w < hi)
            counts[k] = burst_w[mask].sum()

        total = counts.sum()
        return counts / total if total > 0 else counts

    def _pbc_signature(self, hist: np.ndarray) -> dict:
        weights = hist / hist.sum() if hist.sum() > 0 else hist
        x = np.sum(weights * np.cos(self.phase_centres))
        y = np.sum(weights * np.sin(self.phase_centres))
        resultant_length = np.sqrt(x**2 + y**2)
        preferred_phase = np.arctan2(y, x)
        peak_bin = int(np.argmax(hist))
        mean_burst_density = hist.mean()

        return {
            "histogram": hist,
            "preferred_phase": preferred_phase,
            "resultant_length": resultant_length,
            "mean_burst_density": mean_burst_density,
            "peak_phase_bin": peak_bin,
        }

    def extract(self, beta_lfp: np.ndarray, gamma_lfp: np.ndarray) -> dict:
        """
        Extract PBC signature from raw LFP arrays.

        Parameters
        ----------
        beta_lfp : (n_trials, T)
        gamma_lfp : (n_trials, T)

        Returns
        -------
        signature : dict with histogram, preferred_phase, resultant_length,
                    mean_burst_density, peak_phase_bin, feature_vector
        """
        beta_filtered  = self._bandpass(beta_lfp,  self.beta_band)
        gamma_filtered = self._bandpass(gamma_lfp, self.gamma_band)

        beta_phase   = self._instantaneous_phase(beta_filtered)
        gamma_amp    = self._instantaneous_amplitude(gamma_filtered)
        gamma_bursts = self._detect_bursts(gamma_amp)

        win = (int(self.delay_window[0]), int(self.delay_window[1]))
        # Clamp window to actual signal length
        win = (min(win[0], beta_phase.shape[1]), min(win[1], beta_phase.shape[1]))
        hist = self._pbc_histogram(beta_phase, gamma_bursts, win)
        sig  = self._pbc_signature(hist)

        sig["feature_vector"] = np.concatenate([
            hist,
            [sig["preferred_phase"] / np.pi,
             sig["resultant_length"]]
        ])
        return sig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Healthy-State Cluster Manifold
# ─────────────────────────────────────────────────────────────────────────────

class HealthyClusterManifold:
    """
    Builds a low-dimensional manifold of healthy PBC states and fits a GMM
    to define the healthy cluster boundary.
    """

    def __init__(
        self,
        n_manifold_dims: int = 2,
        n_clusters: int = 3,
        gmm_covariance_type: str = "full",
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_manifold_dims = n_manifold_dims
        self.n_clusters = n_clusters
        self.gmm_covariance_type = gmm_covariance_type
        self.contamination = contamination
        self.random_state = random_state

        self.scaler  = StandardScaler()
        self.reducer = None
        self.gmm     = GaussianMixture(
            n_components=n_clusters,
            covariance_type=gmm_covariance_type,
            random_state=random_state,
        )
        self.log_prob_threshold = None
        self.fitted = False

    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)

        if UMAP_AVAILABLE:
            self.reducer = umap.UMAP(
                n_components=self.n_manifold_dims,
                n_neighbors=15,
                min_dist=0.1,
                random_state=self.random_state,
            )
        else:
            self.reducer = PCA(
                n_components=self.n_manifold_dims,
                random_state=self.random_state,
            )

        X_embedded = self.reducer.fit_transform(X_scaled)
        self.gmm.fit(X_embedded)

        log_probs = self.gmm.score_samples(X_embedded)
        self.log_prob_threshold = np.percentile(log_probs, self.contamination * 100)
        self.fitted = True
        return self

    def embed(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.reducer.transform(X_scaled)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.gmm.score_samples(self.embed(X))

    def predict(self, X: np.ndarray):
        log_prob = self.score(X)
        inside   = log_prob >= self.log_prob_threshold
        return inside, log_prob

    def plot_manifold(self, X_healthy, X_query=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        Z_h = self.embed(X_healthy)
        ax.scatter(Z_h[:, 0], Z_h[:, 1], c="steelblue", alpha=0.4,
                   label="Healthy states", s=20)

        if X_query is not None:
            Z_q = self.embed(X_query)
            inside, _ = self.predict(X_query)
            colors = ["green" if i else "red" for i in inside]
            ax.scatter(Z_q[:, 0], Z_q[:, 1], c=colors, marker="*",
                       s=200, zorder=5, label="Query (green=safe, red=trigger)")

        ax.set_xlabel("Manifold dim 1")
        ax.set_ylabel("Manifold dim 2")
        ax.set_title("Healthy PBC Cluster Manifold")
        ax.legend()
        return ax


# ─────────────────────────────────────────────────────────────────────────────
# 3. Temporal Forecasting Models
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class _CausalConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                                  padding=self.padding, dilation=dilation)

        def forward(self, x):
            out = self.conv(x)
            return out[:, :, :-self.padding] if self.padding else out

    class _TCNBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation):
            super().__init__()
            self.net = nn.Sequential(
                _CausalConv1d(in_ch, out_ch, kernel_size, dilation),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
                _CausalConv1d(out_ch, out_ch, kernel_size, dilation),
                nn.BatchNorm1d(out_ch), nn.ReLU(),
            )
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, x):
            residual = self.downsample(x) if self.downsample else x
            return nn.functional.relu(self.net(x) + residual)

    class TCNForecaster(nn.Module):
        def __init__(self, feature_dim=20, hidden_dim=64, n_layers=4,
                     kernel_size=3, horizon=5):
            super().__init__()
            self.horizon = horizon
            self.feature_dim = feature_dim
            layers, in_ch = [], feature_dim
            for i in range(n_layers):
                layers.append(_TCNBlock(in_ch, hidden_dim, kernel_size, dilation=2**i))
                in_ch = hidden_dim
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, horizon * feature_dim)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            h = self.tcn(x)[:, :, -1]
            return self.head(h).view(-1, self.horizon, self.feature_dim)


class PBCForecaster:
    """
    Wrapper: TCN if PyTorch available, else per-dimension AR(3) fallback.
    """

    def __init__(self, seq_len=10, horizon=3, feature_dim=20, hidden_dim=64,
                 n_tcn_layers=4, lr=1e-3, n_epochs=100, batch_size=32, device="cpu"):
        self.seq_len     = seq_len
        self.horizon     = horizon
        self.feature_dim = feature_dim
        self.n_epochs    = n_epochs
        self.batch_size  = batch_size
        self.device      = device
        self.scaler      = StandardScaler()
        self._fitted     = False

        if TORCH_AVAILABLE:
            self.model = TCNForecaster(
                feature_dim=feature_dim, hidden_dim=hidden_dim,
                n_layers=n_tcn_layers, horizon=horizon,
            ).to(device)
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.loss_fn   = nn.MSELoss()
        else:
            self.model = None

    def _make_sequences(self, X):
        seqs, targets = [], []
        for i in range(len(X) - self.seq_len - self.horizon + 1):
            seqs.append(X[i : i + self.seq_len])
            targets.append(X[i + self.seq_len : i + self.seq_len + self.horizon])
        return np.array(seqs), np.array(targets)

    def fit(self, trajectories: list):
        all_X = np.vstack(trajectories)
        self.scaler.fit(all_X)

        if TORCH_AVAILABLE:
            all_seqs, all_targets = [], []
            for traj in trajectories:
                X_scaled = self.scaler.transform(traj)
                seqs, targets = self._make_sequences(X_scaled)
                if len(seqs):
                    all_seqs.append(seqs)
                    all_targets.append(targets)

            if not all_seqs:
                raise ValueError("Trajectories too short for seq_len + horizon.")

            seqs_t    = torch.tensor(np.vstack(all_seqs),    dtype=torch.float32)
            targets_t = torch.tensor(np.vstack(all_targets), dtype=torch.float32)
            loader    = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(seqs_t, targets_t),
                batch_size=self.batch_size, shuffle=True,
            )
            self.model.train()
            for epoch in range(self.n_epochs):
                epoch_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    self.optimiser.zero_grad()
                    loss = self.loss_fn(self.model(xb), yb)
                    loss.backward()
                    self.optimiser.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.n_epochs}  loss={epoch_loss/len(loader):.4f}")

        else:
            print("[INFO] Using per-dimension numpy AR(3) fallback.")
            self._ar_order = 3
            combined = self.scaler.transform(np.vstack(trajectories))
            p = self._ar_order
            self._ar_coeffs = []
            for d in range(self.feature_dim):
                y = combined[:, d]
                T_len = len(y)
                if T_len <= p:
                    self._ar_coeffs.append(np.zeros(p))
                    continue
                acf = np.array([np.dot(y[k:], y[:T_len - k]) / T_len for k in range(p + 1)])
                R   = np.array([[acf[abs(i - j)] for j in range(p)] for i in range(p)])
                r   = acf[1:p + 1]
                try:
                    coeffs = np.linalg.solve(R, r)
                except np.linalg.LinAlgError:
                    coeffs = np.zeros(p)
                self._ar_coeffs.append(coeffs)

        self._fitted = True
        return self

    def predict(self, context: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        X_scaled = self.scaler.transform(context)

        if TORCH_AVAILABLE:
            self.model.eval()
            with torch.no_grad():
                inp  = torch.tensor(X_scaled[None], dtype=torch.float32).to(self.device)
                pred = self.model(inp).cpu().numpy()[0]
            return self.scaler.inverse_transform(pred)
        else:
            p = self._ar_order
            forecast_scaled = np.zeros((self.horizon, self.feature_dim))
            for d in range(self.feature_dim):
                hist   = list(X_scaled[-p:, d])
                coeffs = self._ar_coeffs[d]
                for h in range(self.horizon):
                    nxt = float(np.dot(coeffs, hist[-p:][::-1]))
                    forecast_scaled[h, d] = nxt
                    hist.append(nxt)
            return self.scaler.inverse_transform(forecast_scaled)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DBS Stimulation Controller
# ─────────────────────────────────────────────────────────────────────────────

class AnticipatoryStimulusController:
    """
    Decision logic for anticipatory closed-loop DBS.
    """

    def __init__(self, manifold, forecaster, extractor,
                 recovery_gain=50.0, max_amplitude_ua=200.0):
        self.manifold         = manifold
        self.forecaster       = forecaster
        self.extractor        = extractor
        self.recovery_gain    = recovery_gain
        self.max_amplitude_ua = max_amplitude_ua

    def _compute_lfp_drive(self, current_hist, target_hist):
        sig_c = self.extractor._pbc_signature(current_hist)
        sig_t = self.extractor._pbc_signature(target_hist)

        density_deficit = max(0.0, sig_t["mean_burst_density"] - sig_c["mean_burst_density"])
        rl_deficit      = max(0.0, sig_t["resultant_length"]   - sig_c["resultant_length"])
        phase_error     = sig_t["preferred_phase"] - sig_c["preferred_phase"]
        phase_error     = (phase_error + np.pi) % (2 * np.pi) - np.pi

        amplitude_ua = np.clip(
            self.recovery_gain * (density_deficit + rl_deficit),
            0.0, self.max_amplitude_ua,
        )
        return {
            "amplitude_ua"    : float(amplitude_ua),
            "phase_offset_rad": float(phase_error),
            "frequency_hz"    : float(np.mean(self.extractor.beta_band)),
            "coherence"       : float(np.clip(
                rl_deficit / max(sig_t["resultant_length"], 1e-6), 0, 1
            )),
        }

    def decide(self, pbc_history, current_beta_lfp, current_gamma_lfp):
        forecast         = self.forecaster.predict(pbc_history)
        predicted_feature = forecast[-1:, :]

        inside_healthy, log_prob = self.manifold.predict(predicted_feature)
        inside_healthy = bool(inside_healthy[0])

        if inside_healthy:
            return {
                "stimulate"             : False,
                "reason"                : "Predicted PBC inside healthy cluster.",
                "horizon"               : self.forecaster.horizon,
                "log_prob"              : float(log_prob[0]),
                "stim_params"           : None,
                "predicted_pbc_feature" : predicted_feature[0],
            }

        current_sig = self.extractor.extract(current_beta_lfp, current_gamma_lfp)
        predicted_hist = predicted_feature[0, :self.extractor.n_phase_bins]
        target_hist    = np.abs(predicted_hist + (predicted_hist.mean() - predicted_hist) * 0.5)
        target_hist   /= target_hist.sum() if target_hist.sum() > 0 else 1.0

        stim_params = self._compute_lfp_drive(current_sig["histogram"], target_hist)

        return {
            "stimulate"             : True,
            "reason"                : "Predicted PBC outside healthy cluster — DBS triggered.",
            "horizon"               : self.forecaster.horizon,
            "log_prob"              : float(log_prob[0]),
            "stim_params"           : stim_params,
            "predicted_pbc_feature" : predicted_feature[0],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Closed-Loop Simulator
# ─────────────────────────────────────────────────────────────────────────────

class ClosedLoopSimulator:

    def __init__(self, controller, extractor, seq_len=10):
        self.controller = controller
        self.extractor  = extractor
        self.seq_len    = seq_len

    def run(self, beta_lfp_trials, gamma_lfp_trials, labels=None):
        n_trials       = beta_lfp_trials.shape[0]
        feature_buffer = []
        log_rows       = []

        for t in range(n_trials):
            beta_t  = beta_lfp_trials[t : t + 1]
            gamma_t = gamma_lfp_trials[t : t + 1]
            sig     = self.extractor.extract(beta_t, gamma_t)
            feature_buffer.append(sig["feature_vector"])

            row = {
                "trial"             : t,
                "preferred_phase"   : sig["preferred_phase"],
                "resultant_length"  : sig["resultant_length"],
                "mean_burst_density": sig["mean_burst_density"],
                "stimulate"         : False,
                "amplitude_ua"      : 0.0,
                "phase_offset_rad"  : 0.0,
                "log_prob"          : np.nan,
                "label"             : int(labels[t]) if labels is not None else np.nan,
            }

            if len(feature_buffer) >= self.seq_len:
                history  = np.array(feature_buffer[-self.seq_len:])
                decision = self.controller.decide(history, beta_t, gamma_t)
                row["stimulate"] = decision["stimulate"]
                row["log_prob"]  = decision["log_prob"]
                if decision["stim_params"]:
                    row["amplitude_ua"]     = decision["stim_params"]["amplitude_ua"]
                    row["phase_offset_rad"] = decision["stim_params"]["phase_offset_rad"]

            log_rows.append(row)

        return pd.DataFrame(log_rows)

    def evaluate(self, log):
        if "label" not in log.columns or log["label"].isna().all():
            return {"error": "No ground-truth labels provided."}

        healthy = log[log["label"] == 1]
        failing = log[log["label"] == 0]

        sensitivity    = failing["stimulate"].mean()  if len(failing)  else np.nan
        specificity    = (~healthy["stimulate"]).mean() if len(healthy) else np.nan
        false_trigger  = healthy["stimulate"].mean()  if len(healthy)  else np.nan
        triggered      = log[log["stimulate"]]["amplitude_ua"]
        mean_amplitude = triggered.mean() if len(triggered) else 0.0

        return {
            "sensitivity"       : float(sensitivity),
            "specificity"       : float(specificity),
            "false_trigger_rate": float(false_trigger),
            "mean_amplitude_ua" : float(mean_amplitude),
            "n_stimulations"    : int(log["stimulate"].sum()),
            "n_trials"          : len(log),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(
    fs=1000.0,
    beta_band=(13, 30),
    gamma_band=(70, 150),
    n_phase_bins=18,
    delay_window=(3000, 5500),
    seq_len=10,
    horizon=3,
    n_manifold_dims=2,
    n_clusters=3,
    n_tcn_layers=4,
    hidden_dim=64,
    n_epochs=100,
    recovery_gain=50.0,
    max_amplitude_ua=200.0,
    device="cpu",
):
    feature_dim = n_phase_bins + 2

    extractor  = PBCFeatureExtractor(
        fs=fs, beta_band=beta_band, gamma_band=gamma_band,
        n_phase_bins=n_phase_bins, delay_window=delay_window,
    )
    manifold   = HealthyClusterManifold(n_manifold_dims=n_manifold_dims, n_clusters=n_clusters)
    forecaster = PBCForecaster(
        seq_len=seq_len, horizon=horizon, feature_dim=feature_dim,
        hidden_dim=hidden_dim, n_tcn_layers=n_tcn_layers, n_epochs=n_epochs, device=device,
    )
    controller = AnticipatoryStimulusController(
        manifold=manifold, forecaster=forecaster, extractor=extractor,
        recovery_gain=recovery_gain, max_amplitude_ua=max_amplitude_ua,
    )
    return extractor, manifold, forecaster, controller


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main — uses real OpenNeuro data exactly as figure_4.ipynb
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Paths (edit if needed) ────────────────────────────────────────────────
    ROOT_DIR  = "/home/rocky/FT-bursting-WM-main"
    DATA_PATH = "/home/rocky/ds006136-1.0.0"
    SUBJECTS  = [f"sub-P{k}" for k in range(1, 8)]

    sys.path.append(os.path.join(ROOT_DIR))
    sys.path.append("/home/rocky/burst_toolbox-main/src")

    import utils
    from burst_toolbox.dsp    import compute_power
    from burst_toolbox.bursts import detect_bursts

    FS = 1000  # Hz, matches dataset

    # ── Step 1: Load data (identical to figure_4 cell 6) ─────────────────────
    print("[1] Loading data ...")
    data      = utils.read_openneuro(path=DATA_PATH, subjects=SUBJECTS)
    data["sc"] = data["subject"] + "_" + data["channel"].astype(str)
    LFP_COLS  = utils.LFP_COLS

    # ── Step 2: Compute power + detect bursts (identical to figure_4 cell 7) ─
    print("[2] Computing power and detecting bursts ...")
    processed_data = []

    for region in ["LMFG", "RMTG"]:
        for freq_band in ["beta", "high_gamma"]:
            for sc in data[data["region"] == region]["sc"].unique():
                subject, channel = sc.split("_")
                channel = int(channel)

                channel_data = data[
                    (data["subject"] == subject) & (data["channel"] == channel)
                ]
                power  = compute_power(
                    LFP=channel_data[LFP_COLS].to_numpy(),
                    freq_band=utils.FREQ_BANDS[freq_band],
                )
                bursts = detect_bursts(
                    power=power,
                    reference_period=np.array([1, 1000]),
                    min_dur_ms=3 * 1000 / np.mean(utils.FREQ_BANDS[freq_band]),
                )

                composite               = pd.DataFrame(
                    np.hstack((power, bursts)),
                    columns=utils.POWER_COLS + utils.BURST_COLS,
                )
                composite["subject"]    = subject
                composite["region"]     = region
                composite["channel"]    = channel
                composite["freq_band"]  = freq_band
                composite["trial_idx"]  = channel_data["trial_idx"].to_numpy()
                composite["modulated"]  = channel_data["modulated"].to_numpy()
                composite["n_correct"]  = channel_data["n_correct"].to_numpy()
                processed_data.append(composite)

    processed_data = pd.concat(processed_data, axis=0, ignore_index=True)
    print(f"   processed_data shape: {processed_data.shape}")

    # ── Step 3: Extract per-trial LFP signals for PBC ─────────────────────────
    # Use LMFG as beta-phase reference and RMTG as gamma-burst channel,
    # mirroring the region pairing studied in figure_4.
    print("[3] Extracting LFP matrices for PBC feature computation ...")

    def get_lfp_matrix(df_region, region_name, modulated_only=True):
        """Return (n_trials, T) LFP matrix for one region."""
        subset = df_region[df_region["region"] == region_name]
        if modulated_only:
            subset = subset[subset["modulated"] == 1]
        return subset[LFP_COLS].to_numpy()   # (n_trials, 6498)

    # Align rows: keep only subject-channel combos that appear in both regions
    def get_aligned_lfp(data, region_a, region_b, modulated_only=True):
        """
        Return (beta_mat, gamma_mat, labels) with matched trial ordering.
        Labels: 1 = n_correct == 3 (correct), 0 = n_correct <= 1 (incorrect).
        """
        mask = data["modulated"] == 1 if modulated_only else slice(None)
        sub_a = data[(data["region"] == region_a) & (data["modulated"] == 1)].copy()
        sub_b = data[(data["region"] == region_b) & (data["modulated"] == 1)].copy()

        # Intersect on (subject, trial_idx)
        key_a = set(zip(sub_a["subject"], sub_a["trial_idx"]))
        key_b = set(zip(sub_b["subject"], sub_b["trial_idx"]))
        common_keys = sorted(key_a & key_b)

        if len(common_keys) == 0:
            raise ValueError(f"No shared (subject, trial_idx) between {region_a} and {region_b}.")

        rows_a, rows_b, labels = [], [], []
        for subj, tidx in common_keys:
            row_a = sub_a[(sub_a["subject"] == subj) & (sub_a["trial_idx"] == tidx)]
            row_b = sub_b[(sub_b["subject"] == subj) & (sub_b["trial_idx"] == tidx)]
            if len(row_a) == 0 or len(row_b) == 0:
                continue
            rows_a.append(row_a[LFP_COLS].to_numpy()[0])
            rows_b.append(row_b[LFP_COLS].to_numpy()[0])
            n_corr = int(row_a["n_correct"].values[0])
            labels.append(1 if n_corr == 3 else (0 if n_corr <= 1 else -1))

        beta_mat  = np.array(rows_a)
        gamma_mat = np.array(rows_b)
        labels    = np.array(labels)

        # Drop ambiguous (n_correct == 2)
        keep       = labels != -1
        beta_mat   = beta_mat[keep]
        gamma_mat  = gamma_mat[keep]
        labels     = labels[keep]
        return beta_mat, gamma_mat, labels

    beta_all, gamma_all, labels_all = get_aligned_lfp(data, "LMFG", "RMTG")
    print(f"   Aligned trials: {len(labels_all)}  "
          f"(correct={labels_all.sum()}, incorrect={(labels_all==0).sum()})")

    # ── Step 4: Build pipeline ─────────────────────────────────────────────────
    print("[4] Building pipeline ...")
    extractor, manifold, forecaster, controller = build_pipeline(
        fs=FS,
        delay_window=(3000, min(5500, beta_all.shape[1])),
        seq_len=8,
        horizon=2,
        n_epochs=30,
        n_clusters=2,   # fewer components for small N
    )

    # ── Step 5: Extract PBC features ──────────────────────────────────────────
    print("[5] Extracting PBC features for all trials ...")
    all_features = []
    for i in range(len(beta_all)):
        sig = extractor.extract(beta_all[i:i+1], gamma_all[i:i+1])
        all_features.append(sig["feature_vector"])
    all_features = np.array(all_features)

    # ── Step 6: Fit manifold on correct (healthy) trials ──────────────────────
    correct_mask    = labels_all == 1
    incorrect_mask  = labels_all == 0
    healthy_features  = all_features[correct_mask]
    failing_features  = all_features[incorrect_mask]

    print(f"[6] Fitting healthy cluster manifold on {healthy_features.shape[0]} correct trials ...")
    manifold.fit(healthy_features)

    # ── Step 7: Fit forecaster on correct-trial PBC trajectory ─────────────────
    print("[7] Fitting forecaster on healthy PBC trajectory ...")
    # Treat correct-trial features as the "healthy" temporal trajectory
    forecaster.fit([healthy_features])

    # ── Step 8: Run closed-loop simulation on ALL trials ──────────────────────
    print("[8] Running closed-loop simulation ...")
    sim = ClosedLoopSimulator(controller, extractor, seq_len=8)
    log = sim.run(beta_all, gamma_all, labels=labels_all)

    metrics = sim.evaluate(log)
    print("\n[Results]")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    # ── Step 9: Visualisation (3-panel, matching figure_4 style) ──────────────
    print("\n[9] Generating plots ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A — PBC manifold
    ax = axes[0]
    manifold.plot_manifold(healthy_features, failing_features, ax=ax)
    ax.set_title("PBC Manifold: Correct vs Incorrect")

    # Panel B — Stimulation timeline
    ax = axes[1]
    ax.plot(log["trial"], log["resultant_length"],
            label="Resultant length", color="steelblue", lw=1.2)
    stim_trials = log[log["stimulate"]]
    ax.scatter(stim_trials["trial"], stim_trials["resultant_length"],
               color="red", zorder=5, label="DBS triggered", marker="v", s=40)
    # Mark boundary between correct and incorrect trials
    n_correct_trials = int(correct_mask.sum())
    ax.axvline(n_correct_trials, color="grey", linestyle="--",
               label="Correct→Incorrect boundary")
    ax.set_xlabel("Trial index")
    ax.set_ylabel("PBC Resultant Length")
    ax.set_title("Closed-Loop Stimulation Timeline\n(LMFG beta phase × RMTG gamma burst)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C — Stimulation amplitude distribution
    ax = axes[2]
    triggered = log[log["stimulate"] & (log["amplitude_ua"] > 0)]
    if len(triggered):
        ax.hist(triggered["amplitude_ua"], bins=15, color="salmon",
                edgecolor="black", alpha=0.8)
    ax.set_xlabel("Stimulation Amplitude (µA)")
    ax.set_ylabel("Count")
    ax.set_title("DBS Amplitude Distribution")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(ROOT_DIR, "anticipatory_dbs_result.png")
    plt.savefig(out_path, dpi=150)
    print(f"\n[Saved] {out_path}")