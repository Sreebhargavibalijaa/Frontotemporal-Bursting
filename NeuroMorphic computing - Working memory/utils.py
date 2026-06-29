import mne
import os
import numpy as np
import pandas as pd

# ---------
# Constants
# ---------
T = 6498 # (ms)
LFP_COLS = [f"LFP_{k}" for k in range(T)]
POWER_COLS = [f"power_{k}" for k in range(T)]
BURST_COLS = [f"bursts_{k}" for k in range(T)]
CUTOFF = 6300
BASELINE = np.array([400, 600])
FREQ_BANDS = {
    "beta": np.array([12, 30]), # (Hz)
    "high_gamma": np.array([70, 140]) # (Hz)
}
PLOTTING = {
    "beta": {
        "title": "Beta (12-30 Hz)",
        "color": "darkblue",
        "fillcolor": "lightsteelblue"
    },
    "high_gamma": {
        "title": "High gamma (70-140 Hz)",
        "color": "darkred",
        "fillcolor": "darksalmon"
    },
    "correct": {
        "title": "Correct",
        "color": "green",
        "fillcolor": "lightgreen"
    },
    "incorrect": {
        "title": "Incorrect",
        "color": "red",
        "fillcolor": "lightpink"
    }
}

# ----
# Data
# ----

def read_openneuro(path, subjects):

    data = []

    for subject in subjects:
        # Read data
        subject_data = mne.io.read_raw_edf(os.path.join(path, subject, "ieeg", f"{subject}_task-OWM_ieeg.edf")).get_data(units = "uV")

        # Read channel info
        channel_info = pd.read_csv(os.path.join(path, subject, "ieeg", f"{subject}_task-OWM_channels.tsv"), sep = "\t")

        # Read performance info
        performance = pd.read_csv(os.path.join(path, "sourcedata", f"{subject}_task-OWM_performance.tsv"), sep = "\t")

        # Read trial id info
        trial_ids = pd.read_csv(os.path.join(path, "sourcedata", f"{subject}_task-OWM_trialids.tsv"), sep = "\t")

        for channel_idx in range(len(channel_info)):

            # Split each channel into trials
            n_trials = channel_info.iloc[channel_idx]["n_trials"]
            channel_data = subject_data[channel_idx, :n_trials * T].reshape((n_trials, T))
            channel_data = pd.DataFrame(channel_data, columns = LFP_COLS)
            
            channel_data["subject"] = subject
            channel_data["region"] = channel_info.iloc[channel_idx]["name"].split("-")[0]
            channel = int(channel_info.iloc[channel_idx]["name"].split("-")[1])
            channel_data["channel"] = channel
            channel_data["modulated"] = channel_info.iloc[channel_idx]["modulated"]
            channel_data["trial_idx"] = trial_ids[trial_ids["channel"] == channel]["trial_idx"].to_numpy()

            # Add performance info
            performance_channel = performance[performance["channel"] == channel].reset_index(drop = True)
            n_correct = performance_channel[["perf1", "perf2", "perf3"]].sum(axis = 1).to_numpy()
            channel_data["n_correct"] = n_correct

            data.append(channel_data)

    data = pd.concat(data, axis = 0, ignore_index = True)

    return data