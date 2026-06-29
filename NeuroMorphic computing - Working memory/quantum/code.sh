# ── 1. Set your paths (do this once) ─────────────────────────────────────────
export DATA_PATH="/home/ubuntu/ds006136"
export BURST_TOOLBOX="/home/ubuntu/burst_toolbox/src"
export ROOT_DIR="/home/ubuntu/FT-bursting-WM"

# ── 2. Install dependencies ───────────────────────────────────────────────────
pip install numpy torch torchvision tqdm scikit-learn scipy mne astropy scikit-image
pip install "snntorch==0.7.0"

# ── 3. Run Experiment 1  (~5 min, CPU fine) ───────────────────────────────────
python3 experiment1_amplitude_vector.py

# ── 4. Run Experiment 2  (~10 min, CPU fine) ──────────────────────────────────
python3 experiment2_qi_pbc.py

# ── 5. Run Experiment 3  (~20 min on GPU, ~3 hrs CPU) ─────────────────────────
python3 experiment3_energy_sim.py