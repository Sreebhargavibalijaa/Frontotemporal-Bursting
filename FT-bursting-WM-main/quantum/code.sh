# ── 1. Set your paths (do this once) ─────────────────────────────────────────
export DATA_PATH="/home/rocky/ds006136-1.0.0"
export BURST_TOOLBOX="/home/rocky/burst_toolbox-main/src"
export ROOT_DIR="/home/rocky/FT-bursting-WM-main"

# ── 2. Install dependencies ───────────────────────────────────────────────────
pip install snntorch torch torchvision tqdm scikit-learn scipy astropy scikit-image

# ── 3. Run Experiment 1  (~5 min, CPU fine) ───────────────────────────────────
python3 experiment1_amplitude_vector.py

# ── 4. Run Experiment 2  (~10 min, CPU fine) ──────────────────────────────────
python3 experiment2_qi_pbc.py

# ── 5. Run Experiment 3  (~20 min on GPU, ~3 hrs CPU) ─────────────────────────
python3 experiment3_energy_sim.py