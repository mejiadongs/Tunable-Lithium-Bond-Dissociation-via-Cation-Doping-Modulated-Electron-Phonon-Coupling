#!/usr/bin/env bash
# One-time environment setup on the server (GPU node).
#
# NOTE: the existing `mace_eval` env has all packages but its torch is built for
# CUDA 13.0, while the server driver is CUDA 12.9 -> torch.cuda.is_available()==False.
# Fix: clone that env and swap torch for a cu12x build (driver 12.9 runs any cu12x).
#
# If you don't have mace_eval, use the "fresh env" block at the bottom instead.
set -e

SRC_ENV="${SRC_ENV:-mace_eval}"     # existing env to clone
NEW_ENV="${NEW_ENV:-mace_gpu}"      # GPU-capable env to create
CU="${CU:-cu124}"                   # cu124/cu126/cu128 all work with a 12.9 driver

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- clone the known-good MACE stack, then fix torch ---
conda create -y -n "$NEW_ENV" --clone "$SRC_ENV"
conda activate "$NEW_ENV"
pip uninstall -y torch
pip install torch --index-url "https://download.pytorch.org/whl/$CU"

# --- sanity check ---
python - <<'PY'
import torch, mace, ase, scipy
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("mace", mace.__version__, "ase", ase.__version__)
PY

echo
echo ">>> set PY in config.sh to:  $(conda info --base)/envs/$NEW_ENV/bin/python"

# ---------------------------------------------------------------------------
# FRESH-ENV alternative (uncomment if you have no mace_eval to clone):
#   conda create -y -n "$NEW_ENV" python=3.12
#   conda activate "$NEW_ENV"
#   pip install torch --index-url "https://download.pytorch.org/whl/$CU"
#   pip install mace-torch ase scipy matplotlib
# ---------------------------------------------------------------------------
