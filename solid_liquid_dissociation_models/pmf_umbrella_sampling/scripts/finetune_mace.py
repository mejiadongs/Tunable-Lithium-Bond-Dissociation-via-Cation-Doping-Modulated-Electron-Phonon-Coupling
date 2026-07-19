"""
Fine-tune MACE-MP-0 on PMF umbrella-sampling data.

Usage:
    python finetune_mace.py --train-file train.xyz --valid-file valid.xyz
        --output-dir mace_pmf_run
"""

import subprocess
import sys
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train-file", required=True)
parser.add_argument("--valid-file", required=True)
parser.add_argument("--output-dir", default="mace_pmf_run")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

TRAIN_XYZ = Path(args.train_file)
VALID_XYZ = Path(args.valid_file)
MODEL_DIR = Path(args.output_dir)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE  = MODEL_DIR / "train.log"

# -- Check inputs -------------------------------------------------------------
for p in [TRAIN_XYZ, VALID_XYZ]:
    if not p.exists():
        sys.exit(f"[ERROR] {p} not found. Run extract_training_data_pmf.py first.")

# -- Hyperparameters ----------------------------------------------------------
cmd = [
    sys.executable, "-m", "mace.cli.run_train",
    "--name=mace_pmf",
    "--foundation_model=mace_mp",        # MACE-MP-0, auto-downloaded
    f"--train_file={TRAIN_XYZ}",
    f"--valid_file={VALID_XYZ}",
    "--r_max=5.0",
    "--num_radial_basis=8",
    "--hidden_irreps=256x0e+256x1o+256x2e",
    "--max_num_epochs=200",
    "--batch_size=4",
    "--lr=5e-4",
    "--energy_weight=1.0",
    "--forces_weight=100.0",
    "--E0s=average",
    "--energy_key=REF_energy",
    "--forces_key=REF_forces",
    "--valid_fraction=0.0",
    f"--device={args.device}",
    "--default_dtype=float64",
    f"--log_dir={MODEL_DIR}",
    f"--results_dir={MODEL_DIR}",
    f"--checkpoints_dir={MODEL_DIR / 'checkpoints'}",
    "--save_cpu",
]

print("MACE fine-tuning for PMF umbrella-sampling data")
print(f"Train : {TRAIN_XYZ}")
print(f"Valid : {VALID_XYZ}")
print(f"Output: {MODEL_DIR}")
print(f"Log   : {LOG_FILE}")
print()

with open(LOG_FILE, "w") as log:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    proc.wait()

if proc.returncode != 0:
    sys.exit(f"\n[ERROR] Training failed (exit code {proc.returncode}). See {LOG_FILE}")

print(f"\nTraining complete.")
print(f"Final model : {MODEL_DIR / 'mace_pmf.model'}")
print()
print("Use the resulting model path with run_umbrella_mace.py.")
