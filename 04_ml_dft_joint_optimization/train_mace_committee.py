"""Minimal bulk/interface MACE fine-tuning launcher for pre-split DFT datasets."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def training_command(args, seed):
    out = Path(args.out).resolve() / f"seed{seed}"
    weights = ({"periodic_interface": 5, "Li_transfer_seed": 5,
                "oxygen_vacancy_interface": 4, "legacy_surface": 1}
               if args.kind == "interface" else {})
    options = {
        "name": f"{args.kind}_seed{seed}", "seed": seed,
        "foundation_model": str(Path(args.foundation).resolve()),
        "multiheads_finetuning": "False", "foundation_filter_elements": "True",
        "train_file": str(Path(args.train).resolve()),
        "valid_file": str(Path(args.valid).resolve()),
        "test_file": str(Path(args.test).resolve()), "valid_fraction": 0.0,
        "energy_key": "REF_energy", "forces_key": "REF_forces",
        "E0s": "estimated", "loss": "weighted", "energy_weight": 1,
        "forces_weight": 100, "lr": 1e-4, "weight_decay": 1e-8,
        "batch_size": 2 if args.kind == "interface" else 8,
        "max_num_epochs": 100, "patience": 15, "ema_decay": 0.995,
        "default_dtype": args.dtype, "device": args.device,
        "config_type_weights": json.dumps(weights),
        "model_dir": str(out / "models"), "log_dir": str(out / "logs"),
        "results_dir": str(out / "results"), "checkpoints_dir": str(out / "checkpoints"),
    }
    return [sys.executable, "-m", "mace.cli.run_train",
            *(f"--{key}={value}" for key, value in options.items()),
            "--ema", "--amsgrad", "--save_cpu"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("foundation", "train", "valid", "test", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--kind", choices=["bulk", "interface"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("foundation", "train", "valid", "test"):
        if not Path(getattr(args, name)).is_file():
            parser.error(f"Missing input: {getattr(args, name)}")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct seeds.")
    splits = [Path(getattr(args, name)).resolve() for name in ("train", "valid", "test")]
    if len(set(splits)) != 3:
        parser.error("Train, validation and test inputs must be separate files.")
    for seed in args.seeds:
        out = Path(args.out) / f"seed{seed}"
        if out.exists() and any(out.iterdir()):
            parser.error(f"Use a fresh output directory: {out}")
    for seed in args.seeds:
        command = training_command(args, seed)
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
