# MACE-assisted multi-window PMF workflow

This module contains scripts for reconstructing a potential of mean force (PMF) by
multi-window umbrella sampling with a MACE interatomic potential. Input structures,
DFT-AIMD trajectories, trained model weights, collective-variable time series, and
PMF outputs are intentionally excluded.

## Workflow

1. Run `scripts/01_extract_training_data.py` on a local directory containing the
   umbrella-sampling `OUTCAR` files. It discards the first 200 MD steps, samples every
   20th frame, and creates a deterministic 90/10 training/validation split.
2. Run `scripts/02_finetune_mace.py` to fine-tune MACE-MP-0 on the generated datasets.
3. Run `scripts/03_run_umbrella_mace.py` for each system, providing a trained model and
   local `CONTCAR` and `ICONST` paths. The script uses a distance collective variable
   and writes one CV time series per window.
4. Run `scripts/04_calc_pmf_wham.py` to discard the initial 20% of each window and solve
   the WHAM equations.
5. Use `scripts/05_compare_pmf.py` to compare independently generated PMF profiles.

## Recorded production settings

| Setting | Value |
| --- | --- |
| Temperature | 300 K |
| MD time step | 0.5 fs |
| Harmonic restraint | 5.0 eV/Angstrom^2 |
| Window spacing | 0.1 Angstrom |
| Window warm-up | 2,000 steps |
| Production per window | 50,000 steps |
| CV save interval | 10 steps |
| WHAM bins | 100 |
| WHAM equilibration removal | first 20% |

The collective-variable atom indices and distance ranges are system-specific and are
declared in `03_run_umbrella_mace.py`; verify them against the corresponding `ICONST`
before running a new system.

## MACE references

The workflow fine-tunes the MACE-MP-0 foundation model. Cite the foundation-model
paper, [A foundation model for atomistic materials chemistry](https://arxiv.org/abs/2401.00096),
and the MACE architecture paper, [MACE: Higher Order Equivariant Message Passing
Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2206.07697).
