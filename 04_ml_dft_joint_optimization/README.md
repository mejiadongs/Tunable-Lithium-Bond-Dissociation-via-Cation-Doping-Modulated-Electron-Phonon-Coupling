# ML and DFT Joint Optimization

This module stores machine-learning-assisted notebooks used alongside DFT workflows.

## Main Files

- `doping_model.ipynb`
  Notebook for doping-model exploration or optimization studies.
- `m3gnet.ipynb`
  Notebook related to M3GNet-based structure or property modeling.

## Second-revision method

For the HfO2 calculations, separate bulk and interface potentials were fine-tuned
from MACE-OMAT-0 medium using DFT energies and forces. Three independently trained
models formed a committee, with validation against held-out DFT configurations
and representative interfacial Li-transfer configurations.

`train_mace_committee.py` provides a minimal three-seed launcher. Supply a local
MACE-OMAT-0 medium checkpoint and separate bulk or interface datasets containing
`REF_energy` and `REF_forces` (with `config_type` labels for interface weighting):

```bash
python train_mace_committee.py --kind bulk --foundation foundation.model --train train.extxyz --valid valid.extxyz --test test.extxyz --out bulk_models
```

Run separately with `--kind interface` for the interface potential. Use Python 3.10+
with `mace-torch` (training options follow v0.3.16); `--dry-run` prints commands.
The local checkpoint must match `--dtype` (default `float64`); use `--dtype float32`
with a checkpoint already converted to float32.
Inputs must already have independent training/validation/test splits. These are
compact method examples, not the complete production data-preparation workflow.

## Reproducible MLIP Workflows

- `mace_molecular_dynamics/`
  MACE fine-tuning, molecular-dynamics, and vibrational-density-of-states workflow.
  It contains scripts only; trajectories, model checkpoints, structures, and generated
  figures are deliberately excluded.
- `omat24_phonon_validation/`
  OMat24/Fair-Chem fixed-cell relaxation and finite-displacement phonon-validation
  scripts. Force constants and density-of-states outputs are deliberately excluded.

## Typical Use

- Rapid candidate screening before expensive DFT runs
- Feature exploration for doped systems
- Model-assisted comparison across structures or compositions
