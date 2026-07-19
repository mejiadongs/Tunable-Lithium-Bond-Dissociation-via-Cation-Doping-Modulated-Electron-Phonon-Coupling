# ML and DFT Joint Optimization

This module stores machine-learning-assisted notebooks used alongside DFT workflows.

## Main Files

- `doping_model.ipynb`
  Notebook for doping-model exploration or optimization studies.
- `m3gnet.ipynb`
  Notebook related to M3GNet-based structure or property modeling.

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
