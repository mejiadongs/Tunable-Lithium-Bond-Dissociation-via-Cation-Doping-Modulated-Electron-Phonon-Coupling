# Engineering Electron-Phonon Coupling via Cation Doping

This repository collects scripts, notebooks, and workflow fragments used to study lithium bond dissociation through cation doping, with a focus on coupled electronic, phonon, electrochemical, and spectroscopy analysis.

## Repository Structure

- `dft_parameters/`
  DFT model preparation and surface-building utilities for structural optimization and AIMD workflows.
- `electrochemical_parameters_calculations/`
  Electrochemical property calculations, including ionic conductivity fitting and transference number estimation.
- `electronic_phonon_structure_features/`
  Electronic descriptors, phonon calculations, and phonon-scattering analysis workflows.
- `ml_dft_joint_optimization/`
  Machine-learning-assisted DFT optimization notebooks, including doping-model exploration.
- `rixs_thz_tds_data_processing/`
  Data-processing workflows for mRIXS and THz-TDS experiments.
- `solid_liquid_dissociation_models/`
  Modeling workflows for solid-phase and liquid-phase dissociation systems.

Each module contains its own `README.md` with a more detailed description of its internal layout and main files.

## Naming Conventions

- Filenames and directories use lowercase `snake_case` where practical.
- Notebooks kept as alternative or archived variants are labeled with `_backup`.
- Top-level modules group related workflows instead of enforcing a single runtime environment.

## Notes

- Most workflows are stored as Jupyter notebooks and should be reviewed together with their local input assumptions.
- Environment setup, calculation inputs, and dataset provenance can be expanded later inside each module README as the repository matures.
