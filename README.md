# Engineering Electron-Phonon Coupling via Cation Doping

This repository collects scripts, notebooks, and workflow fragments used to study lithium bond dissociation through cation doping, with a focus on coupled electronic, phonon, electrochemical, and spectroscopy analysis.

## Installation

Create a Python environment and install the core dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you plan to run the machine-learning-assisted notebooks in `04_ml_dft_joint_optimization/`, install the optional ML stack as well:

```bash
pip install -r requirements-ml.txt
```

## External Tools

Several workflows depend on domain-specific software beyond Python packages:

- `VASP` for DFT and AIMD calculations
- `phonopy` for phonon workflows and post-processing
- `JupyterLab` for notebook-based analysis

Some scripts and notebooks also assume local databases, VASP outputs, or experiment-specific input files that are not bundled in this repository.

## Quick Start

1. Read the root module map below and open the `README.md` inside the module you want to use.
2. Install `requirements.txt` and, if needed, `requirements-ml.txt`.
3. Start from one of the scripts or notebooks most closely matched to your task.

Example entry points:

```bash
python 05_rixs_thz_tds_data_processing/mrixs/mrixs_pipeline.py --help
python 03_electronic_phonon_structure_features/phonon/phonon/epc_viz.py --help
python 06_solid_liquid_dissociation_models/liquid/freeze_layers_from_db.py --help
```

Notebook-centered workflows can be opened directly in JupyterLab after the environment is installed.

## Repository Structure

- `01_dft_parameters/`
  DFT model preparation and surface-building utilities for structural optimization and AIMD workflows.
- `02_electrochemical_parameters_calculations/`
  Electrochemical property calculations, including ionic conductivity fitting and transference number estimation.
- `03_electronic_phonon_structure_features/`
  Electronic descriptors, phonon calculations, and phonon-scattering analysis workflows.
- `04_ml_dft_joint_optimization/`
  Machine-learning-assisted DFT optimization notebooks, including doping-model exploration.
- `05_rixs_thz_tds_data_processing/`
  Data-processing workflows for mRIXS and THz-TDS experiments.
- `06_solid_liquid_dissociation_models/`
  Modeling workflows for solid-phase and liquid-phase dissociation systems.

Each module contains its own `README.md` with a more detailed description of its internal layout and main files.

## Naming Conventions

- Filenames and directories use lowercase `snake_case` where practical.
- Notebooks kept as alternative or archived variants are labeled with `_backup`.
- Top-level modules group related workflows instead of enforcing a single runtime environment.

## Citation

Citation metadata is provided in `CITATION.cff`. If you use this repository in research or derived workflows, please cite the repository and any corresponding publication you prepare from it.

## Repository Status

- The repository is now organized by workflow module and includes per-module README files.
- Core and optional ML dependency lists are provided in `requirements.txt` and `requirements-ml.txt`.
- Most workflows are currently notebook-driven and may require project-specific input files or calculation outputs.
- Tests, packaged command-line entry points, and CI automation are still limited and can be expanded later.

## License

This repository is released under the [MIT License](LICENSE).

## Notes

- Most workflows are stored as Jupyter notebooks and should be reviewed together with their local input assumptions.
- Environment setup, calculation inputs, and dataset provenance can be expanded later inside each module README as the repository matures.
