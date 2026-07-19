# Electronic and Phonon Structure Features

This module collects descriptor-generation and analysis workflows related to electronic structure, phonon behavior, and electron-phonon coupling.

## Submodules

- `electronic/`
  Electronic-structure descriptors, surface-modeling notebooks, and visualization helpers.
- `phonon/`
  Phonon calculation utilities and phonon-scattering analysis workflows.

## Suggested Workflow

1. Prepare or inspect surface models in `electronic/`.
2. Extract electronic descriptors such as band-center features.
3. Run phonon or scattering analysis in `phonon/`.
4. Combine descriptors across modules for downstream interpretation or ML-assisted screening.
