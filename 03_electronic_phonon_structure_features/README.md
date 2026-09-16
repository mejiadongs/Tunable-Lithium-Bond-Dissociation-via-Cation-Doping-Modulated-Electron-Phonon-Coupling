# Electronic and Phonon Structure Features

This module collects descriptor-generation and analysis workflows related to electronic structure, phonon behavior, and electron-phonon coupling.

## Second-revision vibrational method

For HfO2, finite-displacement phonons were calculated with phonopy using the
fine-tuned MACE potentials and compared with direct DFT results at matched wavevectors.
Element-projected spectra and molecular-dynamics VDOS were used to examine
vibrational contributions and composition-dependent trends. These vibrational
descriptors do not directly determine an electron-phonon coupling constant.

Minimal MACE–DFT comparison (run from this module with `mace-torch`, `phonopy`,
`ase`, `numpy`, and `PyYAML` installed):

```bash
python phonon/mace_phonon_validate.py --structure POSCAR --dft-band band.yaml --models seed11.model seed23.model seed37.model --out phonon_check
```

Use an already relaxed primitive cell with the same lattice/basis as the DFT
reference. Frequencies are sorted at each q point for comparison; no branch
tracking or non-analytical correction is applied.

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
