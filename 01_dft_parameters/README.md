# DFT Parameters

This module contains DFT-oriented structure-generation utilities and setup scripts used before production calculations.

## Submodules

- `aimd/`
  Scripts for preparing surface models intended for ab initio molecular dynamics workflows.
- `structural_optimization/`
  Surface-construction utilities used for geometry optimization and static structure preparation.

## Main Files

- `aimd/surfaces_aimd.py`
  Surface-generation helper for AIMD-ready models.
- `structural_optimization/surfaces.py`
  Surface-construction helper for structural optimization workflows.

## Suggested Expansion

- Add calculation templates such as `INCAR`, `KPOINTS`, or job scripts if this module becomes the home for reusable DFT inputs.
