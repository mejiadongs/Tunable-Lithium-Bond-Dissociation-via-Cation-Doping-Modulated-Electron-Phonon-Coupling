# MACE fine-tuning, molecular dynamics, and VDOS workflow

This directory contains a reusable workflow for fine-tuning a MACE foundation model
on DFT-AIMD configurations, validating molecular-dynamics vibrational density of
states (VDOS), and running composition-series or surface-projected VDOS calculations.
Only methods and scripts are included: source trajectories, structures, checkpoints,
and generated results are not distributed.

## What runs where
- **Fine-tuning + MD**: server GPU node (no scheduler, run directly, node has internet).
- Copy `config.example.sh` to `config.sh`, then edit it for the local system.

## 0. Put the bundle + point it at your data
```bash
# on the server
scp -r mace_molecular_dynamics  you@server:~/project/
```
Edit **`config.sh`** to point at local input data:
- `AIMD_DIR`: directory containing the AIMD input trajectories
- `RELAX_DIR`: directory containing the relaxed structures
- `BUNDLE`: directory containing this workflow
- `PY`: Python executable in the MACE environment

## 1. Environment (once)
```bash
bash 00_setup_env.sh
# then set PY in config.sh to the printed .../envs/mace_ft/bin/python
```
Match the CUDA wheel (`cu121`/`cu124`) in `00_setup_env.sh` to the node's driver
(`nvidia-smi`). Nothing else needs the network after this.

## 2. Fine-tune  (`10_finetune.sh`)
Extracts training configurations from AIMD and fine-tunes MACE-OMAT-0
(energy+forces, float64, GPU). Inspect held-out force errors before using the model
for production dynamics.
The model path is auto-recorded in `work/model_path.txt`.

## 3. Validate (`20_validate.sh`)
Runs MACE-MD on validation cells and overlays the VDOS. Inspect
`work/fig_validation_25pct.png`:
- MACE-MD (red) should track DFT-AIMD (black), especially the **< 10 THz** shape;
- compare the spectral centroid before using the potential for production cells.

If it does not match, adjust the training set or loss weights before running the
composition series.

## 4. Concentration series  (`30_md_series.sh`)
MACE-MD on a user-specified dopant-concentration series. Uncomment or adapt the
corresponding block for additional dopants.

## 5. Plotting  (`40_plot.sh`)
Produces `work/md/series/fig_concentration_vdos.png`:
- (a) local (Sc + nearest-O) VDOS overlaid per concentration
- (b) **0-10 THz VDOS centroid versus concentration**
- (c) local dominant peak position vs concentration

## 6. (Optional) Slab workflow  (`50_slab.sh`)
Runs MD on a user-provided slab and computes surface-projected VDOS.
- Input slab structures are intentionally not included in this repository.
- Uses **NVT** production (surface stability). Run after `20_validate.sh`.
- Validate the slab workflow by comparing its interior and bulk spectral behavior.
- Use a model trained on surface-relevant environments; do not extrapolate a bulk-only
  model to under-coordinated surface atoms without validation.

## Descriptor & honest scope
- Report the chosen low-frequency VDOS centroid and local peak consistently across
  compositions.
- MD-VDOS is a vibrational descriptor, not a direct calculation of electron-phonon
  coupling strength.

## Files
| file | role |
|---|---|
| `config.example.sh` | template for paths and MD settings; copy to `config.sh` |
| `00_setup_env.sh` | conda env + torch(CUDA) + mace-torch |
| `extract_dataset.py` | AIMD trajectory to train/validation/test datasets |
| `10_finetune.sh` | fine-tune MACE-OMAT-0 |
| `vdos_dft_aimd.py` | DFT-AIMD reference VDOS (25 %) |
| `run_md.py` | 300 K NVT-equil + NVE-production, saves velocities |
| `vdos_from_md.py` | projected VDOS and low-frequency centroid/peak |
| `20_validate.sh` + `compare_validation.py` | MACE-MD vs DFT-AIMD overlay |
| `30_md_series.sh` | MD over the concentration series |
| `40_plot.sh` + `plot_concentration_vdos.py` | reviewer figure |
