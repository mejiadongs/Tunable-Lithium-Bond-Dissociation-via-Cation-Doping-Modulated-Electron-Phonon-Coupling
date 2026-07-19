# OMat24 validation and phonon workflow

These scripts use legacy Fair-Chem-compatible OMat24 checkpoints for fixed-cell
relaxation and finite-displacement phonon calculations. Create an isolated Python
environment with compatible Fair-Chem, PyTorch/CUDA, and phonopy versions before
running them.

Screen candidate models using residual forces on DFT-relaxed structures. Low residual
forces are necessary but not sufficient for accurate phonons; validate finite-
displacement forces and phonon observables against DFT before using a model for new
structures.

`run_phonons.py` uses central finite displacements, removes the net force drift,
symmetrizes the force constants, and writes total/projected DOS plus a JSON summary.
Use the same displacement and low-frequency cutoff for every compared structure.
