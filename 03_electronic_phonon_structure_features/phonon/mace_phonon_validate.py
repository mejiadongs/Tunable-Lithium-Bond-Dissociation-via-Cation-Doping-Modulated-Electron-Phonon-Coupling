"""Finite-displacement MACE phonons compared at the q points of a DFT band.yaml."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def calculate_frequencies(atoms, calculator, supercell, qpoints, distance):
    from ase import Atoms
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unitcell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell,
                           scaled_positions=atoms.get_scaled_positions())
    phonon = Phonopy(unitcell, np.diag(supercell), primitive_matrix="P")
    phonon.generate_displacements(distance=distance, is_plusminus=True)
    forces = []
    for cell in phonon.supercells_with_displacements:
        displaced = Atoms(cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True)
        displaced.calc = calculator
        force = displaced.get_forces()
        forces.append(force - force.mean(axis=0))
    phonon.forces = forces
    phonon.produce_force_constants(fc_calculator="traditional")
    phonon.symmetrize_force_constants()
    phonon.run_qpoints(qpoints)
    return np.asarray(phonon.qpoints.frequencies)


def compare_frequencies(reference, predictions):
    reference, predictions = np.asarray(reference), np.asarray(predictions)
    if (predictions.ndim != 3 or predictions.shape[1:] != reference.shape
            or not np.isfinite(predictions).all() or not np.isfinite(reference).all()):
        raise ValueError("DFT and MACE frequency arrays must be finite with matching q/mode counts.")
    reference, predictions = np.sort(reference, axis=-1), np.sort(predictions, axis=-1)
    mean, std = predictions.mean(axis=0), predictions.std(axis=0)
    metrics = {"committee_RMSE_THz": float(np.sqrt(np.mean((mean - reference) ** 2))),
               "per_model_RMSE_THz": np.sqrt(np.mean((predictions - reference) ** 2, axis=(1, 2))).tolist()}
    return reference, mean, std, metrics


def main():
    from ase.io import read

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("structure", "dft-band", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--supercell", nargs=3, type=int, default=[2, 2, 2])
    parser.add_argument("--distance", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.supercell) < 1 or args.distance <= 0:
        parser.error("Use positive supercell sizes and displacement.")
    atoms = read(args.structure)
    if not atoms.pbc.all() or atoms.constraints:
        parser.error("Supply an unconstrained periodic primitive cell matching the DFT reference.")
    reference = yaml.safe_load(Path(args.dft_band).read_text(encoding="utf-8"))
    if "lattice" not in reference or not np.allclose(reference["lattice"], atoms.cell, atol=1e-5):
        parser.error("The structure must use the same lattice/basis as the DFT band.yaml.")
    if "points" in reference and [p["symbol"] for p in reference["points"]] != atoms.get_chemical_symbols():
        parser.error("The DFT primitive-cell species/order does not match the input structure.")
    qpoints = np.array([p["q-position"] for p in reference["phonon"]])
    dft = np.array([[b["frequency"] for b in p["band"]] for p in reference["phonon"]])
    if dft.shape != (len(qpoints), 3 * len(atoms)):
        parser.error("DFT band count does not match the supplied primitive cell.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    from mace.calculators import MACECalculator

    predictions = []
    for model in args.models:
        calculator = MACECalculator(model_paths=model, device=args.device, default_dtype="float64")
        predictions.append(calculate_frequencies(atoms, calculator, args.supercell, qpoints, args.distance))
    dft, mean, std, metrics = compare_frequencies(dft, predictions)
    indices = np.indices(dft.shape).reshape(2, -1).T
    np.savetxt(out / "frequencies.csv", np.column_stack([indices, dft.ravel(), mean.ravel(), std.ravel()]),
               delimiter=",", header="q_index,mode_index,DFT_THz,MACE_mean_THz,MACE_std_THz", comments="")
    np.savetxt(out / "qpoints.csv", qpoints, delimiter=",", header="q1,q2,q3", comments="")
    (out / "summary.json").write_text(json.dumps({**vars(args), **metrics}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
