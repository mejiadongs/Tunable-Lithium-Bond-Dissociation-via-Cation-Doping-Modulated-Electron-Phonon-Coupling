"""Finite-displacement phonons using a local legacy OMat24 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


def to_phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def to_ase_atoms(atoms: PhonopyAtoms) -> Atoms:
    return Atoms(
        symbols=atoms.symbols,
        cell=atoms.cell,
        scaled_positions=atoms.scaled_positions,
        pbc=True,
    )


def integrate_window(frequencies: np.ndarray, dos: np.ndarray, cutoff: float) -> float:
    mask = (frequencies >= 0.0) & (frequencies <= cutoff)
    return float(np.trapz(dos[mask], frequencies[mask])) if mask.sum() > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--supercell", nargs=3, type=int, default=(1, 1, 1))
    parser.add_argument("--mesh", nargs=3, type=int, default=(12, 12, 12))
    parser.add_argument("--distance", type=float, default=0.01)
    parser.add_argument("--cutoff-thz", type=float, default=5.0)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atoms = read(args.structure)
    calculator = OCPCalculator(
        checkpoint_path=Path(args.model).resolve(),
        cpu=False,
        seed=0,
        disable_amp=True,
    )
    phonon = Phonopy(
        to_phonopy_atoms(atoms),
        supercell_matrix=np.diag(args.supercell),
        primitive_matrix="P",
        symprec=1e-5,
    )
    phonon.generate_displacements(distance=args.distance, is_plusminus=True)
    displaced_cells = phonon.supercells_with_displacements
    print(f"Finite-displacement calculations: {len(displaced_cells)}", flush=True)
    force_cache = output / "displacement_forces"
    force_cache.mkdir(exist_ok=True)
    force_sets = []
    for index, displaced_cell in enumerate(displaced_cells, start=1):
        cache_file = force_cache / f"forces_{index:04d}.npy"
        if cache_file.exists():
            forces = np.load(cache_file)
        else:
            displaced = to_ase_atoms(displaced_cell)
            displaced.calc = calculator
            forces = np.asarray(displaced.get_forces())
            forces -= forces.mean(axis=0, keepdims=True)
            np.save(cache_file, forces)
        force_sets.append(forces)
        print(f"  {index}/{len(displaced_cells)}", flush=True)

    phonon.forces = force_sets
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()
    phonon.save(filename=output / "phonopy_params.yaml", settings={"force_sets": True})
    np.save(output / "force_constants.npy", phonon.force_constants)

    phonon.run_mesh(args.mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    mesh = phonon.get_mesh_dict()
    all_frequencies = np.asarray(mesh["frequencies"])
    gamma_frequencies = phonon.get_frequencies([0, 0, 0])

    phonon.run_total_dos()
    phonon.write_total_dos(filename=output / "total_dos.dat")
    total = phonon.get_total_dos_dict()
    frequency_points = np.asarray(total["frequency_points"])
    total_dos = np.asarray(total["total_dos"])

    phonon.run_projected_dos()
    phonon.write_projected_dos(filename=output / "projected_dos.dat")
    projected = phonon.get_projected_dos_dict()
    projected_dos = np.asarray(projected["projected_dos"])

    symbols = atoms.get_chemical_symbols()
    atom_low = [integrate_window(frequency_points, row, args.cutoff_thz) for row in projected_dos]
    positive_total = integrate_window(frequency_points, total_dos, float(frequency_points.max()))
    low_total = integrate_window(frequency_points, total_dos, args.cutoff_thz)
    summary = {
        "model": Path(args.model).name,
        "structure": str(Path(args.structure).resolve()),
        "n_unitcell_atoms": len(atoms),
        "supercell": args.supercell,
        "n_displacements": len(displaced_cells),
        "displacement_A": args.distance,
        "mesh": args.mesh,
        "minimum_mesh_frequency_THz": float(all_frequencies.min()),
        "minimum_gamma_frequency_THz": float(gamma_frequencies.min()),
        "n_gamma_imaginary_below_minus_0p1_THz": int(np.sum(gamma_frequencies < -0.1)),
        "cutoff_THz": args.cutoff_thz,
        "total_low_frequency_weight": low_total,
        "total_low_frequency_fraction": low_total / positive_total if positive_total else None,
        "atom_low_frequency_weights": [
            {"index": i, "symbol": symbol, "weight": value}
            for i, (symbol, value) in enumerate(zip(symbols, atom_low))
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "atom_low_frequency_weights"}, indent=2))


if __name__ == "__main__":
    main()
