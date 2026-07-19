"""Relax atomic positions with a legacy OMat24 model while keeping the DFT cell."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import FIRE
from fairchem.core.common.relaxation.ase_utils import OCPCalculator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--structures", nargs="+", required=True)
    parser.add_argument("--output-dir", default="mlip_workflow/relaxed")
    parser.add_argument("--fmax", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    calc = OCPCalculator(
        checkpoint_path=model_path,
        cpu=False,
        seed=0,
        disable_amp=True,
    )
    rows = []
    for structure_string in args.structures:
        source = Path(structure_string).resolve()
        name = source.parent.name
        target = output_root / name
        target.mkdir(parents=True, exist_ok=True)
        atoms = read(source)
        initial = atoms.copy()
        atoms.calc = calc
        initial_fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        optimizer = FIRE(
            atoms,
            trajectory=str(target / "relax.traj"),
            logfile=str(target / "relax.log"),
        )
        converged = optimizer.run(fmax=args.fmax, steps=args.steps)
        final_forces = atoms.get_forces()
        final_fmax = float(np.linalg.norm(final_forces, axis=1).max())
        displacements = atoms.get_positions() - initial.get_positions()
        displacements -= np.rint(displacements @ np.linalg.inv(atoms.cell.array)) @ atoms.cell.array
        displacement_norms = np.linalg.norm(displacements, axis=1)
        write(target / "CONTCAR", atoms, format="vasp", direct=True, sort=False)
        row = {
            "model": model_path.name,
            "structure": name,
            "n_atoms": len(atoms),
            "converged": converged,
            "steps": optimizer.nsteps,
            "initial_fmax_eV_A": initial_fmax,
            "final_fmax_eV_A": final_fmax,
            "rms_displacement_A": float(np.sqrt(np.mean(displacement_norms**2))),
            "max_displacement_A": float(displacement_norms.max()),
        }
        rows.append(row)
        print(row, flush=True)

    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
