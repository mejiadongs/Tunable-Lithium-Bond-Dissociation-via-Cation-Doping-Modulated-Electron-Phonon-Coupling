"""
Run 300 K MD with the fine-tuned MACE potential and record velocities for VDOS.
Protocol (identical for every structure so concentrations are comparable):
  1. relax atomic positions at the fixed DFT cell (FIRE) to the MACE minimum
  2. NVT equilibration (Nose-Hoover chain), same T as the DFT-AIMD (300 K)
  3. NVE production; velocities recorded every step -> velocities.npy

Usage:
  run_md.py <structure> <model> <outdir> [--equil 5000] [--prod 15000]
            [--dt 1.0] [--T 300] [--fix-cell-relax]
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase import units
from mace.calculators import MACECalculator

p = argparse.ArgumentParser()
p.add_argument("structure")
p.add_argument("model")
p.add_argument("outdir")
p.add_argument("--equil", type=int, default=5000)
p.add_argument("--prod", type=int, default=15000)
p.add_argument("--dt", type=float, default=1.0)          # fs
p.add_argument("--T", type=float, default=300.0)
p.add_argument("--relax_steps", type=int, default=200)
p.add_argument("--prod_ensemble", choices=["nve", "nvt"], default="nve",
               help="nve for bulk (clean VACF); nvt for slabs (surface stability)")
p.add_argument("--friction", type=float, default=0.02,
               help="Langevin friction for equilibration (1/ASE-time). Raise for "
                    "unstable slabs (e.g. 0.1).")
p.add_argument("--prod_thermostat", action="store_true",
               help="keep a (weak) Langevin thermostat during production instead of "
                    "NVE — use if NVE production of a slab still heats up.")
p.add_argument("--tmax_factor", type=float, default=3.0,
               help="per-step rescue: if T exceeds tmax_factor*T, reset velocities to "
                    "Maxwell-Boltzmann at T. Catches OOD force spikes before runaway.")
p.add_argument("--dtype", choices=["float32", "float64"], default="float32",
               help="MACE dtype; use float64 for the PMF surface potential.")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--min_len", type=float, default=0.0,
               help="replicate the cell so every lattice vector >= this length (A). "
                    "Ensures size-consistent q-sampling for comparable VDOS across "
                    "concentrations. 0 = no replication.")
args = p.parse_args()

os.makedirs(args.outdir, exist_ok=True)
np.random.seed(args.seed)

try:
    atoms = read(args.structure)
except Exception:
    atoms = read(args.structure, format="vasp")   # e.g. ToBeDelete_POSCAR

# replicate to reach a consistent minimum cell dimension (VDOS q-sampling)
if args.min_len > 0:
    reps = [max(1, int(np.ceil(args.min_len / L))) for L in atoms.cell.lengths()]
    if reps != [1, 1, 1]:
        atoms = atoms * reps
    print(f"supercell x{reps} -> {len(atoms)} atoms, "
          f"cell {np.round(atoms.cell.lengths(),2)} A")

calc = MACECalculator(model_paths=[args.model], device="cuda", default_dtype=args.dtype)
atoms.calc = calc

T_MAX = args.T * args.tmax_factor          # rescue threshold


def make_monitor(dyn, tag, counter, every=1000):
    """Per-step temperature watchdog: reset velocities on runaway (OOD spike),
    else print progress periodically. Returns the attached callback."""
    def m():
        Tk = atoms.get_temperature()
        if Tk > T_MAX:
            MaxwellBoltzmannDistribution(atoms, temperature_K=args.T)
            Stationary(atoms); ZeroRotation(atoms)
            counter[0] += 1
            if counter[0] <= 15 or counter[0] % 50 == 0:
                print(f"  [{tag}] step {dyn.nsteps} !RESCUE #{counter[0]} "
                      f"T={Tk:.0f}->{args.T:.0f} K", flush=True)
        elif dyn.nsteps % every == 0:
            print(f"  [{tag}] step {dyn.nsteps}  T={Tk:.0f} K", flush=True)
    return m


# 1. relax positions at fixed cell (loose fmax so rough surfaces don't spin)
if args.relax_steps > 0:
    print("relax ...", flush=True)
    FIRE(atoms, logfile=os.path.join(args.outdir, "relax.log")).run(
        fmax=0.05, steps=args.relax_steps)
    print(f"relax done, maxF={np.abs(atoms.get_forces()).max():.3f} eV/A", flush=True)
write(os.path.join(args.outdir, "relaxed.vasp"), atoms, format="vasp")

# 2. Langevin equilibration with per-step rescue (survive initial surface instability)
MaxwellBoltzmannDistribution(atoms, temperature_K=args.T)
Stationary(atoms); ZeroRotation(atoms)
dt = args.dt * units.fs
equil_rescues = [0]
equil = Langevin(atoms, timestep=dt, temperature_K=args.T, friction=args.friction)
equil.attach(make_monitor(equil, "equil", equil_rescues), interval=1)
print("equilibration ...", flush=True)
equil.run(args.equil)
print(f"equil done: {equil_rescues[0]} rescues", flush=True)

# 3. production with velocity recording + rescue counting (VDOS quality gate)
nat = len(atoms)
vel = np.zeros((args.prod, nat, 3), dtype=np.float32)
step = {"i": 0}
prod_rescues = [0]
if args.prod_thermostat:
    prod = Langevin(atoms, timestep=dt, temperature_K=args.T, friction=0.005)
else:
    prod = VelocityVerlet(atoms, timestep=dt)


def record():
    i = step["i"]
    if i < args.prod:
        vel[i] = atoms.get_velocities()      # Angstrom / (ASE time unit)
    step["i"] += 1


# monitor BEFORE record so a rescued (sane) velocity is what gets stored
prod.attach(make_monitor(prod, "prod", prod_rescues, every=2000), interval=1)
prod.attach(record, interval=1)
print("production ...", flush=True)
prod.run(args.prod - 1)
record()  # capture final

# VDOS-quality verdict
nr = prod_rescues[0]
if nr == 0:
    print(">>> production STABLE (0 rescues): VDOS is clean.", flush=True)
else:
    print(f">>> production had {nr} rescues ({100*nr/args.prod:.2f}% of steps). "
          f"A few near the start are OK; many => surface is OOD-unstable and the "
          f"VDOS is unreliable.", flush=True)

# velocities in Angstrom/fs for VDOS
vel_A_per_fs = vel * (units.fs)              # (A/ASE-time) * (ASE-time/fs) = A/fs
np.save(os.path.join(args.outdir, "velocities.npy"), vel_A_per_fs)
np.save(os.path.join(args.outdir, "symbols.npy"),
        np.array(atoms.get_chemical_symbols()))
np.save(os.path.join(args.outdir, "cell.npy"), np.array(atoms.cell))
with open(os.path.join(args.outdir, "md_info.txt"), "w") as f:
    f.write(f"structure={args.structure}\nmodel={args.model}\n"
            f"formula={atoms.get_chemical_formula()}\nn_atoms={nat}\n"
            f"dt_fs={args.dt}\nequil={args.equil}\nprod={args.prod}\nT={args.T}\n"
            f"equil_rescues={equil_rescues[0]}\nprod_rescues={prod_rescues[0]}\n")
print(f"done {atoms.get_chemical_formula()}: prod {args.prod} steps saved to {args.outdir}")
