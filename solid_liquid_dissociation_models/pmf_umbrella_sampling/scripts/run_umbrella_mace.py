"""
MACE umbrella sampling for PMF calculation.

Workflow:
  1. For each window, briefly pull CV to target distance (warm-up)
  2. Run NVT production MD with harmonic bias
  3. Save CV time series -> colvar_<label>_win_<xi>.dat

Usage:
  python run_umbrella_mace.py --system Li2S --model mace_pmf.model
  python run_umbrella_mace.py --system Li3N  --model mace_pmf.model
  python run_umbrella_mace.py --system LiTFSI --model mace_pmf.model

Output: umbrella_mace/<system>/colvar_win_X.XX.dat
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase import units
from mace.calculators import MACECalculator

# ── System definitions ────────────────────────────────────────────────────────
BASE = Path(__file__).parent

# CV atom indices (0-based) read from VASP ICONST files:
#   Li2S  : R 49 52  ->  i=48, j=51  (Li-S distance)
#   Li3N  : R 53 57  ->  i=52, j=56  (Li-N distance)
#   LiTFSI: R 64 61  ->  i=63, j=60  (S-N distance)
SYSTEMS = {
    "Li2S": {
        "contcar": None,   # must be provided via --contcar on server
        "cv_i": 48, "cv_j": 51,
        "xi_min": 2.2, "xi_max": 5.0,
    },
    "Li3N": {
        "contcar": None,
        "cv_i": 52, "cv_j": 56,
        "xi_min": 1.5, "xi_max": 4.0,
    },
    "LiTFSI": {
        "contcar": None,
        "cv_i": 63, "cv_j": 60,
        "xi_min": 2.5, "xi_max": 6.0,
    },
}

# ── MD parameters ─────────────────────────────────────────────────────────────
SPRING_K      = 5.0    # eV/Å²
T             = 300.0  # K
DT            = 0.5    # fs
N_WARMUP      = 2000   # steps to equilibrate each window
N_PROD        = 50000  # steps for production (= 50 ps)
SAVE_EVERY    = 10     # record CV every N steps

# ── Harmonic bias calculator ──────────────────────────────────────────────────
class HarmonicBias(Calculator):
    """Harmonic restraint on the distance between atoms i and j."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, k, r0, i, j, **kwargs):
        super().__init__(**kwargs)
        self.k  = k    # eV/Å²
        self.r0 = r0   # Å  (window center)
        self.i  = i    # 0-based index
        self.j  = j

    def calculate(self, atoms=None, properties=["energy", "forces"],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos  = atoms.get_positions()
        cell = atoms.get_cell()
        rij  = pos[self.j] - pos[self.i]

        # Minimum image
        if atoms.get_pbc().any():
            from ase.geometry import find_mic
            rij, _ = find_mic([rij], cell)
            rij = rij[0]

        r    = np.linalg.norm(rij)
        rhat = rij / r
        energy  = 0.5 * self.k * (r - self.r0) ** 2
        f_mag   = -self.k * (r - self.r0)
        forces  = np.zeros_like(pos)
        forces[self.j] += f_mag * rhat
        forces[self.i] -= f_mag * rhat

        self.results["energy"] = energy
        self.results["forces"] = forces

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_iconst(path):
    """Return (i0, j0) 0-based atom indices from VASP ICONST file."""
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "R":
                return int(parts[1]) - 1, int(parts[2]) - 1
    raise ValueError(f"No 'R' constraint in {path}")

def get_distance(atoms, i, j):
    pos  = atoms.get_positions()
    rij  = pos[j] - pos[i]
    if atoms.get_pbc().any():
        from ase.geometry import find_mic
        rij, _ = find_mic([rij], atoms.get_cell())
        rij = rij[0]
    return float(np.linalg.norm(rij))

# ── Main ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--system",  required=True, choices=list(SYSTEMS))
parser.add_argument("--model",   required=True)
parser.add_argument("--contcar", default=None,
                    help="Override CONTCAR path (default: from SYSTEMS dict)")
parser.add_argument("--iconst",  default=None,
                    help="Override ICONST path (default: same dir as CONTCAR)")
parser.add_argument("--device",  default="cuda")
args = parser.parse_args()

cfg = dict(SYSTEMS[args.system])   # copy so we can override

# --contcar overrides the default structure path
if args.contcar:
    cfg["contcar"] = Path(args.contcar)

if cfg["contcar"] is None:
    sys.exit(f"[ERROR] No CONTCAR set for {args.system}. Use --contcar <path>")

# Output dir is always named by system
outdir = BASE / "umbrella_mace" / args.system
outdir.mkdir(parents=True, exist_ok=True)

# Window centers: 0.1 Å spacing
xi_centers = np.round(np.arange(cfg["xi_min"], cfg["xi_max"] + 0.05, 0.1), 2)
print(f"System  : {args.system}")
print(f"Windows : {len(xi_centers)}  ({cfg['xi_min']} → {cfg['xi_max']} Å, Δ=0.1 Å)")
print(f"Model   : {args.model}")
print(f"Output  : {outdir}\n")

# CV atom indices (hardcoded from ICONST, 0-based)
i_cv = cfg["cv_i"]
j_cv = cfg["cv_j"]
print(f"CV atoms: {i_cv+1} – {j_cv+1} (1-based, VASP convention)\n")

# Load MACE calculator once
mace_calc = MACECalculator(
    model_paths=str(args.model),
    device=args.device,
    default_dtype="float64",
)

# Load initial structure and extract constraints (FixAtoms from F F F flags)
atoms0 = read(str(cfg["contcar"]))
r_init = get_distance(atoms0, i_cv, j_cv)
print(f"Initial CV distance: {r_init:.3f} Å")

# Save constraints to re-apply after loading extxyz frames (extxyz drops FixAtoms)
original_constraints = atoms0.constraints

def reapply_constraints(atoms):
    """Re-attach FixAtoms constraints (lost when loading from extxyz)."""
    atoms.set_constraint(original_constraints)
    return atoms

# Run each window sequentially (ladder: start from last frame of previous window)
atoms = atoms0.copy()
MaxwellBoltzmannDistribution(atoms, temperature_K=T)

for win_idx, xi0 in enumerate(xi_centers):
    label    = f"{xi0:.2f}"
    colvar_path = outdir / f"colvar_win_{label}.dat"

    if colvar_path.exists():
        print(f"[skip] window {label} Å already done")
        # Still need to load last frame for ladder
        last_frame_path = outdir / f"last_win_{label}.extxyz"
        if last_frame_path.exists():
            atoms = reapply_constraints(read(str(last_frame_path)))
        continue

    print(f"Window {win_idx+1:3d}/{len(xi_centers)}  xi0={label} Å ...", flush=True)

    # Attach combined calculator: MACE + harmonic bias
    bias = HarmonicBias(k=SPRING_K, r0=xi0, i=i_cv, j=j_cv)
    atoms.calc = SumCalculator([mace_calc, bias])

    # ── Pre-relaxation: remove bad forces before MD ───────────────────────
    from ase.optimize import FIRE
    print(f"  Pre-relaxing structure (fmax=0.5 eV/Å, max 200 steps)...", flush=True)
    atoms.calc = SumCalculator([mace_calc, bias])
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=0.5, steps=200)
    print(f"  Pre-relax done. CV={get_distance(atoms, i_cv, j_cv):.3f} Å", flush=True)

    # Re-initialize velocities after relaxation
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)

    friction = 0.01 / units.fs
    dyn = Langevin(atoms, timestep=DT*units.fs, temperature_K=T,
                   friction=friction, logfile=None)

    # ── Warm-up with progress ─────────────────────────────────────────────
    step_counter = [0]
    PRINT_EVERY  = 100   # print progress every N steps
    T_MAX        = T * 5   # rescale velocities if temperature exceeds this

    def progress():
        step_counter[0] += 1
        s = step_counter[0]
        T_k = atoms.get_temperature()
        if T_k > T_MAX:
            print(f"    [!RESCUE] step {s}  T={T_k:.0f} K -> rescaling to {T:.0f} K", flush=True)
            MaxwellBoltzmannDistribution(atoms, temperature_K=T)
        if s % PRINT_EVERY == 0:
            r   = get_distance(atoms, i_cv, j_cv)
            T_k = atoms.get_temperature()
            phase = "warmup" if s <= N_WARMUP else "prod  "
            print(f"    [{phase}] step {s:6d}  CV={r:.3f} Å  T={T_k:.0f} K",
                  flush=True)

    dyn.attach(progress, interval=1)
    dyn.run(N_WARMUP)

    # ── Production: record CV (written to disk in real time) ──────────────
    cv_record = []
    colvar_file = open(colvar_path, "w")
    colvar_file.write(f"# time(fs)  CV(Ang)  window={label}  k={SPRING_K}\n")
    prod_step = [0]

    def record():
        r = get_distance(atoms, i_cv, j_cv)
        t = prod_step[0] * DT * SAVE_EVERY
        cv_record.append(r)
        colvar_file.write(f"{t:.3f}  {r:.6f}\n")
        colvar_file.flush()
        prod_step[0] += 1

    dyn.attach(record, interval=SAVE_EVERY)
    dyn.run(N_PROD)
    colvar_file.close()

    # Save last frame for next window (ladder initialization)
    write(str(outdir / f"last_win_{label}.extxyz"), atoms)

    r_mean = np.mean(cv_record)
    r_std  = np.std(cv_record)
    print(f"  CV: mean={r_mean:.3f} std={r_std:.3f} Å  -> {colvar_path.name}")

print(f"\nDone. {len(xi_centers)} windows finished.")
print(f"Next: python calc_pmf_wham.py --system {args.system} --outdir {outdir}")













