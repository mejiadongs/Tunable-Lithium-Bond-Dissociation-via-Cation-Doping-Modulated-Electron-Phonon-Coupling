"""
WHAM analysis to reconstruct PMF from umbrella sampling.

Reads colvar_win_*.dat files produced by 03_run_umbrella_mace.py,
solves WHAM equations iteratively, outputs PMF.

Usage:
  python calc_pmf_wham.py --system Li2S
  python calc_pmf_wham.py --system Li3N
  python calc_pmf_wham.py --system LiTFSI
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from revision2_pmf import solve_wham

# ── Parameters ────────────────────────────────────────────────────────────────
SPRING_K  = 5.0    # eV/Å²  (must match run_umbrella_mace.py)
T         = 300.0  # K
KB        = 8.617333262e-5  # eV/K
BETA      = 1.0 / (KB * T)
N_BINS    = 100    # PMF histogram bins
SKIP_FRAC = 0.2    # skip first 20% of each colvar as equilibration
WHAM_ITER = 10000
WHAM_TOL  = 1e-8

BASE   = Path(__file__).parent
SYSTEMS = {
    "Li2S":   {"xi_min": 2.0, "xi_max": 5.0},
    "Li3N":   {"xi_min": 1.5, "xi_max": 4.0},
    "LiTFSI": {"xi_min": 2.5, "xi_max": 6.0},
}

parser = argparse.ArgumentParser()
parser.add_argument("--system", required=True, choices=list(SYSTEMS))
args = parser.parse_args()

cfg    = SYSTEMS[args.system]
indir  = BASE / "umbrella_mace" / args.system
outdir = indir

# ── Load colvar files ─────────────────────────────────────────────────────────
colvar_files = sorted(indir.glob("colvar_win_*.dat"))
if not colvar_files:
    raise FileNotFoundError(f"No colvar files in {indir}")

print(f"System  : {args.system}")
print(f"Windows : {len(colvar_files)}")

xi0_list   = []   # window centers (Å)
cv_data    = []   # list of 1D arrays of CV values

for f in colvar_files:
    xi0 = float(f.stem.replace("colvar_win_", ""))
    data = np.loadtxt(f, comments="#")
    cv   = data[:, 1]
    skip = int(len(cv) * SKIP_FRAC)
    cv   = cv[skip:]
    xi0_list.append(xi0)
    cv_data.append(cv)
    print(f"  win {xi0:.2f} Å : {len(cv)} samples  mean={cv.mean():.3f} std={cv.std():.3f}")

xi0_arr = np.array(xi0_list)
N_win   = len(xi0_arr)

# Shared, numerically stable WHAM: reject disconnected windows; mask empty bins.
bin_mids, pmf = solve_wham(cv_data, xi0_arr, SPRING_K, T, N_BINS)

# ── Save PMF ──────────────────────────────────────────────────────────────────
pmf_path = outdir / f"pmf_{args.system}.dat"
np.savetxt(
    pmf_path,
    np.column_stack([bin_mids, pmf]),
    header=f"CV(Ang)  PMF(eV)  system={args.system} T={T}K k={SPRING_K}eV/A2",
    fmt="%.6f",
)
print(f"\nPMF saved: {pmf_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: CV distributions per window
ax = axes[0]
for w, cv in enumerate(cv_data):
    h, edges = np.histogram(cv, bins=50, density=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(mids, h, lw=0.8, alpha=0.6)
ax.set_xlabel("CV (Å)")
ax.set_ylabel("Probability density")
ax.set_title(f"{args.system}: Window histograms")

# Right: PMF
ax = axes[1]
mask = np.isfinite(pmf)
ax.plot(bin_mids, pmf, "b-", lw=2)  # NaNs preserve gaps in sampled support.
ax.set_xlabel("CV (Å)")
ax.set_ylabel("PMF (eV)")
ax.set_title(f"{args.system}: PMF  (T={T} K)")
ax.axhline(0, color="gray", ls="--", lw=0.8)

fig.tight_layout()
fig_path = outdir / f"pmf_{args.system}.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Plot saved: {fig_path}")

barrier = pmf[mask].max()
print(f"\nSampled PMF range: {barrier:.3f} eV  ({barrier*96.485:.1f} kJ/mol)")

plt.show()
