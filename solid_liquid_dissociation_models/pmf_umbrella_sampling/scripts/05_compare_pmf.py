"""
Compare PMF curves from multiple umbrella sampling runs.

Usage:
  python compare_pmf.py --dirs umbrella_mace/Li2S umbrella_mace/Li2S_fixed
                        --labels "Li2S (free)" "Li2S (fixed)"
                        --out pmf_comparison.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── WHAM parameters (must match run_umbrella_mace.py) ────────────────────────
SPRING_K  = 5.0
T         = 300.0
KB        = 8.617333262e-5
BETA      = 1.0 / (KB * T)
N_BINS    = 100
SKIP_FRAC = 0.2
WHAM_ITER = 10000
WHAM_TOL  = 1e-8

def run_wham(indir: Path, label: str):
    colvar_files = sorted(indir.glob("colvar_win_*.dat"))
    if not colvar_files:
        raise FileNotFoundError(f"No colvar files in {indir}")
    print(f"\n[{label}] {len(colvar_files)} windows from {indir}")

    xi0_list, cv_data = [], []
    for f in colvar_files:
        xi0  = float(f.stem.replace("colvar_win_", ""))
        data = np.loadtxt(f, comments="#")
        cv   = data[:, 1]
        cv   = cv[int(len(cv) * SKIP_FRAC):]
        xi0_list.append(xi0)
        cv_data.append(cv)

    xi0_arr = np.array(xi0_list)
    N_win   = len(xi0_arr)
    xi_all  = np.concatenate(cv_data)
    bin_edges = np.linspace(xi_all.min() - 0.01, xi_all.max() + 0.01, N_BINS + 1)
    bin_mids  = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    H = np.zeros((N_win, N_BINS))
    for w, cv in enumerate(cv_data):
        H[w], _ = np.histogram(cv, bins=bin_edges)
    N_samples = np.array([len(cv) for cv in cv_data], dtype=float)

    U_bias = 0.5 * SPRING_K * (bin_mids[None, :] - xi0_arr[:, None]) ** 2
    F = np.zeros(N_win)

    for it in range(WHAM_ITER):
        denom = (N_samples[:, None] * np.exp(BETA * (F[:, None] - U_bias))).sum(axis=0)
        rho   = H.sum(axis=0) / (denom + 1e-300)
        F_new = -(1.0 / BETA) * np.log(
            (rho[None, :] * np.exp(-BETA * U_bias)).sum(axis=1) + 1e-300)
        F_new -= F_new[0]
        delta  = np.max(np.abs(F_new - F))
        F      = F_new
        if delta < WHAM_TOL:
            print(f"  WHAM converged at iter {it} (delta={delta:.2e})")
            break

    pmf = -(1.0 / BETA) * np.log(rho + 1e-300)
    pmf -= pmf.min()

    # Save dat file
    out_dat = indir / f"pmf_{indir.name}.dat"
    np.savetxt(out_dat,
               np.column_stack([bin_mids, pmf]),
               header=f"CV(Ang)  PMF(eV)  label={label}",
               fmt="%.6f")
    print(f"  Saved: {out_dat}")

    barrier = pmf[np.isfinite(pmf)].max()
    print(f"  Max PMF: {barrier:.3f} eV  ({barrier*96.485:.1f} kJ/mol)")

    return bin_mids, pmf, cv_data, xi0_list

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dirs",   nargs="+", required=True)
parser.add_argument("--labels", nargs="+", default=None)
parser.add_argument("--out",    default="pmf_comparison.png")
args = parser.parse_args()

dirs   = [Path(d) for d in args.dirs]
labels = args.labels if args.labels else [d.name for d in dirs]
assert len(dirs) == len(labels)

COLORS = ["#2166AC", "#D6604D", "#4DAC26", "#F4A582", "#762A83"]

# ── Run WHAM for each ─────────────────────────────────────────────────────────
results = []
for d, lbl in zip(dirs, labels):
    results.append((lbl, *run_wham(d, lbl)))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: histograms
ax = axes[0]
for (lbl, xi, pmf, cv_data, xi0_list), color in zip(results, COLORS):
    for cv in cv_data:
        h, edges = np.histogram(cv, bins=50, density=True)
        mids = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(mids, h, lw=0.6, alpha=0.4, color=color)
    ax.plot([], [], color=color, lw=2, label=lbl)   # legend entry
ax.set_xlabel("CV (Å)", fontsize=13)
ax.set_ylabel("Probability density", fontsize=13)
ax.set_title("Window histograms", fontsize=12)
ax.legend(fontsize=11)

# Right: PMF comparison
ax = axes[1]
for (lbl, xi, pmf, cv_data, xi0_list), color in zip(results, COLORS):
    mask = np.isfinite(pmf) & (pmf < 5.0)
    ax.plot(xi[mask], pmf[mask], lw=2.5, color=color, label=lbl)
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.set_xlabel("CV (Å)", fontsize=13)
ax.set_ylabel("PMF (eV)", fontsize=13)
ax.set_title(f"PMF comparison  (T={T} K)", fontsize=12)
ax.legend(fontsize=11)

fig.tight_layout()
out_path = Path(args.out)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
plt.show()
