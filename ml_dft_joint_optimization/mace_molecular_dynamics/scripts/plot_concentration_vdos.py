"""
Overlay concentration-dependent VDOS and plot the 0-10 THz red-shift descriptor.
Scans a parent directory for run subdirs (each with summary.json + vdos_curve.dat),
groups by dopant, sorts by cation fraction, and produces:
  (a) normalized local VDOS overlay (0-15 THz) per Sc concentration
  (b) 0-10 THz VDOS centroid vs concentration (red-shift)
  (c) local dominant peak position vs concentration
Usage: plot_concentration_vdos.py <parent_dir>
"""
import os
import sys
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cation_fraction(formula):
    def n(sym):
        m = re.search(sym + r"(\d*)", formula)
        if not m:
            return 0
        return int(m.group(1)) if m.group(1) else (1 if sym in formula else 0)
    nhf, nsc, ny = n("Hf"), n("Sc"), n("Y")
    nd = nsc + ny
    return nd / (nd + nhf) if (nd + nhf) else 0.0


parent = sys.argv[1]
runs = []
for d in sorted(os.listdir(parent)):
    sj = os.path.join(parent, d, "summary.json")
    if os.path.isfile(sj):
        s = json.load(open(sj))
        s["dir"] = os.path.join(parent, d)
        s["x"] = cation_fraction(s["formula"]) * 100
        runs.append(s)

sc = sorted([r for r in runs if r["dopant"] == "Sc"], key=lambda r: r["x"])
y = sorted([r for r in runs if r["dopant"] == "Y"], key=lambda r: r["x"])
prist = [r for r in runs if r["dopant"] in (None, "null")]

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

# (a) TOTAL VDOS overlay for Sc series (consistent with the total-centroid descriptor)
def load_curve(r):
    dat = np.loadtxt(os.path.join(r["dir"], "vdos_curve.dat"))
    with open(os.path.join(r["dir"], "vdos_curve.dat")) as f:
        cols = f.readline().lstrip("# ").split()
    return dat, cols

cmap = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(sc), 1)))
for c, r in zip(cmap, sc):
    dat, cols = load_curve(r)
    f_ax = dat[:, cols.index("freq_THz")]; m = f_ax <= 15
    ax[0].plot(f_ax[m], dat[m, cols.index("total")], color=c, label=f'{r["x"]:.2f}%')
if prist:
    dat, cols = load_curve(prist[0])
    f_ax = dat[:, cols.index("freq_THz")]; m = f_ax <= 15
    ax[0].plot(f_ax[m], dat[m, cols.index("total")], "k--", label="pristine")
ax[0].axvspan(0, 4, color="green", alpha=0.10)   # acoustic window
ax[0].set_xlabel("Frequency (THz)"); ax[0].set_ylabel("Normalized total VDOS")
ax[0].set_title("(a) Total VDOS vs concentration (acoustic window shaded)")
ax[0].legend(frameon=False, fontsize=8)

# (b) HEADLINE: low-frequency (acoustic-dominated) spectral weight, <4 THz.
# Trend is cutoff-independent (panel c shows 3/4/5 THz); 4 THz keeps the acoustic
# branch (edge ~3 THz) well-sampled in the finite MD cell.
WKEY = "W_acoustic_4"
for series, col, mk, lab in [(sc, "#c0392b", "o", "Sc"), (y, "#2471a3", "s", "Y")]:
    if series:
        ax[1].plot([r["x"] for r in series], [r[WKEY] for r in series],
                   mk + "-", color=col, label=lab)
if prist:
    ax[1].axhline(prist[0][WKEY], ls="--", color="gray", label="pristine")
ax[1].set_xlabel("Cation doping fraction (%)")
ax[1].set_ylabel("Low-freq. weight $W_{ac}$ (<4 THz)")
ax[1].set_title("(b) Acoustic spectral weight vs concentration"); ax[1].legend(frameon=False)

# (c) LOCAL (dopant + nearest-O) acoustic weight — the environment the Li bond couples
# to. The cell average in (b) is diluted by the undoped matrix and by the dopant mass
# effect (Sc 45, Y 89 vs Hf 178.5 amu), so it is NOT the quantity of interest.
LKEY = "W_acoustic_local_4"
for series, colr, mk, lab in [(sc, "#c0392b", "o", "Sc"), (y, "#2471a3", "s", "Y")]:
    if series:
        ax[2].plot([r["x"] for r in series], [r.get(LKEY, np.nan) for r in series],
                   mk + "-", color=colr, label=f"{lab} + O$_{{NN}}$")
if prist:
    ax[2].axhline(prist[0].get(LKEY, np.nan), ls="--", color="gray",
                  label="pristine (Hf + O$_{NN}$)")
ax[2].set_xlabel("Cation doping fraction (%)")
ax[2].set_ylabel("Local acoustic weight $W_{ac}^{loc}$ (<4 THz)")
ax[2].set_title("(c) Dopant-local acoustic weight"); ax[2].legend(frameon=False, fontsize=8)

plt.tight_layout()
out = os.path.join(parent, "fig_concentration_vdos.png")
fig.savefig(out, dpi=200)
print("wrote", out)
# FULL cutoff sweep of the total (cell-average) low-frequency weight.
# 3-5 THz = acoustic branch; 10 THz = the manuscript's lambda_low window.
print("\nTotal (cell-average) low-frequency weight vs cutoff:")
print(f'{"dopant":7s}{"x%":>8s}' + "".join(f'{f"W_{c:.0f}THz":>10s}' for c in (3, 4, 5, 6, 8, 10)))
for r in sc + y + prist:
    g = lambda k: r.get(k, float("nan"))
    print(f'{str(r["dopant"]):7s}{r["x"]:8.3f}' +
          "".join(f'{g(f"W_acoustic_{c:.0f}"):10.4f}' for c in (3, 4, 5, 6, 8, 10)))

print("\nO-projected (mass-screened) low-frequency weight vs cutoff:")
print(f'{"dopant":7s}{"x%":>8s}' + "".join(f'{f"O_{c:.0f}THz":>10s}' for c in (3, 4, 5, 6, 8, 10))
      + f'{"cen_ONN":>9s}')
for r in sc + y + prist:
    g = lambda k: r.get(k, float("nan"))
    print(f'{str(r["dopant"]):7s}{r["x"]:8.3f}' +
          "".join(f'{g(f"W_acoustic_O_{c:.0f}"):10.4f}' for c in (3, 4, 5, 6, 8, 10))
          + f'{g("centroid_ONN"):9.3f}')
