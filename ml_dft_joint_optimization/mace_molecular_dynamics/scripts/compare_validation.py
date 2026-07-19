"""
Validation: overlay MACE-MD VDOS against DFT-AIMD VDOS for the 25% cells and
report the centroid/peak agreement. If these match, the fine-tuned potential is
trusted for the dilute concentration series.
Reads:  $WORK/dft_vdos/vdos_dft_<sys>.dat   (from vdos_dft_aimd.py)
        $MD_DIR/valid_<sys>/vdos_curve.dat   (from run_md.py + vdos_from_md.py)
"""
import os
import json
import numpy as np
_trapz = getattr(np, "trapezoid", np.trapz)  # numpy<2.0 lacks trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = os.environ["WORK"]
MD = os.environ["MD_DIR"]
DFT = os.path.join(WORK, "dft_vdos")
SYS = [("HfO2", "valid_HfO2"), ("Sc-HfO2", "valid_Sc"), ("Y-HfO2", "valid_Y")]


def load(path, key):
    dat = np.loadtxt(path)
    cols = open(path).readline().lstrip("# ").split()
    f = dat[:, cols.index("freq_THz")]
    k = key if key in cols else "total"
    return f, dat[:, cols.index(k)]


def centroid(f, g, wc=10.0):
    m = (f >= 0) & (f <= wc); return float(_trapz(f[m]*g[m], f[m])/_trapz(g[m], f[m]))


fig, ax = plt.subplots(1, 3, figsize=(14, 4))
print(f'{"system":9s}{"centroid_DFT":>14s}{"centroid_MACE":>14s}{"diff(THz)":>11s}')
for i, (name, sub) in enumerate(SYS):
    dft = os.path.join(DFT, f"vdos_dft_{name}.dat")
    mace = os.path.join(MD, sub, "vdos_curve.dat")
    if not (os.path.isfile(dft) and os.path.isfile(mace)):
        print(f"{name}: missing ({dft} or {mace})"); continue
    key = "local" if name != "HfO2" else "total"
    fd, gd = load(dft, key); fm, gm = load(mace, key)
    md_ = fd <= 20
    ax[i].plot(fd[md_], gd[md_], "k-", lw=1.5, label="DFT-AIMD")
    mm = fm <= 20
    ax[i].plot(fm[mm], gm[mm], "r--", lw=1.5, label="MACE-MD")
    ax[i].axvspan(0, 10, color="green", alpha=0.06)
    ax[i].set_title(name); ax[i].set_xlabel("Frequency (THz)"); ax[i].legend(frameon=False)
    cd, cm = centroid(fd, gd), centroid(fm, gm)
    print(f"{name:9s}{cd:14.3f}{cm:14.3f}{cm-cd:11.3f}")
ax[0].set_ylabel("Normalized VDOS")
plt.tight_layout()
out = os.path.join(WORK, "fig_validation_25pct.png")
plt.savefig(out, dpi=200)
print("wrote", out)
