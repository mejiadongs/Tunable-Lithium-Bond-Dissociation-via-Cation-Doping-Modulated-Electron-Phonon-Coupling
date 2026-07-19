"""
Overlay slab VDOS: surface vs interior for pristine and Sc slabs, plus surface Sc.
Reads $MD_DIR/slab_HfO2 and $MD_DIR/slab_ScHfO2 (vdos_surface.dat + summary_surface.json).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MD = os.environ["MD_DIR"]
runs = {"HfO2": os.path.join(MD, "slab_HfO2"),
        "Sc-HfO2": os.path.join(MD, "slab_ScHfO2"),
        "Y-HfO2": os.path.join(MD, "slab_YHfO2")}
COL = {"HfO2": "#7f8c8d", "Sc-HfO2": "#c0392b", "Y-HfO2": "#2471a3"}


def load(d):
    p = os.path.join(d, "vdos_surface.dat")
    dat = np.loadtxt(p); cols = open(p).readline().lstrip("# ").split()
    return dat, cols


fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
print(f'{"slab":9s}{"Wac_interior":>13s}{"Wac_surface":>12s}{"Wac_dopant":>11s}'
      f'{"cen_surf":>10s}')
for name, d in runs.items():
    if not os.path.isdir(d):
        print(f"{name}: missing {d}"); continue
    dat, cols = load(d); f = dat[:, cols.index("freq_THz")]; m = f <= 15
    s = json.load(open(os.path.join(d, "summary_surface.json")))
    print(f'{name:9s}{s["Wac_interior"]:13.4f}{s["Wac_surface"]:12.4f}'
          f'{s.get("Wac_dopant_local", float("nan")):11.4f}{s["centroid_surface"]:10.3f}')
    col = COL[name]
    ax[0].plot(f[m], dat[m, cols.index("interior")], color=col, ls="--", lw=1.3,
               label=f"{name} interior")
    ax[0].plot(f[m], dat[m, cols.index("surface")], color=col, ls="-", lw=1.8,
               label=f"{name} surface")
    # panel (b): surface dopant-projected local mode (Sc and Y)
    if "dopant_local" in cols:
        ax[1].plot(f[m], dat[m, cols.index("dopant_local")], color=col, lw=1.8,
                   label=f"surface {s.get('dopant','?')} + O$_{{NN}}$")
# reference: pristine surface Hf+O in panel (b)
if os.path.isdir(runs["HfO2"]):
    dat, cols = load(runs["HfO2"]); f = dat[:, cols.index("freq_THz")]; m = f <= 15
    ax[1].plot(f[m], dat[m, cols.index("surface")], color=COL["HfO2"], lw=1.2, ls=":",
               label="pristine surface")

for a in ax:
    a.axvspan(0, 4, color="green", alpha=0.10)   # acoustic window
    a.set_xlabel("Frequency (THz)"); a.set_ylabel("Normalized VDOS"); a.legend(frameon=False, fontsize=8)
ax[0].set_title("(a) Surface vs interior (bulk->surface bridge)")
ax[1].set_title("(b) Surface dopant local mode (Sc vs Y)")
plt.tight_layout()
out = os.path.join(MD, "fig_slab_vdos.png")
plt.savefig(out, dpi=200)
print("wrote", out)
