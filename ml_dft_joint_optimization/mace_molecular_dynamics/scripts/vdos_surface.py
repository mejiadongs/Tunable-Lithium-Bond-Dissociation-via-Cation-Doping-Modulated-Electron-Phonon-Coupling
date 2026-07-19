"""
Surface-projected VDOS from a slab MD run (run_md.py output dir).
Bins atoms by z into SURFACE (within ZS of either slab boundary) and INTERIOR,
plus the dopant + its nearest O. Reports the 0-10 THz centroid/peak for each.

Two things this shows:
  - INTERIOR VDOS should reproduce the bulk MD-VDOS -> free consistency check
    (no slab DFT needed): if interior==bulk, the potential behaves on the slab.
  - SURFACE (and surface-Sc) VDOS red-shift vs interior -> the bulk low-frequency
    enhancement carried to the surface = the bulk->interface bridge.
Usage: vdos_surface.py <md_outdir>
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
_trapz = getattr(np, "trapezoid", np.trapz)  # numpy<2.0 lacks trapezoid
from scipy import signal
from ase.io import read

ZS = 3.5           # surface shell thickness (Angstrom) from each slab boundary
WC, FMAX, O_CUT = 10.0, 25.0, 2.7


def vdos(v, idx, dt_fs=1.0, nperseg=4096):
    nperseg = min(nperseg, v.shape[0]); acc = None
    for a in idx:
        for c in range(3):
            f, P = signal.welch(v[:, a, c], fs=1.0/dt_fs, nperseg=nperseg, detrend="constant")
            acc = P if acc is None else acc + P
    return f*1000.0, acc


def centroid(f, g, wc=WC):
    m = (f >= 0) & (f <= wc); return float(_trapz(f[m]*g[m], f[m])/_trapz(g[m], f[m]))


def peak(f, g, lo=1.0, wc=WC):
    m = (f >= lo) & (f <= wc); return float(f[m][np.argmax(g[m])])


def wfrac(f, g, wc=4.0):
    """acoustic-window weight fraction (same descriptor as the bulk series)."""
    tot = _trapz(g[f <= FMAX], f[f <= FMAX])
    return float(_trapz(g[f <= wc], f[f <= wc]) / tot) if tot > 0 else float("nan")


outdir = sys.argv[1]
v = np.load(os.path.join(outdir, "velocities.npy"))
sym = np.array(np.load(os.path.join(outdir, "symbols.npy")))
at = read(os.path.join(outdir, "relaxed.vasp"))
z = at.positions[:, 2]
zlo, zhi = z.min(), z.max()

surf = np.where((z < zlo + ZS) | (z > zhi - ZS))[0]
interior = np.where((z >= zlo + ZS) & (z <= zhi - ZS))[0]

freq, g_surf = vdos(v, surf)
_, g_int = vdos(v, interior)
res = {"outdir": outdir, "formula": at.get_chemical_formula(), "ZS": ZS,
       "n_surface": int(len(surf)), "n_interior": int(len(interior)),
       "centroid_surface": centroid(freq, g_surf), "peak_surface": peak(freq, g_surf),
       "centroid_interior": centroid(freq, g_int), "peak_interior": peak(freq, g_int),
       "Wac_surface": wfrac(freq, g_surf), "Wac_interior": wfrac(freq, g_int)}

cols = [freq]; hdr = ["freq_THz"]; m = freq <= FMAX
for name, g in [("surface", g_surf), ("interior", g_int)]:
    cols.append(g/_trapz(g[m], freq[m])); hdr.append(name)

frac = at.get_scaled_positions(); cell = np.array(at.cell)
o_idx = np.where(sym == "O")[0]


def cation_shell(centers):
    """indices of the given cations + their first-shell O."""
    nn = set()
    for c in centers:
        df = frac[o_idx] - frac[c]; df -= np.round(df)
        nn.update(o_idx[np.linalg.norm(df @ cell, axis=1) < O_CUT].tolist())
    return np.concatenate([np.asarray(centers), np.array(sorted(nn), dtype=int)])


dop = "Sc" if "Sc" in sym else ("Y" if "Y" in sym else None)
if dop:
    centers = np.where(sym == dop)[0]
else:
    # PRISTINE REFERENCE: an equivalent SURFACE Hf (+ its O shell), so the
    # dopant-local number has a like-for-like counterpart. Without this the
    # doped "Wac_dopant_local" has nothing valid to be compared against.
    hf_surf = np.array([i for i in surf if sym[i] == "Hf"])
    z_hf = at.positions[hf_surf, 2]
    centers = hf_surf[np.argsort(-z_hf)][:1]      # topmost surface Hf

if len(centers):
    loc = cation_shell(centers)
    _, g_loc = vdos(v, loc)
    res["dopant"] = dop if dop else "Hf(ref)"
    res["n_local"] = int(len(loc))
    res["centroid_dopant_local"] = centroid(freq, g_loc)
    res["peak_dopant_local"] = peak(freq, g_loc)
    res["Wac_dopant_local"] = wfrac(freq, g_loc)
    cols.append(g_loc/_trapz(g_loc[m], freq[m])); hdr.append("dopant_local")

np.savetxt(os.path.join(outdir, "vdos_surface.dat"), np.column_stack(cols), header=" ".join(hdr))
with open(os.path.join(outdir, "summary_surface.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
