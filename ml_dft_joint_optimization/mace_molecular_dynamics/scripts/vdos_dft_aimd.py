"""
Reference VDOS from the DFT-AIMD trajectories (XDATCAR), for the 25% cells.
Used to validate the MACE-MD VDOS. Same velocity-power-spectrum method and the
same 0-10 THz centroid / local-peak descriptor as vdos_from_md.py.
Reads AIMD_DIR from env; writes vdos_dft_<system>.dat + a summary line.
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import signal
_trapz = getattr(np, "trapezoid", np.trapz)  # numpy<2.0 lacks trapezoid

AIMD = os.environ["AIMD_DIR"]
OUT = os.path.join(os.environ["WORK"], "dft_vdos")
os.makedirs(OUT, exist_ok=True)
DT_FS, N_EQUIL, WC, FMAX, O_CUT = 1.0, 1000, 10.0, 25.0, 2.7
SYSTEMS = [("HfO2", None), ("Sc-HfO2", "Sc"), ("Y-HfO2", "Y")]


def read_xdatcar(path):
    lines = open(path).readlines()
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)]) * scale
    elems = lines[5].split(); counts = [int(x) for x in lines[6].split()]
    symbols = [e for e, c in zip(elems, counts) for _ in range(c)]
    nat = sum(counts); frames = []; i = 7; n = len(lines)
    while i < n:
        if lines[i].lstrip().startswith("Direct"):
            block = lines[i + 1:i + 1 + nat]
            if len(block) < nat:
                break
            frames.append([[float(v) for v in r.split()[:3]] for r in block])
            i += 1 + nat
        else:
            i += 1
    return lat, np.array(symbols), np.array(frames)


def velocities(frac, lat):
    d = np.diff(frac, axis=0); d -= np.round(d)
    disp = np.concatenate([np.zeros((1,) + d.shape[1:]), np.cumsum(d, axis=0)], 0)
    return np.gradient((frac[0] + disp) @ lat, DT_FS, axis=0)


def vdos(v, idx, nperseg=4096):
    nperseg = min(nperseg, v.shape[0]); acc = None
    for a in idx:
        for c in range(3):
            f, P = signal.welch(v[:, a, c], fs=1.0 / DT_FS, nperseg=nperseg, detrend="constant")
            acc = P if acc is None else acc + P
    return f * 1000.0, acc


def centroid(f, g, wc=WC):
    m = (f >= 0) & (f <= wc); return float(_trapz(f[m] * g[m], f[m]) / _trapz(g[m], f[m]))


def peak(f, g, lo=1.0, wc=WC):
    m = (f >= lo) & (f <= wc); return float(f[m][np.argmax(g[m])])


for name, dop in SYSTEMS:
    lat, sym, frac = read_xdatcar(os.path.join(AIMD, name, "ToBeDelete_XDATCAR"))
    frac = frac[N_EQUIL:]; v = velocities(frac, lat)
    freq, g_tot = vdos(v, np.arange(v.shape[1]))
    res = dict(system=name, dopant=dop, centroid_total=centroid(freq, g_tot), peak_total=peak(freq, g_tot))
    m = freq <= FMAX; cols = [freq, g_tot / _trapz(g_tot[m], freq[m])]; hdr = ["freq_THz", "total"]
    if dop:
        dop_idx = np.where(sym == dop)[0]; o_idx = np.where(sym == "O")[0]
        mean = frac.mean(0); nn = set()
        for c in dop_idx:
            df = mean[o_idx] - mean[c]; df -= np.round(df)
            nn.update(o_idx[np.linalg.norm(df @ lat, axis=1) < O_CUT].tolist())
        loc = np.concatenate([dop_idx, np.array(sorted(nn), dtype=int)])
        _, g_loc = vdos(v, loc)
        res["centroid_local"] = centroid(freq, g_loc); res["peak_local"] = peak(freq, g_loc)
        cols.append(g_loc / _trapz(g_loc[m], freq[m])); hdr.append("local")
    np.savetxt(os.path.join(OUT, f"vdos_dft_{name}.dat"), np.column_stack(cols), header=" ".join(hdr))
    print(json.dumps(res))
