"""
Projected VDOS + low-frequency descriptor from a run_md.py output directory.
Descriptor (matches the paper's 0-10 THz EPC statement):
  - centroid (first moment) of VDOS over 0-10 THz  -> red-shift on doping
  - dominant local (dopant + nearest-O) peak position over 1-10 THz
Writes summary.json and vdos_curve.dat inside the run directory.
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
_trapz = getattr(np, "trapezoid", np.trapz)  # numpy<2.0 lacks trapezoid
from scipy import signal

WC = 10.0          # low-frequency window (THz) per the manuscript's lambda_low
FMAX = 25.0
O_CUT = 2.7


def read_md(outdir):
    v = np.load(os.path.join(outdir, "velocities.npy"))       # (nt, nat, 3) A/fs
    sym = np.load(os.path.join(outdir, "symbols.npy"))
    cell = np.load(os.path.join(outdir, "cell.npy"))
    # positions from relaxed structure for neighbor detection
    from ase.io import read
    at = read(os.path.join(outdir, "relaxed.vasp"))
    dt_fs = 1.0
    with open(os.path.join(outdir, "md_info.txt")) as f:
        for line in f:
            if line.startswith("dt_fs"):
                dt_fs = float(line.split("=")[1])
    return v, sym, cell, at, dt_fs


def groups(sym, at):
    sym = np.array(sym)
    dop = None
    for s in ("Sc", "Y"):
        if s in sym:
            dop = s
    g = {"all": np.arange(len(sym)), "Hf": np.where(sym == "Hf")[0],
         "O": np.where(sym == "O")[0]}
    if dop:
        dop_idx = np.where(sym == dop)[0]
        frac = at.get_scaled_positions()
        cell = np.array(at.cell)
        o_idx = np.where(sym == "O")[0]
        nn = set()
        for c in dop_idx:
            df = frac[o_idx] - frac[c]; df -= np.round(df)
            d = np.linalg.norm(df @ cell, axis=1)
            nn.update(o_idx[d < O_CUT].tolist())
        g[dop] = dop_idx
        g["O_NN"] = np.array(sorted(nn), dtype=int)
        g["local"] = np.concatenate([dop_idx, g["O_NN"]])
        g["dopant"] = dop
    return g, dop


def vdos(v, idx, dt_fs, nperseg=4096):
    nt = v.shape[0]
    nperseg = min(nperseg, nt)
    acc = None
    for a in idx:
        for c in range(3):
            f, P = signal.welch(v[:, a, c], fs=1.0 / dt_fs, nperseg=nperseg,
                                detrend="constant")
            acc = P if acc is None else acc + P
    return f * 1000.0, acc          # THz, power


def centroid(freq, g, wc=WC):
    m = (freq >= 0) & (freq <= wc)
    return float(_trapz(freq[m] * g[m], freq[m]) / _trapz(g[m], freq[m]))


def peak(freq, g, lo=1.0, wc=WC):
    m = (freq >= lo) & (freq <= wc)
    return float(freq[m][np.argmax(g[m])])


def wfrac(freq, g, wc):
    """acoustic-window spectral-weight fraction: int_0^wc g / int_0^FMAX g."""
    tot = _trapz(g[freq <= FMAX], freq[freq <= FMAX])
    return float(_trapz(g[freq <= wc], freq[freq <= wc]) / tot) if tot > 0 else float("nan")


# Low-frequency cutoffs, THz. 3-5 THz = the acoustic branch (edge ~3 THz from the
# DFT dispersion); 10 THz = the window used to define lambda_low in the manuscript.
# The full sweep is reported so the cutoff choice is transparent.
ACOUSTIC_CUTS = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def analyze(outdir):
    v, sym, cell, at, dt_fs = read_md(outdir)
    g, dop = groups(sym, at)
    freq, g_tot = vdos(v, g["all"], dt_fs)
    out = {"outdir": outdir, "formula": at.get_chemical_formula(),
           "n_atoms": len(sym), "dopant": dop, "window_THz": WC,
           "centroid_total": centroid(freq, g_tot),
           "peak_total": peak(freq, g_tot)}
    # acoustic-window spectral weight (headline descriptor: EPC enhancement = acoustic)
    for wc in ACOUSTIC_CUTS:
        out[f"W_acoustic_{wc:.0f}"] = wfrac(freq, g_tot, wc)
    curves = {"freq_THz": freq, "total": g_tot}
    if dop:
        _, g_loc = vdos(v, g["local"], dt_fs)
        _, g_dop = vdos(v, g[dop], dt_fs)
        onn = g["O_NN"]
        out["centroid_local"] = centroid(freq, g_loc)
        out["peak_local"] = peak(freq, g_loc)
        out["peak_dopant"] = peak(freq, g_dop)
        curves["local"] = g_loc
        curves["dopant"] = g_dop
    else:
        # pristine reference: an equivalent cation set (Hf) + its O shell, so the
        # local/cation descriptors are compared like-for-like with the doped cells.
        hf = g["Hf"][:8]
        frac = at.get_scaled_positions(); cellm = np.array(at.cell)
        o_idx = g["O"]; nn = set()
        for c in hf:
            df = frac[o_idx] - frac[c]; df -= np.round(df)
            nn.update(o_idx[np.linalg.norm(df @ cellm, axis=1) < O_CUT].tolist())
        onn = np.array(sorted(nn), dtype=int)
        _, g_loc = vdos(v, np.array(sorted(nn) + hf.tolist()), dt_fs)
        _, g_dop = vdos(v, hf, dt_fs)          # cation-only reference
        out["centroid_local"] = centroid(freq, g_loc)
        out["peak_local"] = peak(freq, g_loc)
        curves["local"] = g_loc
        curves["dopant"] = g_dop

    # ---- MASS-SCREENED descriptors -------------------------------------------
    # O has the same mass (16 amu) in every system, so any change in an O-projected
    # spectrum is a pure force-constant (bond-softening) effect, with the cation mass
    # effect (Hf 178.5 -> Sc 45 / Y 89) removed by construction. This is also the
    # sublattice that dominates the polar modes probed by THz spectroscopy.
    _, g_O = vdos(v, g["O"], dt_fs)            # all oxygen
    _, g_ONN = vdos(v, onn, dt_fs)             # oxygen shell around the dopant / ref cation
    out["centroid_O"] = centroid(freq, g_O)
    out["centroid_ONN"] = centroid(freq, g_ONN)
    curves["O_all"] = g_O
    curves["O_NN"] = g_ONN

    # acoustic weights: cell average, local (dopant+O), cation-only, and O-projected
    for wc in ACOUSTIC_CUTS:
        out[f"W_acoustic_local_{wc:.0f}"] = wfrac(freq, g_loc, wc)
        out[f"W_acoustic_dopant_{wc:.0f}"] = wfrac(freq, g_dop, wc)
        out[f"W_acoustic_O_{wc:.0f}"] = wfrac(freq, g_O, wc)
        out[f"W_acoustic_ONN_{wc:.0f}"] = wfrac(freq, g_ONN, wc)

    # save normalized curves (0-FMAX area = 1)
    m = freq <= FMAX
    hdr = ["freq_THz"]; cols = [freq]
    for k, val in curves.items():
        if k == "freq_THz":
            continue
        norm = val / _trapz(val[m], freq[m])
        hdr.append(k); cols.append(norm)
    np.savetxt(os.path.join(outdir, "vdos_curve.dat"), np.column_stack(cols),
               header=" ".join(hdr))
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    analyze(sys.argv[1])
