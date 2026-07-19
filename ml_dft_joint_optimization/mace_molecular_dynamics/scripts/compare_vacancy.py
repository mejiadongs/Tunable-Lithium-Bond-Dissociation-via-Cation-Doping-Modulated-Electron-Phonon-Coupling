"""
Does the charge-compensating oxygen vacancy produce the low-frequency softening?

Compares three same-size cells:
    vac_pristine   Hf32O64     reference
    vac_2Sc_novac  Hf30Sc2O64  Sc substitution only (charge-uncompensated)
    vac_2Sc_Ovac   Hf30Sc2O63  Sc + V_O (charge-compensated, the real system)

Isolates the two effects:
    Sc effect      = (2Sc_novac) - (pristine)
    VACANCY effect = (2Sc_Ovac)  - (2Sc_novac)     <-- the question

EXPLORATORY: the potential has not been trained on bulk oxygen vacancies.
A large vacancy effect is suggestive and motivates a DFT check; a null result is
inconclusive (it may be the potential, not the physics).
"""
import os
import json
import numpy as np
_trapz = getattr(np, "trapezoid", np.trapz)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MD = os.environ["MD_DIR"]
VAC = os.path.join(MD, "vacancy")
CELLS = [("pristine", "vac_pristine"),
         ("2Sc (no vac)", "vac_2Sc_novac"),
         ("2Sc + V_O", "vac_2Sc_Ovac")]
CUTS = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
FMAX = 25.0
COL = {"pristine": "#7f8c8d", "2Sc (no vac)": "#2471a3", "2Sc + V_O": "#c0392b"}

data = {}
for label, sub in CELLS:
    d = os.path.join(VAC, sub)
    p = os.path.join(d, "vdos_curve.dat")
    sj = os.path.join(d, "summary.json")
    if not (os.path.isfile(p) and os.path.isfile(sj)):
        print(f"missing: {d}"); continue
    dat = np.loadtxt(p)
    cols = open(p).readline().lstrip("# ").split()
    data[label] = dict(freq=dat[:, cols.index("freq_THz")],
                       total=dat[:, cols.index("total")],
                       cols=cols, dat=dat, s=json.load(open(sj)))

if len(data) < 2:
    raise SystemExit("need at least pristine + one doped cell")

freq = next(iter(data.values()))["freq"]

# Table
print("\nLow-frequency weight W(<wc), total VDOS:")
hdr = f'{"cell":16s}' + "".join(("W_%gTHz" % c).rjust(10) for c in CUTS)
print(hdr)
for label, _ in CELLS:
    if label not in data:
        continue
    s = data[label]["s"]
    line = f"{label:16s}"
    for c in CUTS:
        line += f'{s.get("W_acoustic_%d" % int(c), float("nan")):10.4f}'
    print(line)

# effect decomposition
if all(k in data for k in ("pristine", "2Sc (no vac)", "2Sc + V_O")):
    print("\nEffect decomposition (total VDOS):")
    print(f'{"band":>10s}{"Sc effect":>14s}{"VACANCY effect":>16s}')
    for c in CUTS:
        k = "W_acoustic_%d" % int(c)
        p = data["pristine"]["s"].get(k, np.nan)
        n = data["2Sc (no vac)"]["s"].get(k, np.nan)
        v = data["2Sc + V_O"]["s"].get(k, np.nan)
        print(f'{"<%gTHz" % c:>10s}{n - p:>+14.4f}{v - n:>+16.4f}')

# Figure
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
m = freq <= 15
for label, _ in CELLS:
    if label not in data:
        continue
    ax[0].plot(freq[m], data[label]["total"][m], color=COL[label], lw=1.5, label=label)
ax[0].axvspan(0, 3, color="green", alpha=0.10)
ax[0].set_xlabel("Frequency (THz)"); ax[0].set_ylabel("Normalized total VDOS")
ax[0].set_title("(a) VDOS: Sc substitution vs Sc + O vacancy\n(green = acoustic branch)")
ax[0].legend(frameon=False)

if all(k in data for k in ("pristine", "2Sc (no vac)", "2Sc + V_O")):
    gp = data["pristine"]["total"]
    gn = data["2Sc (no vac)"]["total"]
    gv = data["2Sc + V_O"]["total"]
    ax[1].plot(freq[m], (gn - gp)[m], color=COL["2Sc (no vac)"], lw=1.4,
               label="Sc effect  (2Sc $-$ pristine)")
    ax[1].plot(freq[m], (gv - gn)[m], color=COL["2Sc + V_O"], lw=1.8,
               label="VACANCY effect  (+V$_O$ $-$ 2Sc)")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axvspan(0, 3, color="green", alpha=0.10)
    ax[1].set_xlabel("Frequency (THz)"); ax[1].set_ylabel(r"$\Delta g$")
    ax[1].set_title("(b) Which one softens the acoustic region?")
    ax[1].legend(frameon=False)

plt.tight_layout()
out = os.path.join(VAC, "fig_vacancy.png")
fig.savefig(out, dpi=200)
print("\nwrote", out)
