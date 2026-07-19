"""
Summarize the charge-compensated (2M + 1 V_O) concentration series:
low-frequency weight W(<wc) vs dopant concentration, for Sc and Y, against pristine.

Prints the full cutoff sweep (3-10 THz) so the window choice stays transparent, and
plots W vs concentration at the acoustic (4 THz) and lambda_low (10 THz) windows.
Also writes vac_series.csv for Origin.
"""
import os
import re
import csv
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MD = os.environ["MD_DIR"]
VS = os.path.join(MD, "vac_series")
CUTS = (3, 4, 5, 6, 8, 10)


def parse(name):
    m = re.match(r"vs_(Sc|Y|pristine)_([\d.]+)", name)
    if not m:
        return None, None
    who = m.group(1)
    x = 0.0 if who == "pristine" else float(m.group(2))
    return who, x


runs = []
for d in sorted(glob.glob(os.path.join(VS, "vs_*"))):
    sj = os.path.join(d, "summary.json")
    if not os.path.isfile(sj):
        continue
    who, x = parse(os.path.basename(d))
    if who is None:
        continue
    s = json.load(open(sj))
    s["who"] = who; s["x"] = x; s["name"] = os.path.basename(d)
    runs.append(s)

if not runs:
    raise SystemExit("no runs found in " + VS)

sc = sorted([r for r in runs if r["who"] == "Sc"], key=lambda r: r["x"])
y = sorted([r for r in runs if r["who"] == "Y"], key=lambda r: r["x"])
pr = [r for r in runs if r["who"] == "pristine"]

print("\nCharge-compensated series (2M + 1 V_O) - low-frequency weight W(<wc):")
hdr = f'{"cell":22s}{"x%":>8s}' + "".join(("W_%dTHz" % c).rjust(10) for c in CUTS)
print(hdr)
for r in pr + sc + y:
    line = f'{r["name"]:22s}{r["x"]:8.3f}'
    for c in CUTS:
        line += f'{r.get("W_acoustic_%d" % c, float("nan")):10.4f}'
    print(line)

# pristine reference (prefer the 96-atom one; report both so size drift is visible)
p96 = next((r for r in pr if "96" in r["name"]), None)
p288 = next((r for r in pr if "288" in r["name"]), None)
if p96 and p288:
    print("\nPristine size check (96 vs 288 atoms) - any drift is a size artefact:")
    for c in CUTS:
        a = p96.get("W_acoustic_%d" % c, np.nan); b = p288.get("W_acoustic_%d" % c, np.nan)
        print(f"  W_{c}THz: 96-atom {a:.4f}   288-atom {b:.4f}   diff {b-a:+.4f}")

with open(os.path.join(VS, "vac_series.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cell", "dopant", "cation_pct"] + ["W_%dTHz" % c for c in CUTS])
    for r in pr + sc + y:
        w.writerow([r["name"], r["who"], f'{r["x"]:.3f}'] +
                   [f'{r.get("W_acoustic_%d" % c, float("nan")):.5f}' for c in CUTS])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k, wc in enumerate((4, 10)):
    key = "W_acoustic_%d" % wc
    for s, c, mk, lab in [(sc, "#c0392b", "o", "Sc + V$_O$"), (y, "#2471a3", "s", "Y + V$_O$")]:
        if s:
            ax[k].plot([r["x"] for r in s], [r.get(key, np.nan) for r in s], mk + "-",
                       color=c, label=lab)
    if p96:
        ax[k].axhline(p96.get(key, np.nan), ls="--", color="gray", label="pristine (96)")
    if p288:
        ax[k].axhline(p288.get(key, np.nan), ls=":", color="gray", label="pristine (288)")
    ax[k].axvline(2.083, ls=":", color="green", lw=1)
    ax[k].set_xscale("log")
    ax[k].set_xlabel("Cation doping fraction (%)")
    ax[k].set_ylabel(f"W(<{wc} THz)")
    ax[k].set_title(f"({'ab'[k]}) charge-compensated series, <{wc} THz")
    ax[k].legend(frameon=False, fontsize=8)
plt.tight_layout()
out = os.path.join(VS, "fig_vac_series.png")
fig.savefig(out, dpi=200)
print("\nwrote", out)
print("wrote", os.path.join(VS, "vac_series.csv"))
