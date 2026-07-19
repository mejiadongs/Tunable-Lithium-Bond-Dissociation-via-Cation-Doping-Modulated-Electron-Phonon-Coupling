"""
Build the MACE fine-tuning dataset from the 300 K AIMD trajectories.
Reads energy+forces per ionic step from vasprun.xml (no stress in these NVT runs),
discards equilibration, subsamples to decorrelate, writes train/valid/test extxyz.
Paths come from env: AIMD_DIR (input) and DATA (output).
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase.io import iread, write

AIMD = os.environ["AIMD_DIR"]
OUT = os.environ["DATA"]
os.makedirs(OUT, exist_ok=True)

SYSTEMS = ["HfO2", "Sc-HfO2", "Y-HfO2"]
EQUIL, STRIDE, TEST_FRAC, VALID_FRAC = 1000, 20, 0.20, 0.10
E_KEY, F_KEY = "REF_energy", "REF_forces"


def load(system):
    path = os.path.join(AIMD, system, "ToBeDelete_vasprun.xml")
    frames = []
    for i, atoms in enumerate(iread(path, index=":")):
        if i < EQUIL or (i - EQUIL) % STRIDE:
            continue
        a = atoms.copy()
        a.info = {k: v for k, v in atoms.info.items() if k != "energy"}
        a.info[E_KEY] = float(atoms.get_potential_energy())
        a.arrays[F_KEY] = atoms.get_forces()
        a.info["config_type"] = system
        frames.append(a)
    return frames


rng = np.random.default_rng(0)
train, valid, test = [], [], []
for s in SYSTEMS:
    fr = load(s)
    n = len(fr); ncut = int(round(n * (1 - TEST_FRAC)))
    pool, held = fr[:ncut], fr[ncut:]
    idx = rng.permutation(len(pool)); nval = int(round(len(pool) * VALID_FRAC))
    vi, ti = set(idx[:nval].tolist()), set(idx[nval:].tolist())
    train += [pool[k] for k in sorted(ti)]
    valid += [pool[k] for k in sorted(vi)]
    test += held
    print(f"{s:9s}: sampled {n:4d} -> train {len(ti):4d} valid {nval:3d} test {len(held):3d}")

for name, data in [("train", train), ("valid", valid), ("test", test)]:
    write(os.path.join(OUT, f"{name}.xyz"), data, format="extxyz")
print(f"\nTOTAL train {len(train)} valid {len(valid)} test {len(test)}")
