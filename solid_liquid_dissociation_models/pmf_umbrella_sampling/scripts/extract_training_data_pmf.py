"""
Extract VASP AIMD training data from PMF umbrella-sampling OUTCARs.

Auto-discovers all OUTCAR files under PMF_multiwindows/{commercial,long,skl}/...
Strides every STRIDE frames, skips the first SKIP_STEPS equilibration steps,
splits 90/10 train/valid, writes train.xyz and valid.xyz (extxyz, MACE format).

Run from: PMF_multiwindows/
Output:   PMF_multiwindows/train.xyz, valid.xyz
"""

from ase.io import read, write
from pathlib import Path
import random
import re

BASE_DIR   = Path(__file__).parent
STRIDE     = 20     # take every 20th frame  (= 20 fs at POTIM=1 fs)
SKIP_STEPS = 200    # skip first 200 MD steps (0.2 ps equilibration)
TRAIN_FRAC = 0.9
SEED       = 42

# ── Discover all OUTCARs ──────────────────────────────────────────────────────
outcar_paths = sorted(BASE_DIR.rglob("*/win_*/OUTCAR"))
if not outcar_paths:
    raise FileNotFoundError(f"No OUTCAR found under {BASE_DIR}")

print(f"Found {len(outcar_paths)} OUTCARs\n")

all_frames = []

for outcar in outcar_paths:
    # Parse label from path:  <node>/Surfaces_<surf>/<crystal>/<pair>/<win>
    parts = outcar.relative_to(BASE_DIR).parts   # e.g. ('commercial','Surfaces_Li2S',...)
    node     = parts[0]                            # commercial / long / skl
    surf     = parts[1]                            # Surfaces_Li2S_fixed ...
    crystal  = parts[2] if len(parts) > 4 else "?"
    pair     = parts[3] if len(parts) > 5 else "?"
    win      = parts[-2]                           # win_2.46
    label    = f"{node}/{surf}/{win}"

    print(f"  Reading {label} ...", flush=True)
    try:
        frames = read(str(outcar), index=":", format="vasp-out")
    except Exception as e:
        print(f"    [warn] read error, retrying without last frame: {e}")
        try:
            frames = read(str(outcar), index=":-1", format="vasp-out")
        except Exception as e2:
            print(f"    [skip] {outcar}: {e2}")
            continue

    # Skip equilibration + stride
    skip = SKIP_STEPS
    sampled = frames[skip::STRIDE]

    for atoms in sampled:
        atoms.info["source_node"]    = node
        atoms.info["source_surface"] = surf
        atoms.info["source_crystal"] = crystal
        atoms.info["source_pair"]    = pair
        atoms.info["source_window"]  = win

    all_frames.extend(sampled)
    print(f"    {len(frames)} frames -> skipped {skip} -> sampled {len(sampled)}")

print(f"\nTotal frames: {len(all_frames)}")

# ── Shuffle + split ───────────────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(all_frames)
n_train   = int(TRAIN_FRAC * len(all_frames))
train_set = all_frames[:n_train]
valid_set = all_frames[n_train:]

# ── Rename keys for MACE (REF_energy / REF_forces) ───────────────────────────
def rename_keys(frames):
    for atoms in frames:
        if "energy" in atoms.info:
            atoms.info["REF_energy"] = atoms.info.pop("energy")
        if "forces" in atoms.arrays:
            atoms.arrays["REF_forces"] = atoms.arrays.pop("forces")
    return frames

train_path = BASE_DIR / "train.xyz"
valid_path = BASE_DIR / "valid.xyz"

write(str(train_path), rename_keys(train_set), format="extxyz")
write(str(valid_path), rename_keys(valid_set), format="extxyz")

print(f"\nSaved:")
print(f"  {train_path}: {len(train_set)} frames")
print(f"  {valid_path}: {len(valid_set)} frames")

# ── Post-fix key names (ASE write() reverts 'forces' column name) ─────────────
def fix_xyz_keys(path: Path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'\bforces:R:3\b', 'REF_forces:R:3', text)
    text = re.sub(r'(?<!free_)(?<!REF_)\benergy=', 'REF_energy=', text)
    path.write_text(text, encoding="utf-8")
    print(f"  Key-fix applied: {path.name}")

for p in [train_path, valid_path]:
    fix_xyz_keys(p)

print("\nDone. Next: run finetune_mace.sh")
