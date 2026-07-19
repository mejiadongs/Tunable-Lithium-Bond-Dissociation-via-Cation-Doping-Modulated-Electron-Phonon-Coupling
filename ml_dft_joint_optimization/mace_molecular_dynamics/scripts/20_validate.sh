#!/usr/bin/env bash
# Step 2: validate the fine-tuned potential — MACE-MD VDOS vs DFT-AIMD VDOS at 25%.
# This is the key free consistency check: same 96-atom cells, same observable (VDOS).
set -e
source "$(dirname "$0")/config.sh"
[ -z "$MODEL" ] && { echo "MODEL not set — run 10_finetune.sh first"; exit 1; }

echo "=== DFT-AIMD reference VDOS (25% cells) ==="
"$PY" "$BUNDLE/vdos_dft_aimd.py"

echo "=== MACE-MD on the same 96-atom 25% cells ==="
for pair in "HfO2:valid_HfO2" "Sc-HfO2:valid_Sc" "Y-HfO2:valid_Y"; do
  sys="${pair%%:*}"; sub="${pair##*:}"
  echo "--- $sys ---"
  "$PY" "$BUNDLE/run_md.py" \
    "$AIMD_DIR/$sys/ToBeDelete_POSCAR" "$MODEL" "$MD_DIR/$sub" \
    --equil "$MD_EQUIL" --prod "$MD_PROD" --dt "$MD_DT" --T "$MD_T"
  "$PY" "$BUNDLE/vdos_from_md.py" "$MD_DIR/$sub"
done

echo "=== overlay + centroid agreement ==="
"$PY" "$BUNDLE/compare_validation.py"
echo ">>> Inspect $WORK/fig_validation_25pct.png — MACE-MD (red) should track DFT-AIMD (black),"
echo ">>> especially the low-frequency (<10 THz) shape. Centroid diff within ~0.3 THz = good."
