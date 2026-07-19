#!/usr/bin/env bash
# Run MACE-MD and calculate VDOS for a concentration series.
set -e
source "$(dirname "$0")/config.sh"
[ -z "$MODEL" ] && { echo "MODEL is not set. Run 10_finetune.sh first."; exit 1; }
SERIES="$MD_DIR/series"; mkdir -p "$SERIES"

# Edit this list to select structures for the series.
CELLS=(
  "L_111_1Sc"
  "L_211_1Sc"
  "L_221_1Sc"
  "L_222_1Sc"
  "L_223_1Sc"
  "L_221_pristine"
  "L_111_1Y"
  "L_211_1Y"
  "L_221_1Y"
  "L_222_1Y"
  "L_223_1Y"
)

for cell in "${CELLS[@]}"; do
  echo "=== MD: $cell ==="
  "$PY" "$BUNDLE/run_md.py" \
    "$RELAX_DIR/$cell/CONTCAR" "$MODEL" "$SERIES/$cell" \
    --equil "$MD_EQUIL" --prod "$MD_PROD" --dt "$MD_DT" --T "$MD_T" \
    --min_len 10.0
  "$PY" "$BUNDLE/vdos_from_md.py" "$SERIES/$cell"
done
echo "Concentration-series calculations completed."
