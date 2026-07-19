#!/usr/bin/env bash
# Step 3: MACE-MD on the DFT-relaxed concentration series -> VDOS per concentration.
# Only run this AFTER 20_validate.sh looks good.
set -e
source "$(dirname "$0")/config.sh"
[ -z "$MODEL" ] && { echo "MODEL not set — run 10_finetune.sh first"; exit 1; }
SERIES="$MD_DIR/series"; mkdir -p "$SERIES"

# Sc concentration series + one pristine reference.
# Edit this list to add/remove concentrations or the Y series.
CELLS=(
  "L_111_1Sc"      # 25%
  "L_211_1Sc"      # 12.5%
  "L_221_1Sc"      # 6.25%
  "L_222_1Sc"      # 3.125%
  "L_223_1Sc"      # 2.083%  (= experimental 2 mol%)
  "L_221_pristine" # 0%
  # --- Y series (parallel comparison) ---
  "L_111_1Y"       # 25%
  "L_211_1Y"       # 12.5%
  "L_221_1Y"       # 6.25%
  "L_222_1Y"       # 3.125%
  "L_223_1Y"       # 2.083%
)

for cell in "${CELLS[@]}"; do
  echo "=== MD: $cell ==="
  "$PY" "$BUNDLE/run_md.py" \
    "$RELAX_DIR/$cell/CONTCAR" "$MODEL" "$SERIES/$cell" \
    --equil "$MD_EQUIL" --prod "$MD_PROD" --dt "$MD_DT" --T "$MD_T" \
    --min_len 10.0
  "$PY" "$BUNDLE/vdos_from_md.py" "$SERIES/$cell"
done
echo ">>> all concentration MD done. Run 40_plot.sh next."
