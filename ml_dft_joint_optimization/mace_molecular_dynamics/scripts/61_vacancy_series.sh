#!/usr/bin/env bash
# Run a vacancy-containing concentration series with a surface-trained potential.
# Validate vacancy-dependent trends with an independent reference calculation.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../config.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VMODEL="${VACANCY_MODEL:?Set VACANCY_MODEL in config.sh}"
[ -f "$VMODEL" ] || { echo "Vacancy model not found: $VMODEL"; exit 1; }

VS="$MD_DIR/vac_series"; mkdir -p "$VS"
DT=0.5
EQUIL=10000     # 5 ps
PROD=30000      # 15 ps
FRICTION=0.1

# Edit this list to select structures for the series.
CELLS=(
  vs_pristine_96 vs_pristine_288
  vs_Sc_25.000 vs_Sc_12.500 vs_Sc_6.250 vs_Sc_3.125 vs_Sc_2.083
  vs_Y_25.000  vs_Y_12.500  vs_Y_6.250  vs_Y_3.125  vs_Y_2.083
)

for cell in "${CELLS[@]}"; do
  if [ -f "$VS/$cell/summary.json" ]; then
    echo "Skipping completed calculation: $cell"; continue
  fi
  echo "Running vacancy-series MD: $cell"
  "$PY" "$SCRIPT_DIR/run_md.py" \
    "$VACANCY_STRUCTURE_DIR/$cell.vasp" "$VMODEL" "$VS/$cell" \
    --equil "$EQUIL" --prod "$PROD" --dt "$DT" --T "$MD_T" \
    --friction "$FRICTION" --tmax_factor 3.0 --dtype float32
  "$PY" "$SCRIPT_DIR/vdos_from_md.py" "$VS/$cell"
done

echo "Generating vacancy-series summary."
"$PY" "$SCRIPT_DIR/summarize_vac_series.py"
