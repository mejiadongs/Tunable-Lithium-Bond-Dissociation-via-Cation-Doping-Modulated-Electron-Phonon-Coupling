#!/usr/bin/env bash
# Analyze vacancy-containing structures with a surface-trained potential.
# Validate any vacancy-dependent trend with an independent reference calculation.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../config.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VMODEL="${VACANCY_MODEL:?Set VACANCY_MODEL in config.sh}"
[ -f "$VMODEL" ] || { echo "Vacancy model not found: $VMODEL"; exit 1; }

VAC="$MD_DIR/vacancy"; mkdir -p "$VAC"
V_DT=0.5
V_EQUIL=10000     # 5 ps
V_PROD=30000      # 15 ps
V_FRICTION=0.1

for cell in vac_pristine vac_2Sc_novac vac_2Sc_Ovac; do
  if [ -f "$VAC/$cell/summary.json" ]; then
    echo "Skipping completed calculation: $cell"; continue
  fi
  echo "Running vacancy MD: $cell"
  "$PY" "$SCRIPT_DIR/run_md.py" \
    "$VACANCY_STRUCTURE_DIR/$cell.vasp" "$VMODEL" "$VAC/$cell" \
    --equil "$V_EQUIL" --prod "$V_PROD" --dt "$V_DT" --T "$MD_T" \
    --friction "$V_FRICTION" --tmax_factor 3.0 --dtype float32
  "$PY" "$SCRIPT_DIR/vdos_from_md.py" "$VAC/$cell"
done

echo "Generating vacancy comparison."
"$PY" "$SCRIPT_DIR/compare_vacancy.py"
