#!/usr/bin/env bash
# Validate the fine-tuned potential by comparing MACE-MD and DFT-AIMD VDOS.
set -e
source "$(dirname "$0")/config.sh"
[ -z "$MODEL" ] && { echo "MODEL is not set. Run 10_finetune.sh first."; exit 1; }

echo "Computing reference VDOS from DFT-AIMD trajectories."
"$PY" "$BUNDLE/vdos_dft_aimd.py"

echo "Running MACE-MD on the validation structures."
for pair in "HfO2:valid_HfO2" "Sc-HfO2:valid_Sc" "Y-HfO2:valid_Y"; do
  sys="${pair%%:*}"; sub="${pair##*:}"
  echo "--- $sys ---"
  "$PY" "$BUNDLE/run_md.py" \
    "$AIMD_DIR/$sys/ToBeDelete_POSCAR" "$MODEL" "$MD_DIR/$sub" \
    --equil "$MD_EQUIL" --prod "$MD_PROD" --dt "$MD_DT" --T "$MD_T"
  "$PY" "$BUNDLE/vdos_from_md.py" "$MD_DIR/$sub"
done

echo "Generating validation comparison."
"$PY" "$BUNDLE/compare_validation.py"
echo "Inspect $WORK/fig_validation_25pct.png to compare the spectra."
