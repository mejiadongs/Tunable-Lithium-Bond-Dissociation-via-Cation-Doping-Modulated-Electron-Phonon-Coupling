#!/usr/bin/env bash
# Optional slab MD and surface-projected VDOS workflow.
# Run after validating the bulk workflow. Slab structures are expected in $BUNDLE.
set -e
source "$(dirname "$0")/config.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "$MODEL" ] && { echo "MODEL is not set. Run 10_finetune.sh first."; exit 1; }

# Use a surface-trained model for slab calculations.
SLAB_MODEL="${SLAB_MODEL:?Set SLAB_MODEL in config.sh}"
SLAB_DT=0.5
SLAB_EQUIL=10000    # 5 ps
SLAB_PROD=30000     # 15 ps
SLAB_FRICTION=0.1   # ASE units (~100 fs damping)

[ -f "$SLAB_MODEL" ] || { echo "Surface model not found: $SLAB_MODEL"; exit 1; }

# Add the required slab identifiers to this list.
for slab in slab_HfO2 ; do
  if [ -f "$MD_DIR/$slab/summary_surface.json" ]; then
    echo "Skipping completed calculation: $slab"; continue
  fi
  echo "Running slab MD: $slab"
  "$PY" "$BUNDLE/run_md.py" \
    "$BUNDLE/$slab.vasp" "$SLAB_MODEL" "$MD_DIR/$slab" \
    --equil "$SLAB_EQUIL" --prod "$SLAB_PROD" --dt "$SLAB_DT" --T "$MD_T" \
    --friction "$SLAB_FRICTION" --tmax_factor 3.0 --dtype float32
  "$PY" "$BUNDLE/vdos_surface.py" "$MD_DIR/$slab"
done

echo "Generating surface VDOS plot."
"$PY" "$BUNDLE/plot_slab_vdos.py"
echo "Surface VDOS figure: $MD_DIR/fig_slab_vdos.png"
