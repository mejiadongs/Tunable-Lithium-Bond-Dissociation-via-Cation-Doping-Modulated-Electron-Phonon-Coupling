#!/usr/bin/env bash
# Optional step: slab MD -> surface-projected VDOS (bulk -> interface bridge).
# Run AFTER 20_validate.sh (needs the fine-tuned MODEL). Uses NVT production for
# surface stability. Slabs: slab_HfO2.vasp / slab_ScHfO2.vasp (in this bundle).
set -e
source "$(dirname "$0")/config.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "$MODEL" ] && { echo "MODEL not set — run 10_finetune.sh first"; exit 1; }

# slab-stable MD: small step, per-step temperature rescue, and CRUCIALLY the
# surface-trained PMF potential (bulk-only potential is OOD at the surface).
SLAB_MODEL="$BUNDLE/mace_pmf_compiled.model"   # scp'd from the PMF project (float64)
SLAB_DT=0.5
SLAB_EQUIL=10000    # 5 ps
SLAB_PROD=30000     # 15 ps
SLAB_FRICTION=0.1   # ASE units (~100 fs damping)

[ -f "$SLAB_MODEL" ] || { echo "PMF model missing: $SLAB_MODEL — scp it first"; exit 1; }

# Start with pristine ONLY as a stability probe. If its production shows 0 (or a
# few) rescues, uncomment the Sc/Y slabs and re-run (finished slabs are skipped).
for slab in slab_HfO2 ; do   # add: slab_ScHfO2 slab_YHfO2
  if [ -f "$MD_DIR/$slab/summary_surface.json" ]; then
    echo "=== $slab already done — skipping ==="; continue
  fi
  echo "=== slab MD: $slab (PMF surface potential) ==="
  "$PY" "$BUNDLE/run_md.py" \
    "$BUNDLE/$slab.vasp" "$SLAB_MODEL" "$MD_DIR/$slab" \
    --equil "$SLAB_EQUIL" --prod "$SLAB_PROD" --dt "$SLAB_DT" --T "$MD_T" \
    --friction "$SLAB_FRICTION" --tmax_factor 3.0 --dtype float32
  "$PY" "$BUNDLE/vdos_surface.py" "$MD_DIR/$slab"
done

echo "=== plot surface vs interior + surface Sc ==="
"$PY" "$BUNDLE/plot_slab_vdos.py"
echo ">>> figure: $MD_DIR/fig_slab_vdos.png"
echo ">>> CHECK: 'slab interior' centroid should match the BULK MD-VDOS centroid"
echo ">>>        (free consistency check, no slab DFT needed)."
echo ">>> STORY: surface / surface-Sc centroid red-shifted vs interior = bulk->surface bridge."
