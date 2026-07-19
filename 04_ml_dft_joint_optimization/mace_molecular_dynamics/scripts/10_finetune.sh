#!/usr/bin/env bash
# Step 1: extract dataset from AIMD + fine-tune MACE-OMAT-0 on the GPU node.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../config.sh"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM

if [ -f "$DATA/train.xyz" ]; then
  echo "Using the existing training dataset."
else
  echo "=== extracting training set from AIMD ==="
  "$PY" "$SCRIPT_DIR/extract_dataset.py"
fi

echo "=== fine-tuning MACE-OMAT-0 (foundation=$FOUNDATION) ==="
"$PY" -m mace.cli.run_train \
  --name="$MACE_NAME" \
  --foundation_model="$FOUNDATION" \
  --multiheads_finetuning=False \
  --train_file="$DATA/train.xyz" \
  --valid_file="$DATA/valid.xyz" \
  --test_file="$DATA/test.xyz" \
  --energy_key=REF_energy --forces_key=REF_forces \
  --E0s=foundation \
  --loss=weighted --energy_weight=1.0 --forces_weight=100.0 \
  --lr=0.005 --ema --ema_decay=0.99 \
  --max_num_epochs=200 --patience=50 \
  --batch_size=8 --valid_batch_size=8 \
  --default_dtype=float32 --device=cuda --seed=1 \
  --error_table=PerAtomRMSE \
  --work_dir="$TRAIN_DIR" \
  --restart_latest --save_cpu

# resolve the produced model file (prefer the compiled one) and record it
M="$(ls -1 "$TRAIN_DIR"/${MACE_NAME}*compiled*.model 2>/dev/null | head -1)"
[ -z "$M" ] && M="$(ls -1 "$TRAIN_DIR"/${MACE_NAME}*.model 2>/dev/null | grep -v compiled | head -1)"
echo "$M" > "$WORK/model_path.txt"
echo
echo ">>> fine-tuned model: $M"
echo "Model path recorded in $WORK/model_path.txt."
