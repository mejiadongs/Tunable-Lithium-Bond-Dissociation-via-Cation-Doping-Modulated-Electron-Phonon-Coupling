#!/usr/bin/env bash
# Copy this file to config.sh and replace all placeholder paths.

export PY="/path/to/mace-environment/bin/python"
export BUNDLE="/path/to/mace_molecular_dynamics"
export AIMD_DIR="/path/to/aimd-calculations"
export RELAX_DIR="/path/to/relaxed-structures"

export WORK="$BUNDLE/work"
export DATA="$WORK/data"
export TRAIN_DIR="$WORK/train"
export MD_DIR="$WORK/md"
export MACE_NAME="mace_finetuned"
[ -f "$WORK/model_path.txt" ] && export MODEL="$(cat "$WORK/model_path.txt")"

export FOUNDATION="/path/to/mace-foundation.model"
export SLAB_MODEL="/path/to/mace-surface.model"
export VACANCY_MODEL="/path/to/mace-vacancy.model"
export VACANCY_STRUCTURE_DIR="/path/to/vacancy-structures"
export MD_EQUIL=5000
export MD_PROD=15000
export MD_DT=1.0
export MD_T=300

mkdir -p "$WORK" "$DATA" "$TRAIN_DIR" "$MD_DIR"
