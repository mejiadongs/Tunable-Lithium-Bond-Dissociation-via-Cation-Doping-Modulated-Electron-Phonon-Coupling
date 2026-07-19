#!/usr/bin/env bash
# Step 4: overlay concentration-dependent VDOS + plot the 0-10 THz red-shift descriptor.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../config.sh"
"$PY" "$SCRIPT_DIR/plot_concentration_vdos.py" "$MD_DIR/series"
echo ">>> figure: $MD_DIR/series/fig_concentration_vdos.png"
echo ">>> This is the reviewer figure: local VDOS overlay + 0-10 THz centroid/peak vs concentration."
