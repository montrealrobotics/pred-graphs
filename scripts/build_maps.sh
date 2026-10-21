#!/bin/bash
# Build a Predictive Graph for each layout, one after the other.
# Usage (from the repository root): scripts/build_maps.sh 153 [layout ...] [-- extra orchestrate.py options]
set -euo pipefail

layouts=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    layouts+=("$1")
    shift
done
[[ $# -gt 0 ]] && shift

for layout in "${layouts[@]}"; do
    uv run python orchestrate.py layout_num="$layout" keyframes=true \
        output_dir="data/outputs/scene_$layout" "$@"
done
