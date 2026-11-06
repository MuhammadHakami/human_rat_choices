#!/usr/bin/env bash
# Build the WACV 2027 Vision4Neuro camera-ready paper and supplement out of tree (no aux files land in git).
#   bash scripts/build_v4n_camera.sh   -> paper/wacv2027/vision4neuro_camera{,_supp}.pdf
set -euo pipefail
cd "$(dirname "$0")/.."
B=$(mktemp -d); trap 'rm -rf "$B"' EXIT; cp -r paper/wacv2027/* "$B"/
for d in vision4neuro_camera vision4neuro_camera_supp; do
  ( cd "$B" && pdflatex -interaction=nonstopmode $d >/dev/null && bibtex $d >/dev/null \
    && pdflatex -interaction=nonstopmode $d >/dev/null && pdflatex -interaction=nonstopmode $d | grep "Output written" )
  ! grep -q "undefined" "$B/$d.log" || { echo "undefined references in $d"; exit 1; }
  cp "$B/$d.pdf" paper/wacv2027/
done
