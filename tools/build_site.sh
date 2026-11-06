#!/usr/bin/env bash
# Assemble the GitHub Pages site in _site/ (used by .github/workflows/pages.yml, and for a local preview:
#   bash tools/build_site.sh && python -m http.server -d _site 8000).
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf _site && cp -r site _site && rm -rf _site/test
mkdir -p _site/paper && cp paper/wacv2027/vision4neuro_camera.pdf paper/wacv2027/vision4neuro_camera_supp.pdf _site/paper/
touch _site/.nojekyll
echo "site in _site/ ($(du -sh _site | cut -f1))"
