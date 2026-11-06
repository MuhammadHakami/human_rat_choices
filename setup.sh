#!/usr/bin/env bash
# One-time setup.   bash setup.sh            venv + raw maze data (downloaded from Zenodo, not redistributed here)
#                   bash setup.sh --models   + all 30 seeds x 6 priors (release asset; seed 1000 ships in git)
#                   bash setup.sh --records  + large per-episode records (release asset)
#                   bash setup.sh --rosetta  + upstream unpaired-rosetta code (commit fdfad84, unmodified)
set -euo pipefail
cd "$(dirname "$0")"
RELEASE_URL=${RELEASE_URL:-https://github.com/MuhammadHakami/human_rat_choices/releases/download/v1.0}
PY=${PY:-python3}
[ -x .venv/bin/python ] || "$PY" -m venv .venv
[ -n "${SKIP_PIP:-}" ] || .venv/bin/python -m pip install -q -r requirements.txt

Z=data/real_sources/decothi2022; ZIP=$Z/Predictive-maps-in-rats-and-humans-v1.zip; mkdir -p "$Z"
[ -f "$ZIP" ] || curl -fL -o "$ZIP" "https://zenodo.org/api/records/6734981/files/willdecothi/Predictive-maps-in-rats-and-humans-v1.zip/content"
echo "114cac09a20457d301d568ee91e06a20  $ZIP" | md5sum -c --quiet
unzip -qo "$ZIP" -d "$Z"; cp "$Z"/willdecothi-*/"Standard Functions (add to path)/mazes.mat" "$Z/"
.venv/bin/python scripts/import_tartarus.py

fetch() {   # release asset -> checksum -> extract in place
  curl -fL -o "data/$1" "$RELEASE_URL/$1"; curl -fsL "$RELEASE_URL/SHA256SUMS" | grep " $1\$" | (cd data && sha256sum -c --quiet)
  tar -xf "data/$1"; rm "data/$1"; }
for a in "$@"; do case $a in
  --models)  fetch checkpoints_seeds30.tar ;;
  --records) fetch records_full.tar ;;
  --rosetta) R=data/audit/sota_reproduction_20261017/sources/unpaired-rosetta
             [ -d "$R" ] || git clone -q https://github.com/dominik-schnaus/unpaired-rosetta.git "$R"; git -C "$R" checkout -q fdfad84 ;;
  *) echo "unknown option $a"; exit 1 ;;
esac; done
echo "setup done"
