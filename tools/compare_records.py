"""Compare re-computed per-seed records with the shipped ones (outcomes only; timing, GPU name and paths ignored).
  python tools/compare_records.py SHIPPED_DIR NEW_DIR
"""
import json, sys
from pathlib import Path

SKIP = {"wall_s", "gpu_name", "written_utc", "source", "protocol_sha256", "manifest_sha256"}


def strip(x):
    if isinstance(x, dict): return {k: strip(v) for k, v in x.items() if k not in SKIP and not k.endswith("_s")}
    return [strip(v) for v in x] if isinstance(x, list) else x


def main(a, b):
    a, b = Path(a), Path(b); same = diff = missing = 0
    for f in sorted(a.rglob("*.json")):
        g = b / f.relative_to(a)
        if not g.exists(): missing += 1; continue
        if strip(json.loads(f.read_text())) == strip(json.loads(g.read_text())): same += 1
        else: diff += 1; print("differs:", f.relative_to(a))
    print(f"identical {same}  different {diff}  not re-run {missing}"); return diff == 0


if __name__ == "__main__":
    assert main(*sys.argv[1:3]), "some records differ"
