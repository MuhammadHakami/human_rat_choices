"""Zero-shot OOD block: frozen 30-seed replication policies in all 16 fresh rooms with the SOURCE goal (3,3) - the
agent analogue of an animal meeting new barriers with a known goal (de Cothi). No learning. 30 trials per room
(10 native starts x 3 textures), clean tier. Uses wacv_continual_factorial_claude.probe unchanged except the goal.
  python scripts/wacv_ood_zero_shot_claude.py WORKER NWORKERS     (CPU; GPU1 visible only for unpickling)
"""
import json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wacv_continual_factorial_claude as F

OUT = F.ROOT / "data/audit/wacv_ood_zero_shot_claude"
ARMS = ("rat_bc", "rat_legal_uniform", "rat_sham", "human_bc", "human_legal_uniform", "cold_ppo")
_env = F.environment


def env_source_goal(pair, *a, **k):
    e = _env(pair, *a, **k)
    for cfg in range(100, 116): e.env.cache.goals[cfg] = (3, 3)        # source goal everywhere; walls unchanged
    return e


def run(worker, n):
    F.environment = env_source_goal; go = json.loads((OUT / "GO.json").read_text())
    jobs = [(a, s, p, st) for s in go["seeds"] for a in ARMS for p in range(8) for st in ("A", "B")][worker::n]
    manifest = json.loads(F.study.MANIFEST.read_text()); method = next(m for m in manifest["methods"] if m["id"] == "ppo")
    for a, s, p, st in jobs:
        f = OUT / "cohort" / a / f"seed{s}" / f"pair{p}_{st}.json"
        if f.exists(): continue
        F.study.source_model = lambda x, a=a: F.source_checkpoint(a, x); m = F.study.build(method, s)
        r = F.probe(m, p, st, s, "clean"); r.pop("rng_initial"); r.pop("pose_audit")
        r["episodes"] = [dict(spawn=e["spawn"], rotation_index=e["rotation_index"], success=e["success"], spl=e["spl"], decisions=e["decisions"])
                         for e in r["episodes"] if "spawn" in e]
        f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(dict(arm=a, seed=s, pair=p, room=st, goal=[3, 3], hashes=go["hashes"], **r)) + "\n")
    print(json.dumps(dict(worker=worker, done=len(jobs))), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1); run(int(sys.argv[1]), int(sys.argv[2]))
