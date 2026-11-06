"""Brain-signal arm, linked to the story (camera-ready, author 2026-10-31: 'brain signals adapted into our work, not
in isolation'). Extends the registered MICrONS-perception refit (v4n_microns_perception_claude_20261017, unchanged and
imported) from 10 to 30 seeds with 20 NEW odd seeds 1311..1349 on GPU 1 (the driver pins odd seeds to the RTX 3090;
all arms of a seed share it). Question in the story: does the human-choice content (lower revisiting, higher transfer)
survive when the agent sees through predicted mouse visual cortex (twin) rather than pixels (pix), versus the
partially trained random-core control (rcore)?
  prereg | fit W N | ood W N
"""
import json, os, sys, time, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_microns_perception_claude_20261017 as MP

NEW = tuple(range(1311, 1350, 2)); MP.SEEDS = MP.SEEDS + NEW
ORDER = ("twin", "pix", "rcore"); ARMS = ("human_bc", "human_legal_uniform", "cold_ppo")
REG = MP.OUT / "EXTENSION_20261031.json"


def jobs(): return [(a, p, s) for p in ORDER for s in NEW for a in ARMS]


def prereg():
    assert not REG.exists()
    REG.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seeds_new=NEW, gpu=1, perceptions=ORDER,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), driver_sha256=MP.sha(MP.HERE),
        outcomes_seen="10-seed results known (published in the reviewed version); none of the new seeds",
        primary=dict(T1="twin: human_bc - human_legal_uniform success (16 exposed rooms), paired over all 30 seeds",
                     T2="twin: same contrast in revisit rate", T3="pix: same two contrasts", multiplicity="Holm over T1-T3 (4 tests)"),
        secondary="rcore contrasts; perception x choice interaction; 20-new-seed-only estimates (replication)"), indent=1) + "\n")
    print(REG.read_text())


def run(phase, w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and REG.exists()
    for a, p, s in jobs()[w::n]:
        import importlib; importlib.reload(MP.S); MP.SEEDS = MP.SEEDS if 1311 in MP.SEEDS else MP.SEEDS + NEW
        if phase == "fit":
            MP.setup(p); MP.S.SEEDS = MP.SEEDS; MP.S.fit(a, "no_posterior", s, False)
        else:
            MP.S.SEEDS = MP.SEEDS; MP.ood(a, p, s)
        print(json.dumps(dict(phase=phase, arm=a, perc=p, seed=s, done=True)), flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    prereg() if c == "prereg" else run(c, int(sys.argv[2]), int(sys.argv[3]))
