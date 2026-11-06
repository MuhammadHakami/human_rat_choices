"""Registered shared spatial scaffold for seven source-only learning rules.

Reuse the reviewed learner, interaction, accounting and independent replay
implementations. Original files/bindings stay unchanged. New observations append
a frozen source-calibrated 100-cell belief to the original 150 camera inputs.
The old raw-camera results have different seeds/capacity and are diagnostic
comparisons, not a controlled estimate of the scaffold's causal effect.
"""
import argparse
import concurrent.futures
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wacv_embodied_source_runner as base
import audit_wacv_embodied_source as independent
from wacv_embodied_source_cache import SourceCache
from wacv_embodied_source_env import SourceBatch as PhysicalBatch
from wacv_embodied_belief_source import LandmarkBelief

PROTOCOL = ROOT / "docs/WACV_EMBODIED_SPATIAL_SOURCE_PROTOCOL_20261012.json"
EXPECTED = "de6f74294e5560de4c25badd2c77bc4486c87a9a8a80bca1126372618a35f07e"
OUT = ROOT / "data/audit/wacv_embodied_source_20261012/spatial_neural"
GO = OUT / "GO.json"
DRY = False
ENVS = []
CODE = list(base.CODE) + ["scripts/wacv_embodied_spatial_source.py",
                         "scripts/wacv_embodied_belief_source.py",
                         "scripts/audit_wacv_embodied_source.py"]


class SpatialCache:
    def __init__(self, raw, protocol):
        self.raw = raw
        self.protocol = protocol
        self.obs_dim = 250
        self.goal = raw.goal
        self.index = raw.index
        self.rotations = raw.rotations

    def geometry(self, cfg):
        return self.raw.geometry(cfg)


def load():
    assert base.sha(PROTOCOL) == EXPECTED
    P = json.loads(PROTOCOL.read_text())
    assert base.sha(ROOT / P["parent_protocol"]) == P["parent_sha256"]
    assert base.sha(ROOT / P["representation_code"]) == P["representation_code_sha256"]
    assert base.sha(ROOT / P["representation_protocol"]) == P["representation_protocol_sha256"]
    raw = SourceCache(base.MANIFEST, ROOT / P["parent_protocol"])
    if DRY:
        # Technical tests use seed0, as required by the reused independent
        # auditor; full scientific fits use registered fresh seeds10/11/12.
        P["source"]["seeds"] = [0]
    return SpatialCache(raw, P), P


def hashes():
    return dict(protocol=base.sha(PROTOCOL), cache_manifest=base.sha(base.MANIFEST),
                cache_array=base.sha(base.MANIFEST.with_suffix(".npz")),
                code={c: base.sha(ROOT / c) for c in CODE})


class SpatialBatch:
    def __init__(self, cache, n=1, seed=0, frames=None, schedule=None):
        physical = copy.copy(cache.raw)
        physical.protocol = cache.protocol
        self.env = PhysicalBatch(physical, n, seed, frames, schedule)
        atlas = cache.raw.features[0, :, :, :, 0, :64].reshape(100, 4, 64)
        self.filters = [LandmarkBelief(atlas, .03) for _ in range(n)]
        self.obs_dim = 250
        ENVS.append(self)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def augment(self, obs, actions=None, done=None):
        belief = []
        for i, filt in enumerate(self.filters):
            if actions is None or done[i]:
                filt.reset()
                if self.env.exhausted:
                    belief.append(np.zeros(100, np.float32))
                    continue
                filt.observe(obs[i, :64])
            else:
                filt.observe(obs[i, :64], int(actions[i]), bool(obs[i, 144]))
            belief.append(filt.p.ravel().astype(np.float32))
        return np.concatenate((obs, np.stack(belief)), axis=1).astype(np.float32)

    def reset(self):
        return self.augment(self.env.reset())

    def step(self, actions):
        obs, rewards, done = self.env.step(actions)
        return self.augment(obs, actions, done), rewards, done

    def representation_resources(self):
        return dict(fixed_atlas_bytes_each=102400,
                    fixed_atlas_bytes_allocated=sum(f.atlas.nbytes for f in self.filters),
                    posterior_bytes=sum(f.p.nbytes for f in self.filters),
                    likelihood_memoization_bytes=sum(v.nbytes for f in self.filters for v in f.likelihoods.values()),
                    environment_slots=len(self.filters),
                    supervision_views=400,
                    storage_note="Current implementation copies fixed atlas per slot; source400views are unique, not400per slot.")


def configure():
    base.load = independent.load = load
    base.hashes = independent.hashes = hashes
    base.GO = independent.GO = GO
    base.SourceBatch = SpatialBatch


def fit(method, rate, seed, out, dry):
    ENVS.clear()
    result = base.fit(method, rate, seed, out, dry)
    result["frozen_representation"] = [e.representation_resources() for e in ENVS]
    result["source_calibration"] = "400privileged openarena0camera/pose/heading pairs, sharedfrozen sourceatlas; runtime250obs includespixels andcausalposebelief."
    result["comparison_limit"] = "Earlier raw-camera seeds/capacity differ; no isolated causal representation-effect claim."
    d = Path(out) / method / f"rate{rate}" / f"seed{seed}"
    (d / "result.json").write_text(json.dumps(result, indent=1) + "\n")


def select_exact(out, P):
    """Equal 180-trial denominators permit integer count ties, without float artifacts."""
    audit=json.loads((Path(out)/"root_audit.json").read_text())
    assert audit["ok"] and not audit["dry"] and audit["cases"]==63
    choices={}
    for method in P["methods"]:
        candidates=[]
        for ri, rate in enumerate(method["rates"]):
            counts={"adaptive":0,"frozen":0}; failed=0
            for seed in P["source"]["seeds"]:
                r=json.loads((Path(out)/method["name"]/f"rate{ri}"/f"seed{seed}"/"result.json").read_text())
                if r["status"]=="failed":failed+=1;continue
                for arm in counts:
                    value=sum(r["selection"][f"{arm}_{repeat}"]["success"]*20 for repeat in range(3))
                    assert abs(value-round(value))<1e-10
                    counts[arm]+=round(value)
            candidates.append(dict(rate_index=ri,rate=rate,adaptive_count=counts["adaptive"],
                                   frozen_count=counts["frozen"],denominator=180,failed=failed))
        best=min(candidates,key=lambda c:(-c["adaptive_count"],-c["frozen_count"],c["rate_index"]))
        choices[method["name"]]=dict(**best,adaptive=best["adaptive_count"]/180,
                                    frozen=best["frozen_count"]/180,candidates=candidates,
                                    numerical_source_recipe_exists=best["failed"]<3)
    (Path(out)/"selection.json").write_text(json.dumps(dict(rule=P["source"]["select_protocol"],
                     exact_integer_ties=True,failed_accuracy_is_null_not_measured_zero=True,
                     choice=choices),indent=2)+"\n")


def cohort(out):
    g = json.loads(GO.read_text())
    assert g["status"] == "GO" and g["hashes"] == hashes()
    out = Path(out)
    assert out.resolve() == Path(g["output"]).resolve()
    assert not list(out.glob("*/rate*/seed*/result.json")), "Preserve existing fits; no implicit restart"
    _, P = load()
    jobs = [(m["name"], ri, s) for m in P["methods"] for ri in range(3) for s in P["source"]["seeds"]]
    assert len(jobs) == 63
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
    status = dict(state="running", pid=os.getpid(), planned=63, completed=0,
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    start = time.time()

    def worker(job):
        method, ri, seed = job
        path = out / "logs" / f"{method}_{ri}_{seed}"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".out").open("w") as so, path.with_suffix(".err").open("w") as se:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "fit", method,
                                str(ri), str(seed), "--out", str(out)], cwd=ROOT, env=env, stdout=so, stderr=se)
        assert r.returncode == 0, f"Worker failed: {job}, inspect {path}"
        result = json.loads((out/method/f"rate{ri}"/f"seed{seed}"/"result.json").read_text())
        assert result["status"] in ("complete", "failed")
        return job, result["status"]

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for f in concurrent.futures.as_completed([pool.submit(worker, j) for j in jobs]):
                job, state = f.result()
                status["completed"] += 1
                status["elapsed_s"] = round(time.time()-start, 1)
                (out/"status.json").write_text(json.dumps(status, indent=2)+"\n")
                print(json.dumps(dict(completed=status["completed"], planned=63, case=job, status=state)), flush=True)
        independent.audit(out)
        assert base.audit(out)["ok"]
        select_exact(out, P)
        status["state"] = "complete"
    except Exception as e:
        status.update(state="stopped", error=repr(e))
        raise
    finally:
        status["elapsed_s"] = round(time.time()-start, 1)
        (out/"status.json").write_text(json.dumps(status, indent=2)+"\n")


def main():
    global DRY
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["dry", "fit", "dry_audit", "cohort"])
    p.add_argument("method", nargs="?"); p.add_argument("rate", nargs="?", type=int)
    p.add_argument("seed", nargs="?", type=int); p.add_argument("--out", required=True)
    a = p.parse_args(); DRY = a.cmd in ("dry", "dry_audit")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    configure()
    if a.cmd == "dry": fit(a.method, 0, 0, a.out, True)
    elif a.cmd == "fit": fit(a.method, a.rate, a.seed, a.out, False)
    elif a.cmd == "dry_audit": independent.audit(a.out, dry=True)
    else: cohort(a.out)


if __name__ == "__main__":
    main()
