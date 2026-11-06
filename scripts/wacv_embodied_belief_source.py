"""Causal source-calibrated landmark/odometry diagnostic, not species transfer.

The 400 labeled open-arena views and renderer/bounds priors are privileged
source calibration. Visual control uses pixels, own actions and contact only.
True poses/configurations below are restricted to the privileged reference and
audit logs. Original 63-case source code and cache remain unchanged.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wacv_embodied_source_cache import SourceCache
from wacv_embodied_source_env import SourceBatch, selection_schedule
from arc_fly.unimaze import NB

PROTOCOL = ROOT / "docs/WACV_EMBODIED_BELIEF_SOURCE_PROTOCOL_20261012_v2.json"
SHA = "312b270cb2b090d4b6d2001466815793cb20724540230920986f78f16b4dad8d"
OUT = ROOT / "data/audit/wacv_embodied_source_20261012/belief_source"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_headings():
    headings = []
    for r in range(10):
        for c in range(10):
            d = np.array([4.5 - r, 4.5 - c]); k = int(np.argmax(np.abs(d)))
            headings.append(([0, 1] if k == 0 else [2, 3])[int(d[k] < 0)])
    return np.array(headings)


class LandmarkBelief:
    def __init__(self, atlas, sigma):
        self.atlas = atlas.copy()  # [100,4,64] labeled SOURCE views only
        self.sigma = sigma
        self.likelihoods = {}
        self.reset()

    def reset(self):
        self.p = np.ones((10, 10), np.float64) / 100

    def observe(self, cue, action=None, contact=False):
        if action is not None and not contact:
            dr, dc = NB[action]
            self.p = np.roll(self.p, (dr, dc), axis=(0, 1))
            if dr == 1: self.p[0] = 0
            if dr == -1: self.p[-1] = 0
            if dc == 1: self.p[:, 0] = 0
            if dc == -1: self.p[:, -1] = 0
        key = action, np.asarray(cue, np.float32).tobytes()
        if key not in self.likelihoods:
            templates = self.atlas if action is None else self.atlas[:, action]
            ll = -np.square(templates.astype(np.float64) - cue).mean(-1) / (2 * self.sigma ** 2)
            likelihood = np.exp(ll - ll.max())
            if action is None:
                likelihood = likelihood.mean(1)  # uniform initial heading, never simulator heading
            self.likelihoods[key] = likelihood.reshape(10, 10)
        self.p *= self.likelihoods[key]
        total = self.p.sum()
        if not total > 0: raise FloatingPointError("Inconsistent sensory/odometry posterior")
        self.p /= total
        return int(self.p.argmax())


def independent_belief(p, atlas, cue, action, contact, sigma):
    """Separate hypothesis enumeration instead of roll/boundary-mask implementation."""
    prior = np.zeros(100)
    for s, v in enumerate(p):
        r, c = divmod(s, 10)
        if action is not None and not contact:
            dr, dc = NB[action]; r, c = r + dr, c + dc
        if 0 <= r < 10 and 0 <= c < 10:
            prior[r * 10 + c] += v
    ll = []
    for s in range(100):
        heads = range(4) if action is None else [action]
        ll.append([-sum(float(a - b) ** 2 for a, b in zip(atlas[s, h].astype(float), cue)) /
                   (64 * 2 * sigma ** 2) for h in heads])
    ll = np.array(ll)
    q = prior * np.exp(ll - ll.max()).mean(1)
    return q / q.sum()


def run(cache, P, method, seed, Q, learn, path, schedule=None, dry=False):
    limit = 128 if dry and schedule is None else P["training_frames"]
    if schedule is not None: limit = 900
    env = SourceBatch(cache, n=1, seed=seed, frames=limit, schedule=schedule)
    obs = env.reset()
    atlas = cache.features[0, :, :, :, 0, :64].reshape(100, 4, 64)
    filt = LandmarkBelief(atlas, P["belief"]["sigma"])
    visual = method == "visual_belief_Q"
    s = filt.observe(obs[0, :64]) if visual else int(np.ravel_multi_index(env.worlds[0]["position"], (10, 10)))
    rng = np.random.default_rng(seed)
    records = []
    while not env.exhausted and env.frames < env.limit:
        w = env.worlds[0]
        # These labels enter logs/reference only, never the visual controller.
        cfg, pos, heading, ti = w["configuration"], w["position"], tuple(w["heading"]), w["rotation_index"]
        a = int(rng.integers(4)) if rng.random() < P["Q"]["epsilon"] else int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
        q_before = Q[s].copy()
        uncertainty = float(-(filt.p[filt.p > 0] * np.log(filt.p[filt.p > 0])).sum()) if visual else 0.0
        true_probability = float(filt.p[pos]) if visual else 1.0
        obs, r, done = env.step(np.array([a]))
        contact = bool(w["contact"])
        ns = -1
        if not done[0]:
            ns = filt.observe(obs[0, :64], a, contact) if visual else int(np.ravel_multi_index(w["position"], (10, 10)))
        if learn:
            target = float(r[0]) + (0.0 if done[0] else P["Q"]["gamma"] * Q[ns].max())
            Q[s, a] += P["Q"]["alpha"] * (target - Q[s, a])
        records.append([cfg, *pos, NB.index(heading), ti, a, contact, float(r[0]), bool(done[0]), s, ns,
                        uncertainty, true_probability, *q_before])
        if done[0] and not env.exhausted and env.frames < env.limit:
            filt.reset()
            s = filt.observe(obs[0, :64]) if visual else int(np.ravel_multi_index(env.worlds[0]["position"], (10, 10)))
        else:
            s = ns
    assert sum(e["decisions"] for e in env.completed) == env.frames
    assert sum(e["success"] for e in env.completed) == sum(row[7] for row in records)
    np.savez_compressed(path, records=np.array(records, np.float64))
    return dict(frames=env.frames, episodes=env.completed,
                observed_success=float(np.mean([e["success"] for e in env.completed])),
                records_sha256=sha(path), q_final=Q.copy(),
                resources=dict(fixed_atlas_bytes=atlas.nbytes,Q_bytes=Q.nbytes,
                               pose_posterior_bytes=filt.p.nbytes,
                               computational_likelihood_cache_bytes=sum(v.nbytes for v in filt.likelihoods.values()),
                               computational_likelihood_cache_keys=len(filt.likelihoods)))


def audit_run(cache, P, method, initial_Q, result, record_path):
    records = np.load(record_path, allow_pickle=False)["records"]
    assert sha(record_path) == result["records_sha256"]
    Q = initial_Q.copy(); prior = np.ones(100) / 100
    atlas = cache.features[0, :, :, :, 0, :64].reshape(100, 4, 64)
    last_a = None; last_contact = False; successes = 0; correct = 0
    max_p_error = 0.; max_q_error = 0.
    # Cache OBSERVATION likelihoods only, independently of runtime true pose.
    likelihoods = {}
    for index, row in enumerate(records):
        cfg, r, c, h, ti, a, contact, rew, done, s, ns = map(int, row[:11])
        assert cfg in range(8)
        walk, _ = cache.geometry(cfg)
        assert walk[r, c]
        dr, dc = NB[a]; rr, cc = r + dr, c + dc
        valid = 0 <= rr < 10 and 0 <= cc < 10 and walk[rr, cc]
        expected_position = (rr, cc) if valid else (r, c)
        assert contact == (not valid)
        assert rew == int(expected_position == cache.goal)
        assert (ns == -1) == bool(done)
        if not done:
            following = records[index+1]
            assert list(map(int,following[:3])) == [cfg,*expected_position]
            assert int(following[3]) == a and int(following[9]) == ns
        if method == "visual_belief_Q":
            cue = cache.features[cache.index[cfg], r, c, h, ti, :64]
            key = last_a, cue.tobytes()
            if key not in likelihoods:
                likelihoods[key] = independent_belief(np.ones(100)/100, atlas, cue, last_a, True, P["belief"]["sigma"])
            # Separate enumerated movement followed by cached likelihood.
            moved = np.zeros(100)
            for old, v in enumerate(prior):
                br, bc = divmod(old, 10)
                if last_a is not None and not last_contact:
                    drr, dcc = NB[last_a]; br, bc = br + drr, bc + dcc
                if 0 <= br < 10 and 0 <= bc < 10: moved[br*10+bc] += v
            prior = moved * likelihoods[key]; prior /= prior.sum()
            assert float(prior.max()-prior[s]) < 1e-12
            expected_p = float(prior[r*10+c])
            max_p_error = max(max_p_error, abs(expected_p - row[12]))
            assert expected_p > 0
        else:
            assert s == r*10+c
        max_q_error = max(max_q_error, float(np.max(np.abs(Q[s] - row[13:17]))))
        if result["learn"]:
            td_target = rew if done else rew + P["Q"]["gamma"] * float(max(Q[ns]))
            Q[s,a] = (1-P["Q"]["alpha"]) * Q[s,a] + P["Q"]["alpha"] * td_target
        correct += int(s == r*10+c); successes += rew
        last_a, last_contact = a, bool(contact)
        if done: prior = np.ones(100)/100; last_a = None; last_contact = False
    assert np.max(np.abs(Q-result["q_final"])) < 1e-12
    assert max_q_error < 1e-12 and max_p_error < 1e-12
    assert len(records) == result["frames"]
    assert successes == sum(e["success"] for e in result["episodes"])
    return dict(ok=True, decisions=len(records), successes=successes,
                pose_MAP_accuracy=correct/len(records),
                independent_posterior_max_error=max_p_error, independent_Q_max_error=max_q_error)


def main(dry=False):
    assert sha(PROTOCOL) == SHA and not os.environ.get("CUDA_VISIBLE_DEVICES", "")
    P = json.loads(PROTOCOL.read_text())
    assert sha(ROOT/P["parent"]) == P["parent_sha256"]
    assert sha(ROOT/P["cache"]) == P["cache_sha256"]
    cache = SourceCache(ROOT/P["cache"], ROOT/P["parent"])
    out = OUT/"dry" if dry else OUT
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/"results.json").exists(), "Preserve old results; no silent rerun"
    assert not list(out.glob("*_train.npz")), "Preserve any interrupted fits"
    if not dry:
        g=json.loads((OUT/"GO.json").read_text())
        assert g["status"]=="GO" and g["script_sha256"]==sha(__file__) and g["protocol_sha256"]==sha(PROTOCOL)
    results = []; checks = []; started = time.time()
    for method in P["methods"]:
        for seed in P["seeds"]:
            prefix = f"{method}_seed{seed}"
            Q = np.zeros((100, 4))
            train = run(cache, P, method, seed, Q, True, out/(prefix+"_train.npz"), dry=dry)
            train["learn"] = True
            checks.append(audit_run(cache, P, method, np.zeros_like(Q), train, out/(prefix+"_train.npz")))
            learned = Q.copy()
            np.savez_compressed(out/(prefix+"_checkpoint.npz"), Q=learned)
            selections = []
            for learn in (False, True):
                for repeat in range(1 if dry else 3):
                    Q = learned.copy(); schedule = selection_schedule(cache, repeat)
                    if dry: schedule = schedule[:2]
                    file = out/f"{prefix}_{int(learn)}_{repeat}.npz"
                    res = run(cache, P, method, 1000+seed*10+repeat, Q, learn, file, schedule, dry=dry)
                    res["learn"] = learn
                    if not learn: assert np.array_equal(Q, learned)
                    check = audit_run(cache, P, method, learned, res, file)
                    checks.append(check)
                    selections.append(dict(learn=learn, repeat=repeat, **check))
            results.append(dict(method=method, seed=seed, train_frames=train["frames"],resources=train["resources"],
                                train_success=train["observed_success"], selection=selections))
            print(json.dumps(dict(method=method,seed=seed,case_complete=True,dry=dry)),flush=True)
    assert len(results)==6 and all(c["ok"] for c in checks)
    report=dict(ok=True,dry=dry,source_only=True,biological_fits=0,target_fits=0,
                protocol_sha256=sha(PROTOCOL),script_sha256=sha(__file__),
                supervised_source_atlas=P["atlas"],runtime_priors=P["belief"]["information"],
                results=[] if dry else results,audits=checks,
                local_file_hashes={p.name:sha(p) for p in out.glob("*.npz")},
                seconds=time.time()-started,
                limits=P["limits"])
    (out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(ok=True,dry=dry,cases=6,decisions_audited=sum(c["decisions"] for c in checks))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--dry",action="store_true")
    main(parser.parse_args().dry)
