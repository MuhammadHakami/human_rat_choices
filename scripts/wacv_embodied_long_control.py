"""Long conventional source control: shared FF64x64 PPO vs LSTM19 PPO on a frozen belief scaffold. SOURCE ONLY.

Protocol: docs/WACV_EMBODIED_LONG_CONTROL_PROTOCOL_20261012.json (2 architectures x 2 rates x 3 seeds = 12 fits,
524,288 frames each). Reused UNCHANGED (imported, never edited): the reviewed PPO collector/objective
(recurrent_ppo.py), the training/selection loops and episode records (wacv_embodied_source_runner.py), the physical
source env and selection schedule, and the frozen 250-dim belief scaffold (wacv_embodied_spatial_source.SpatialBatch).
New here: 2 posterior-mean coordinates (fixed 100x2 matrix, 800 B, rows/cols scaled to [-1,1]) -> 252 inputs; the FF
network (ppo_feedforward_reference.py); the long curriculum; resource/MAC accounting; an independent auditor.
Commands (CPU only, CUDA_VISIBLE_DEVICES must be empty):
  check                      causal belief/coordinate input check on a short stream (no learning)
  dry ARCH --out DIR         technical run: 1,024 training frames, 2 selection trials per arm (no scores read)
  fit ARCH RATE_IDX SEED --out DIR   registered fit; requires root GO bound to current hashes and DIR
  audit --out DIR [--dry]    independent physics/reward/frame/hash replay + frozen-arm checkpoint replay
  select --out DIR           registered rule over the audited 12 fits (exact integer counts over 180 trials)
"""
import argparse, copy, gzip, hashlib, json, os, resource, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wacv_embodied_source_runner as base
import wacv_embodied_spatial_source as spatial
from wacv_embodied_source_cache import SourceCache
from wacv_embodied_source_env import selection_schedule
from wacv_embodied_belief_source import independent_belief
from recurrent_ppo import RecurrentPPO
from ppo_feedforward_reference import FeedforwardPPO
from arc_fly.unimaze import NB

PROTOCOL = ROOT / "docs/WACV_EMBODIED_LONG_CONTROL_PROTOCOL_20261012.json"
PROTOCOL_SHA = "4cb5604f7b7047e5862313a28630d8fcdc34812dc7626e46865e26ce079dadf1"
GO = ROOT / "data/audit/wacv_embodied_source_20261012/long_control/GO.json"
CODE = sorted(set(spatial.CODE) | {"scripts/wacv_embodied_long_control.py", "scripts/ppo_feedforward_reference.py"})
ARCHS = ("ppo_shared_ff64x64", "ppo_lstm19")
ARMS = ("frozen", "adaptive")
COORD = np.array([[2 * (s // 10) / 9 - 1, 2 * (s % 10) / 9 - 1] for s in range(100)], np.float32)   # 800 B


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    assert sha(PROTOCOL) == PROTOCOL_SHA, "Registered long-control protocol changed"
    L = json.loads(PROTOCOL.read_text())
    assert sha(ROOT / L["parent"]) == L["parent_sha256"] == spatial.EXPECTED
    _, S = spatial.load()                                       # verifies spatial protocol + representation hashes
    P = copy.deepcopy(S)
    P["source"].update(curriculum=L["curriculum"], training_frames=L["training_frames_each"], seeds=L["seeds"],
                       train_configurations=L["train_configurations"], select_configurations=L["select_configurations"])
    raw = SourceCache(base.MANIFEST, ROOT / S["parent_protocol"])
    cache = spatial.SpatialCache(raw, P); cache.obs_dim = L["observations"]["dim"]
    assert cache.obs_dim == 252 and COORD.nbytes == L["observations"]["fixed_coordinate_matrix_bytes"]
    return cache, P, L


def hashes():
    return dict(protocol=sha(PROTOCOL), cache_manifest=sha(base.MANIFEST),
                cache_array=sha(base.MANIFEST.with_suffix(".npz")), code={c: sha(ROOT / c) for c in CODE})


def require_go(out):
    g = json.loads(GO.read_text())
    assert g["status"] == "GO" and g["hashes"] == hashes(), "Bound long-control code/data changed"
    assert Path(out).resolve() == Path(g["output"]).resolve(), "Unregistered output cohort"


class LongBatch(spatial.SpatialBatch):
    """Frozen belief scaffold (250) + posterior-mean row/col in [-1,1] (2). No simulator pose or goal bearing."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.obs_dim = 252

    def augment(self, obs, actions=None, done=None):
        x = super().augment(obs, actions, done)
        return np.concatenate((x, x[:, 150:250] @ COORD), axis=1).astype(np.float32)


def build(arch, rate, seed, L):
    c, A = L["common_learning"], L["architectures"][arch]
    kw = dict(n_envs=c["n_envs"], n_steps=c["n_steps"], gamma=c["gamma"], lam=c["gae_lambda"], lr=rate, clip=c["clip"],
              ent=c["entropy"], vf=c["value_coefficient"], epochs=c["epochs"], minibatches=c["minibatches"], seed=seed)
    assert c["max_grad_norm"] == 0.5 and c["adam_epsilon"] == 1e-5 and c["advantage_normalization"]   # reviewed defaults
    net = FeedforwardPPO(252, 4, layers=tuple(A["shared_layers"]), **kw) if arch == ARCHS[0] else \
        RecurrentPPO(252, 4, hidden=A["hidden"], **kw)
    n = sum(p.numel() for p in net.net.parameters()); assert n == A["expected_parameters"], (arch, n)
    return net


def forward_macs(net):
    if isinstance(net, FeedforwardPPO): return net.forward_macs()
    h, i = net.net.lstm.hidden_size, net.net.lstm.input_size
    return 4 * h * (i + h) + h * 5


def belief_ops(envs):
    """Belief-filter operations, counted from actual memoized likelihoods (templates x 64 x 3) + 3 ops x 100 cells
    per filter step (shift, multiply, normalise) + 200 MACs for the coordinate means."""
    lik = sum((400 if k[0] is None else 100) * 64 * 3 for e in envs for f in e.filters for k in f.likelihoods)
    steps = sum(e.frames for e in envs)
    return dict(likelihood_ops=lik, per_step_ops=steps * (300 + 200), filter_steps=steps,
                representation=[e.representation_resources() for e in envs], coordinate_matrix_bytes=COORD.nbytes)


def resources(net, envs, phase_counts=None, learning_frames=None):
    b = lambda ts: int(sum(t.element_size() * t.numel() for t in ts))
    m = net.memory_bytes(); fwd = forward_macs(net)
    counts = dict(net.count) if phase_counts is None else phase_counts
    learned = counts.get("updates", 0) * net.cfg["n_steps"] * net.cfg["n_envs"] if learning_frames is None else learning_frames
    policy_frames = counts["frames"]
    return dict(weights_bytes=b(net.net.parameters()), optimiser_bytes=m["optimiser"], state_bytes=m["persistent_state"],
                used_recurrent_state=0 if isinstance(net, FeedforwardPPO) else net.hidden,
                buffer_bytes=m["experience_buffer"], plastic_parameters=int(sum(p.numel() for p in net.net.parameters())),
                counters=counts, cumulative_counters=dict(net.count),
                forward_macs_per_decision=fwd, forward_macs_total=fwd * policy_frames,
                learning_macs_estimate=net.cfg["epochs"] * 3 * fwd * learned, belief=belief_ops(envs),
                cost_scope="Currentphase policy decisions; learning estimate charges actual learned frames, including final short segment. Bootstrap/value and backward costs remain estimates, not hardware energy.",
                maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def arm_seed(seed, rate_idx, arch, arm, rep):
    return [20261012, 100 + ARCHS.index(arch), rate_idx, seed, ARMS.index(arm), rep]


def select_arm(trained, cache, arch, rate_idx, seed, arm, rep, trials=None):
    net = copy.deepcopy(trained); base.reseed(net, arm_seed(seed, rate_idx, arch, arm, rep)); count0 = dict(net.count)
    sched = selection_schedule(cache, rep)[:trials] if trials else selection_schedule(cache, rep)
    env = LongBatch(cache, n=1, seed=0, frames=len(sched) * 45, schedule=sched)
    learn = arm == "adaptive"; w0 = [t.clone() for t in net.net.parameters()]; t0 = time.perf_counter()
    probs = base.run_ppo_select(net, env, learn)
    if not learn: assert all(torch.equal(a, b) for a, b in zip(w0, net.net.parameters())), "frozen arm changed weights"
    assert len(env.completed) == len(sched)
    eps = base.episodes(env, probs, net._values_log)
    delta={k: net.count.get(k, 0) - count0.get(k, 0) for k in set(net.count) | set(count0)}
    return dict(successes=int(sum(e["success"] for e in eps)), trials=len(eps), decisions=env.frames, episodes=eps,
                wall_s=time.perf_counter() - t0, resources=resources(net, [env], delta, env.frames if learn else 0),
                counters_delta=delta, frozen_weights_identical=not learn)


def gz_write(path, obj):
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:   # byte-reproducible
        gz.write(json.dumps(obj).encode())
    return sha(path)


def fit(arch, rate_idx, seed, out, dry=False):
    torch.set_num_threads(1); t0 = time.time()
    if not dry: require_go(out)
    cache, P, L = load(); assert arch in ARCHS and (dry or seed in L["seeds"])
    rate = L["rates"][rate_idx]; frames = L["common_learning"]["n_steps"] * L["common_learning"]["n_envs"] if dry \
        else L["training_frames_each"]
    d = Path(out) / arch / f"rate{rate_idx}" / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    rec = dict(arch=arch, rate_index=rate_idx, rate=rate, seed=seed, dry=dry, frames_planned=frames, hashes=hashes(),
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status="running")
    env = net = None
    try:
        net = build(arch, rate, seed, L)
        env = LongBatch(cache, n=L["common_learning"]["n_envs"], seed=seed, frames=frames)
        probs = base.run_ppo_train(net, env); assert env.frames == frames
        train_eps = base.episodes(env, probs, net._values_log)
        torch.save(net, d / "checkpoint_source.pt"); rec["checkpoint_sha256"] = sha(d / "checkpoint_source.pt")
        rec["train"] = dict(frames=env.frames, episodes=len(env.completed), resources=resources(net, [env]),
                            successes=int(sum(e["success"] for e in env.completed)))
        sel = {f"{arm}_{r}": select_arm(net, cache, arch, rate_idx, seed, arm, r, trials=2 if dry else None)
               for arm in ARMS for r in range(3)}
        rec["selection"] = {k: {q: v[q] for q in v if q != "episodes"} for k, v in sel.items()}
        for arm in ARMS:
            rec[f"{arm}_successes"] = sum(sel[f"{arm}_{r}"]["successes"] for r in range(3))
            rec[f"{arm}_trials"] = sum(sel[f"{arm}_{r}"]["trials"] for r in range(3))
        rec["episodes_sha256"] = gz_write(d / "episodes.json.gz", dict(train=train_eps, selection={k: v["episodes"] for k, v in sel.items()}))
        rec["status"] = "complete"
    except (FloatingPointError, ValueError) as err:          # retained, never retried
        rec.update(status="failed", error=repr(err), traceback=traceback.format_exc()[-2000:],
                   failure_phase="selection" if "checkpoint_sha256" in rec else "training",
                   training_frames_completed=env.frames if env is not None else 0)
        if env is not None: rec["partial_training_episodes_sha256"] = gz_write(d / "failed_partial_training_episodes.json.gz", env.completed)
    rec.update(wall_s=round(time.time() - t0, 2), maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               note="technical dry run; no scientific interpretation" if dry else "registered long-control source fit")
    (d / "result.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("arch", "rate", "seed", "dry", "status", "wall_s", "adaptive_successes", "frozen_successes")}))
    return rec


# ---- independent checks ---------------------------------------------------------------------------------------------
def replay_episode(cache, e):
    """Physics/reward from geometry alone: move if in-bounds and walkable, reward 1 on entering the goal, cap 45."""
    walk, _ = cache.geometry(e["configuration"]); p = tuple(e["spawn"]); path = [p]
    for a in e["actions"]:
        q = (p[0] + NB[a][0], p[1] + NB[a][1])
        if 0 <= q[0] < 10 and 0 <= q[1] < 10 and walk[q]: p = q
        path.append(p)
        if p == cache.goal: break
    ok = [list(x) for x in path] == e["path"] and e["decisions"] == len(e["actions"]) <= 45
    return ok and e["success"] == (p == cache.goal) and e["reward_sum"] == float(p == cache.goal) and \
        (e["success"] or e["decisions"] == 45 or e["forced_boundary"])


def audit(out, dry=False):
    cache, P, L = load(); rows = []; ok = True
    for arch in ARCHS:
        for ri in ([0] if dry else range(len(L["rates"]))):
            for seed in ([0] if dry else L["seeds"]):
                f = Path(out) / arch / f"rate{ri}" / f"seed{seed}" / "result.json"
                if not f.exists(): rows.append(dict(arch=arch, rate_index=ri, seed=seed, present=False)); ok = False; continue
                r = json.loads(f.read_text()); row = dict(arch=arch, rate_index=ri, seed=seed, present=True, status=r["status"])
                ok &= r["dry"] == dry and r["hashes"]["code"] == hashes()["code"]
                if r["status"] == "complete":
                    d = f.parent; assert sha(d / "episodes.json.gz") == r["episodes_sha256"]
                    assert sha(d / "checkpoint_source.pt") == r["checkpoint_sha256"]
                    E = json.loads(gzip.decompress((d / "episodes.json.gz").read_bytes()))
                    train = [e for e in E["train"] if "actions" in e]
                    row["train_frames_exact"] = sum(e["decisions"] for e in train) == r["frames_planned"] == r["train"]["frames"]
                    row["physics_ok"] = all(replay_episode(cache, e) for e in train) and all(
                        replay_episode(cache, e) for v in E["selection"].values() for e in v)
                    row["selection_counts_ok"] = all(sum(e["success"] for e in E["selection"][k]) == r["selection"][k]["successes"]
                                                     and len(E["selection"][k]) == (2 if dry else 20) for k in E["selection"])
                    net = torch.load(d / "checkpoint_source.pt", weights_only=False)     # frozen-arm checkpoint replay
                    rep = select_arm(net, cache, arch, ri, seed, "frozen", 0, trials=2 if dry else None)
                    row["frozen_replay_identical"] = [e["actions"] for e in rep["episodes"]] == \
                        [e["actions"] for e in E["selection"]["frozen_0"]] and \
                        [e["probabilities"] for e in rep["episodes"]] == [e["probabilities"] for e in E["selection"]["frozen_0"]]
                    ok &= all(row[k] for k in ("train_frames_exact", "physics_ok", "selection_counts_ok", "frozen_replay_identical"))
                rows.append(row)
    res = dict(ok=bool(ok), dry=dry, cases=sum(r["present"] for r in rows), expected=2 if dry else 12, rows=rows,
               scope="Source accounting/replay of long control; not target, biological or transfer evidence")
    (Path(out) / "audit.json").write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps({k: res[k] for k in ("ok", "dry", "cases")}))
    return res


def check_inputs(steps=400):
    """Causality/content of the 252 inputs on one random-action stream: posterior equals an independent enumeration
    from camera/action/contact only; coordinates equal posterior @ COORD; no pose/goal enters."""
    cache, P, L = load(); env = LongBatch(cache, n=1, seed=7, frames=steps); rng = np.random.default_rng(3)
    atlas = cache.raw.features[0, :, :, :, 0, :64].reshape(100, 4, 64)
    obs = env.reset(); q = independent_belief(np.ones(100) / 100, atlas, obs[0, :64], None, False, .03); err = 0.0; ce = 0.0
    err = max(err, float(np.abs(q - obs[0, 150:250]).max()))
    while env.frames < steps:
        a = int(rng.integers(4)); obs, r, done = env.step(np.array([a]))
        if done[0] and env.frames < steps: q = independent_belief(np.ones(100) / 100, atlas, obs[0, :64], None, False, .03)
        elif not done[0]: q = independent_belief(q, atlas, obs[0, :64], a, bool(obs[0, 144]), .03)
        else: continue
        err = max(err, float(np.abs(q - obs[0, 150:250]).max()))
        ce = max(ce, float(np.abs(obs[0, 250:252] - obs[0, 150:250] @ COORD).max()))
    assert err < 1e-5 and ce < 1e-6, (err, ce)
    res = dict(ok=True, steps=steps, belief_max_err=err, coordinate_max_err=ce, protocol_sha256=sha(PROTOCOL))
    print(json.dumps(res)); return res


def check_budget():
    """No learning: random actions through the full registered budget on 8 slots. Exact frames, forced terminals at
    every curriculum boundary, phase configurations respected, every reward credited once, physics replay."""
    cache, P, L = load(); n = L["common_learning"]["n_envs"]; env = LongBatch(cache, n=n, seed=11)
    rng = np.random.default_rng(5); env.reset(); credited = 0.0; bounds = [p["until_total_frames"] for p in L["curriculum"]]
    while env.frames < env.limit:
        obs, r, done = env.step(rng.integers(4, size=n)); credited += float(r.sum())
        assert obs.shape == (n, 252) and np.isfinite(obs).all()
        if env.frames in bounds:
            assert done.all()
            if env.frames < env.limit: assert all(w["configuration"] in env._phase()["configurations"] for w in env.worlds)
    assert env.frames == L["training_frames_each"] == sum(e["decisions"] for e in env.completed)
    assert credited == sum(e["reward_sum"] for e in env.completed)
    assert all(replay_episode(cache, e) for e in env.completed)
    res = dict(ok=True, frames=env.frames, episodes=len(env.completed), forced=sum(e["forced_boundary"] for e in env.completed),
               configurations=sorted({e["configuration"] for e in env.completed}), belief=belief_ops([env]))
    print(json.dumps({k: v for k, v in res.items() if k != "belief"})); return res


def select(out):
    cache, P, L = load(); a = json.loads((Path(out) / "audit.json").read_text())
    root = json.loads((Path(out) / "root_audit.json").read_text())
    assert root["ok"] and not root["dry"] and root["cases"] == 12 and root["hashes"] == hashes()
    assert a["ok"] and not a["dry"] and a["cases"] == 12, "selection requires the complete audited 12-fit cohort"
    choice = {}
    for arch in ARCHS:
        cands = []
        for ri, rate in enumerate(L["rates"]):
            rs = [json.loads((Path(out) / arch / f"rate{ri}" / f"seed{s}" / "result.json").read_text()) for s in L["seeds"]]
            cands.append(dict(rate_index=ri, rate=rate, failed=sum(r["status"] == "failed" for r in rs), denominator=180,
                              adaptive_count=sum(r.get("adaptive_successes", 0) for r in rs),   # failed: null accuracy,
                              frozen_count=sum(r.get("frozen_successes", 0) for r in rs)))      # zero selector utility
        best = min(cands, key=lambda c: (-c["adaptive_count"], -c["frozen_count"], c["rate_index"]))
        choice[arch] = dict(best, adaptive=best["adaptive_count"] / 180, frozen=best["frozen_count"] / 180,
                            competence_target_met=best["adaptive_count"] / 180 >= .5, candidates=cands)
    (Path(out) / "selection.json").write_text(json.dumps(dict(rule=L["selection"], choice=choice), indent=1) + "\n")
    print(json.dumps({k: (v["rate"], v["adaptive"], v["frozen"]) for k, v in choice.items()}))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check", "dry", "fit", "audit", "select"])
    ap.add_argument("arch", nargs="?"); ap.add_argument("rate", nargs="?", type=int); ap.add_argument("seed", nargs="?", type=int)
    ap.add_argument("--out"); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "CPU only"
    torch.set_num_threads(1)
    if a.cmd == "check": check_inputs(); check_budget()
    elif a.cmd == "dry": fit(a.arch, 0, 0, a.out, dry=True)
    elif a.cmd == "fit": fit(a.arch, a.rate, a.seed, a.out)
    elif a.cmd == "audit": audit(a.out, a.dry)
    else: select(a.out)


if __name__ == "__main__":
    main()
