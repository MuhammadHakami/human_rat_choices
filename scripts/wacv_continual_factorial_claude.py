"""Prior x learner x withheld-room continual factorial (Claude wrapper; scientific fits only after a Codex protocol/GO).
Reuses the audited GO48 development code unchanged (build, sample, LoggingRooms, resource_report, learners, rooms) and
changes only: (1) the source checkpoint = 30-seed replication checkpoint of a prior arm, (2) the room pair (1-7 withheld,
0 for dry plumbing), (3) the output directory. Goals follow Addendum2 exactly as the development runner computes them.
  python scripts/wacv_continual_factorial_claude.py dry  ARM METHOD SEED PAIR TIER
  python scripts/wacv_continual_factorial_claude.py fit  ARM METHOD SEED PAIR TIER        (needs BASE/GO.json)
"""
import argparse, copy, json, os, time
from pathlib import Path

import numpy as np
import torch

import wacv_continual_geometry_development_20261012 as study
from wacv_continual_geometry_development_20261012 import (ProtectedAC, FeedforwardPPO, ReplayRegularizedPPO,  # noqa: F401 (pickling)
                                                           DeviceFFAgent, worlds, sample, tensor_parameters, LoggingRooms, resource_report, sha)

ROOT = study.ROOT
BASE = ROOT / "data/audit/wacv_continual_factorial_claude"
SEEDS30 = ROOT / "data/wacv/seeds30/cohort"
ARMS = ("rat_bc", "rat_legal_uniform", "human_bc", "cold_ppo")
METHODS = ("ppo", "ppo_replay")


def binding():
    return dict(wrapper=sha(Path(__file__)), development=study.binding(), seeds30_audit=sha(ROOT / "data/wacv/seeds30/root_audit.json"))


def source_checkpoint(arm, seed):
    r = json.loads((SEEDS30 / arm / f"seed{seed}" / "result.json").read_text()); f = SEEDS30 / arm / f"seed{seed}" / "checkpoint_source.pt"
    assert r["status"] == "complete" and sha(f) == r["checkpoint_sha256"]
    return torch.load(f, map_location="cpu", weights_only=False)


def environment(pair, n, seed, tier, phase_frames, schedule=None):
    e = study.environment(n, seed, tier, phase_frames, schedule)          # Addendum2 goals for all pairs, pair-0 stream
    phys = e.env
    return worlds.VisualRooms(worlds.ContinuousRooms(phys.cache, n=n, seed=82000 + seed, phase_frames=phase_frames, pair=pair, schedule=schedule),
                              worlds.fresh.load()[2], tier, noise_seed=[81013, seed, int(tier == "noisy")])


@torch.no_grad()
def probe(agent, pair, stage, seed, tier, dry=False):
    cfg = 100 + 2 * pair + int(stage == "B")
    starts = json.loads((worlds.fresh.BASE / "geometry_manifest.json").read_text())["starts"]
    schedule = [dict(configuration=cfg, spawn=s, rotation_index=rot) for rot in range(3) for s in starts]
    if dry: schedule = schedule[:2]
    m = copy.deepcopy(agent); device = "cuda:0" if isinstance(m, ProtectedAC) else "cpu"
    m.tg = torch.Generator(device=device).manual_seed(230017 + seed); rng0 = m.tg.get_state().cpu().tolist()
    env = environment(pair, 1, seed, tier, 45 * len(schedule) // 2, schedule); obs = env.reset()
    before = {k: p.clone() for k, p in tensor_parameters(m).items()}; probs, vals = [], []
    while not env.exhausted and env.frames < env.limit:
        a, p, v = sample(m, obs); probs.append(p[0].cpu().tolist()); vals.append(float(v[0])); obs, _, _ = env.step(a.cpu().numpy())
    assert all(torch.equal(p, before[k]) for k, p in tensor_parameters(m).items())
    eps = study.ppo_source.controls.parent.base.episodes(env, probs, vals); alt = env.cache.goals[cfg + (-1 if stage == "B" else 1)]
    return dict(goals=sum(e["success"] for e in eps), trials=len(eps), frames=env.frames, mean_spl=float(np.mean([e["spl"] for e in eps])),
                mean_decision_efficiency=float(np.mean([e["decision_efficiency"] for e in eps])),
                alternative_goal_visit_fraction=float(np.mean([list(alt) in e["path"] for e in eps])),
                rng_initial=rng0, weights_frozen_verified=True, episodes=eps, pose_audit=env.pose_audit)


def fit(arm, method_id, seed, pair, tier, dry=False):
    assert arm in ARMS and method_id in METHODS and tier in ("clean", "noisy") and 0 <= pair < 8
    if not dry:
        go = json.loads((BASE / "GO.json").read_text()); assert go["status"] == "GO" and go["hashes"] == binding()
        assert pair in go["pairs"] and seed in go["seeds"] and arm in go["arms"] and method_id in go["methods"] and tier in go["tiers"]
    manifest = json.loads(study.MANIFEST.read_text()); method = next(m for m in manifest["methods"] if m["id"] == method_id)
    d = BASE / ("dry" if dry else "cohort") / arm / method_id / f"pair{pair}" / tier / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    with (d / "started.json").open("x") as f: json.dump(dict(hashes=binding()), f)
    study.source_model = lambda s: source_checkpoint(arm, s)              # ponytail: only the initial weights change
    t0 = time.perf_counter(); m = study.build(method, seed); count = 2048 if dry else 65536
    env = environment(pair, 16, seed, tier, count); obs = env.reset(); wrapped = LoggingRooms(env, obs)
    rng0 = m.tg.get_state().cpu().numpy().copy(); ph = dict(A_initial=probe(m, pair, "A", seed, tier, dry)); probs, values = [], []
    state = (torch.zeros(1, 16, 0), torch.zeros(1, 16, 0)); dprev = np.zeros(16, np.float32)
    while m.count["frames"] < 2 * count:
        if m.count["frames"] == count:
            ph["A_after_A_learning"] = probe(m, pair, "A", seed, tier, dry); ph["B_before_learning"] = probe(m, pair, "B", seed, tier, dry)
            torch.save(m, d / "checkpoint_afterA.pt")
        B, s0, obs, state, dprev = m.rollout(wrapped, obs, state, dprev); probs.extend(B["probs"].numpy()); values.extend(B["val"].numpy()); m.update(B, s0)
    ph["A_after_B_learning"] = probe(m, pair, "A", seed, tier, dry); ph["B_after_learning"] = probe(m, pair, "B", seed, tier, dry)
    assert m.count["frames"] == env.frames == 2 * count; torch.save(m, d / "checkpoint_afterB.pt")
    arr = {k: np.asarray(v) for k, v in wrapped.log.items()}
    np.savez_compressed(d / "training.npz", **arr, probabilities=np.asarray(probs), values=np.asarray(values), action_rng_initial=rng0, pose_audit=np.asarray(env.pose_audit))
    study.ppo_source.controls.parent.gz_write(d / "probes.json.gz", ph)
    a, b = 100 + 2 * pair, 101 + 2 * pair; realized = {str(c): int((arr["configurations"] == c).sum()) for c in (a, b)}
    assert sum(realized.values()) == 2 * count and realized[str(a)] >= count
    rec = dict(arm=arm, method=method, seed=seed, pair=pair, tier=tier, dry=dry, status="complete", hashes=binding(), phase_frames=count,
               goals={str(a): list(env.cache.goals[a]), str(b): list(env.cache.goals[b])}, realized_interactions=realized, counters=m.count,
               probes={k: {q: v for q, v in x.items() if q not in ("episodes", "rng_initial", "pose_audit")} for k, x in ph.items()},
               resources=resource_report(m, env), wall_s=time.perf_counter() - t0, gpu_name=torch.cuda.get_device_name(0),
               files={n: sha(d / n) for n in ("training.npz", "probes.json.gz", "checkpoint_afterA.pt", "checkpoint_afterB.pt")})
    (d / "result.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(dict(arm=arm, method=method_id, seed=seed, pair=pair, tier=tier, dry=dry, status="complete", wall_s=rec["wall_s"])), flush=True)


def queue(worker, nworkers):
    go = json.loads((BASE / "GO.json").read_text())
    jobs = [(a, m, s, p, t) for s in go["seeds"] for p in go["pairs"] for t in go["tiers"] for a in go["arms"] for m in go["methods"]]
    for a, m, s, p, t in jobs[worker::nworkers]:
        if (BASE / "cohort" / a / m / f"pair{p}" / t / f"seed{s}" / "result.json").exists(): continue
        fit(a, m, s, p, t)


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "queue":
    torch.set_num_threads(1); queue(int(__import__("sys").argv[2]), int(__import__("sys").argv[3])); raise SystemExit

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["dry", "fit"])
    for k in ("arm", "method"): ap.add_argument(k)
    for k in ("seed", "pair"): ap.add_argument(k, type=int)
    ap.add_argument("tier"); a = ap.parse_args(); torch.set_num_threads(1)
    assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" and os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1")
    fit(a.arm, a.method, a.seed, a.pair, a.tier, a.command == "dry")
