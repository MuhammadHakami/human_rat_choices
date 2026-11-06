"""WACV Vision4Neuro camera-ready analyses requested by the Program Chairs / reviewers (2026-10-31).
Frozen, already-audited seeds30 checkpoints (6 arms x 30 seeds) on the 16 pre-registered fresh confirmation rooms.
The probe is the unchanged E2 probe (wacv_continual_factorial_claude.probe); only (a) the goal cell and (b) the action
rule (registered = multinomial sample; greedy = argmax of the same probabilities) change. No learning.
  prereg                         write PROTOCOL.json (before any new outcome)
  probe COND WORKER NWORKERS     COND in CONDS; GPU 1 only
  bcacc                          held-out behaviour-cloning accuracy (grouped by trial), GPU 1
  analyse                        all registered + retrospective metrics -> analysis.json
"""
import gzip, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fresh_rooms_confirm_claude as FR

F = FR.F; worlds = FR.worlds; ROOT = FR.ROOT
OUT = ROOT / "data/audit/v4n_camera_ready_20261031"; PROTOCOL = OUT / "PROTOCOL.json"
ARMS = ("rat_bc", "rat_legal_uniform", "rat_sham", "human_bc", "human_legal_uniform", "cold_ppo")
S30 = FR.S30; SOURCE_GOAL = FR.GOAL
MOVED = ((3, 6), (6, 3), (6, 6))          # mirror images of (3,3) across the grid's mid-lines
CONDS = ("greedy_source", "sample_moved", "greedy_moved")
TRAP_WINDOW, TRAP_CELLS = 16, 4           # trapped = timeout with <=4 distinct cells in the final 16 decisions
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def moved_goal(walk, target, starts):
    """Registered rule: the target cell if walkable, not a start, and reachable from every start; else the nearest such
    cell (Manhattan, ties by row then column). Rooms are never filtered."""
    ok = lambda c: walk[c] and c not in starts and c != SOURCE_GOAL and all(s in worlds.fresh.source.goal_distances(walk, c) for s in starts)
    cells = sorted(((abs(r - target[0]) + abs(c - target[1]), r, c) for r in range(10) for c in range(10)))
    return next((r, c) for _, r, c in cells if ok((r, c)))


def goals_for(target):
    m = json.loads(FR.MAN.read_text()); starts = [tuple(s) for s in m["starts"]]
    return {r["configuration"]: moved_goal(np.asarray(r["walkable"], bool), target, starts) for r in m["rows"]}


def env_goal(goal_map, pair, n, seed, tier, phase_frames, schedule=None):
    _, _, source = worlds.fresh.load(); cache = worlds.RoomCache(source, json.loads(FR.MAN.read_text()))
    assert all(g == SOURCE_GOAL for g in cache.goals.values()) and len(cache.worlds) == 16
    if goal_map is not None: cache.goals.update(goal_map)
    phys = worlds.ContinuousRooms(cache, n=n, seed=82000 + seed, phase_frames=phase_frames, pair=pair, schedule=schedule)
    return worlds.VisualRooms(phys, source, tier, noise_seed=[81013, seed, int(tier == "noisy")])


_sample = F.sample
def greedy(m, observations):
    a, p, v = _sample(m, observations)                 # same RNG consumption as the registered sampler
    return p.argmax(-1), p, v


def episode_metrics(e, goal):
    path = [tuple(p) for p in e["path"]]; steps = path[1:]; seen = {path[0]}; revisits = 0
    for c in steps:
        revisits += c in seen; seen.add(c)
    blocked = sum(a == b for a, b in zip(path, steps))
    tail = steps[-TRAP_WINDOW:]
    return dict(success=bool(e["success"]), decisions=e["decisions"], spl=e["spl"], unique_cells=len(seen),
                revisit_rate=revisits / max(len(steps), 1), contact_rate=blocked / max(len(steps), 1),
                trapped=bool(not e["success"] and len(steps) >= 45 and len(set(tail)) <= TRAP_CELLS),
                visited_source_goal=SOURCE_GOAL in path, visited_moved_goal=tuple(goal) in path)


def pose_metrics(audit):
    a = np.asarray([x[2:] for x in audit], float)       # (argmax, true, p_true)
    return dict(n=len(a), top1=float((a[:, 0] == a[:, 1]).mean()), mean_p_true=float(a[:, 2].mean()),
                within1=float((np.abs(a[:, 0] // 10 - a[:, 1] // 10) + np.abs(a[:, 0] % 10 - a[:, 1] % 10) <= 1).mean()))


def summarise(rec, goal):
    eps = [episode_metrics(e, goal) for e in rec["episodes"] if "spawn" in e]; assert len(eps) == 30
    return dict(goals=rec["goals"], trials=len(eps), episodes=eps, pose=pose_metrics(rec["pose_audit"]))


def prereg():
    assert not PROTOCOL.exists(), "never overwrite the pre-registration"
    OUT.mkdir(parents=True, exist_ok=True)
    moved = {str(t): {str(k): list(v) for k, v in goals_for(t).items()} for t in MOVED}
    PROTOCOL.write_text(json.dumps(dict(
        written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), new_outcomes_seen=False,
        authority="Program Chairs' camera-ready requests (decision 2026-10-29): moved-goal test, pose-belief and evaluation-policy "
                  "clarification with loop-rate reporting; author 2026-10-31: take the upper limit, GPU 1.",
        checkpoints="seeds30 main family (6 arms x 30 seeds), sha256 bound by data/audit/wacv_fresh_rooms_confirm_claude/GO.json",
        rooms="16 fresh confirmation rooms (fresh_geometry_manifest.json, generated 2026-10-14 before any outcome)",
        probe="wacv_continual_factorial_claude.probe unchanged: 10 native starts x 3 textures = 30 trials/room, 45 decisions, clean",
        conditions=dict(registered_sample_source="existing confirmation records (multinomial sample, goal (3,3))",
                        greedy_source="argmax of the same action probabilities, goal (3,3)",
                        sample_moved="multinomial sample, each of 3 moved goals", greedy_moved="argmax, each of 3 moved goals"),
        moved_goal_rule="targets (3,6),(6,3),(6,6); nearest walkable non-start cell reachable from every start (Manhattan, row, col)",
        moved_goals=moved,
        primary=dict(
            M1="moved goal (sample, mean over 3 goals x 16 rooms): human_bc - human_legal_uniform, paired t over 30 seeds",
            M2="moved goal: human_bc - cold_ppo", M3="moved goal: rat_bc - rat_legal_uniform",
            G1="greedy source goal: human_bc - human_legal_uniform", multiplicity="Holm over M1-M3, G1"),
        secondary=dict(
            L1="revisit rate and trapped fraction per arm (all conditions incl. registered records); human_bc - human_legal_uniform paired",
            L2="old-goal attraction: fraction of moved-goal episodes entering (3,3), human_bc - human_legal_uniform and - cold",
            B1="pose belief top-1 / within-1 accuracy and mean posterior at the true cell, per condition, incl. retrospective generality relit/retextured records",
            BC="held-out BC accuracy and NLL (5 trial-grouped folds, same net/recipe, 3 seeds) for real, legal-uniform, sham and shortest-path labels"),
        reading="moved-goal gain positive and Holm p<.05 => choice content beyond goal memory; otherwise claims restricted to same-goal transfer",
        exclusions="none"), indent=1) + "\n")
    print(PROTOCOL.read_text()[:1500])


def probe(cond, worker, n):
    assert cond in CONDS and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    if cond == "greedy_moved": print(json.dumps(dict(cond=cond, skipped="deviation 1: deprioritised, CPU freed for learning studies"))); return
    proto = json.loads(PROTOCOL.read_text()); go = json.loads((FR.D / "GO.json").read_text())
    method = next(x for x in json.loads(F.study.MANIFEST.read_text())["methods"] if x["id"] == "ppo")
    targets = [SOURCE_GOAL] if cond.endswith("source") else list(MOVED)
    F.sample = greedy if cond.startswith("greedy") else _sample
    jobs = [(a, s, t, p, st) for s in S30 for a in ARMS for t in targets for p in range(8) for st in "AB"][worker::n]
    for a, s, t, p, st in jobs:
        f = OUT / "cohort" / cond / f"goal{t[0]}{t[1]}" / a / f"seed{s}" / f"pair{p}_{st}.json"
        if f.exists(): continue
        gm = None if t == SOURCE_GOAL else {int(k): tuple(v) for k, v in proto["moved_goals"][str(t)].items()}
        F.environment = lambda *x, gm=gm, **k: env_goal(gm, *x, **k)
        ck = sha(ROOT / "data/wacv/seeds30/cohort" / a / f"seed{s}" / "checkpoint_source.pt"); assert ck == go["checkpoints"][f"main/{a}/seed{s}"]
        rec, inp = FR.probe_one(ROOT / "data/wacv/seeds30/cohort", a, s, p, st, method); assert rec["trials"] == 30 and rec["weights_frozen_verified"] and inp == "full"
        goal = SOURCE_GOAL if gm is None else gm[100 + 2 * p + int(st == "B")]
        f.parent.mkdir(parents=True, exist_ok=True); full = f.with_suffix(".full.json.gz")
        full.write_bytes(gzip.compress(json.dumps(rec, default=float).encode()))
        f.write_text(json.dumps(dict(cond=cond, arm=a, seed=s, pair=p, room=st, goal=list(goal), checkpoint_sha256=ck, protocol_sha256=sha(PROTOCOL),
                                     full_record_sha256=sha(full), gpu_name=torch.cuda.get_device_name(0), **summarise(rec, goal))) + "\n")
    print(json.dumps(dict(cond=cond, worker=worker, done=len(jobs))), flush=True)


METRICS = ("success", "revisit_rate", "contact_rate", "trapped", "unique_cells", "visited_source_goal", "visited_moved_goal")


def per_seed(summaries):
    """summaries: list of probe summaries (one seed, all rooms/goals) -> seed-level means (success in /30 per room)."""
    eps = [e for r in summaries for e in r["episodes"]]
    out = {m: float(np.mean([e[m] for e in eps])) for m in METRICS}; out["success"] *= 30
    fails = [e for e in eps if not e["success"]]
    out["trapped_share_of_failures"] = float(np.mean([e["trapped"] for e in fails])) if fails else 0.
    for k in ("top1", "within1", "mean_p_true"): out[f"pose_{k}"] = float(np.mean([r["pose"][k] for r in summaries]))
    return out


def gather():
    """{family: {arm: {seed: seed-level metrics}}} for new conditions, registered confirmation records and generality."""
    G = {}
    for cond in CONDS:
        for a in ARMS:
            for s in S30:
                fs = sorted((OUT / "cohort" / cond).glob(f"goal*/{a}/seed{s}/pair*.json"))
                fs = [f for f in fs if not f.name.endswith(".full.json.gz")]
                if len(fs) == 16 * (1 if cond.endswith("source") else 3):
                    G.setdefault(cond, {}).setdefault(a, {})[s] = per_seed([json.loads(f.read_text()) for f in fs])
    load = lambda f, goal: summarise(json.loads(gzip.decompress(f.read_bytes())), goal)
    for a in ARMS:
        for s in S30:
            fs = sorted((FR.D / "cohort/main" / a / f"seed{s}").glob("pair*.full.json.gz")); assert len(fs) == 16
            G.setdefault("registered_sample_source", {}).setdefault(a, {})[s] = per_seed([load(f, SOURCE_GOAL) for f in fs])
    for c in ("noisy18", "clean26", "noisy26"):                       # pixel-noise / dense-room challenge records (FtUu: does noise break loops?)
        base = ROOT / "data/audit/wacv_fresh_challenge_claude/cohort" / c
        for d in sorted(base.iterdir()) if base.exists() else []:
            for sd in sorted(d.iterdir()):
                fs = sorted(sd.glob("pair*.full.json.gz"))
                if len(fs) == 16: G.setdefault(f"challenge_{c}", {}).setdefault(d.name, {})[int(sd.name[4:])] = per_seed([load(f, SOURCE_GOAL) for f in fs])
    for c in ("clean", "relit", "retextured"):
        for d in sorted((ROOT / "data/audit/wacv_generality_claude/cohort/main" / c).iterdir()):
            for sd in sorted(d.iterdir()):
                fs = sorted(sd.glob("pair*.full.json.gz"))
                if len(fs) == 16: G.setdefault(f"generality_{c}", {}).setdefault(d.name, {})[int(sd.name[4:])] = per_seed([load(f, SOURCE_GOAL) for f in fs])
    return G


def analyse():
    from prior_seed_power_claude import contrast
    from analyse_seeds30_claude import holm
    G = gather(); out = dict(protocol_sha256=sha(PROTOCOL), n_seeds={k: {a: len(v) for a, v in fam.items()} for k, fam in G.items()})
    out["per_seed"] = {k: {a: {str(sd): x for sd, x in v.items()} for a, v in fam.items()} for k, fam in G.items()}
    out["means"] = {k: {a: {m: float(np.mean([x[m] for x in v.values()])) for m in next(iter(v.values()))} for a, v in fam.items()} for k, fam in G.items()}
    metric = lambda fam, m: {a: {s: x[m] for s, x in v.items()} for a, v in G[fam].items()}
    pri = {}
    if "sample_moved" in G:
        S = metric("sample_moved", "success")
        pri.update(M1=contrast(S, "human_bc", "human_legal_uniform", 3.), M2=contrast(S, "human_bc", "cold_ppo", 3.), M3=contrast(S, "rat_bc", "rat_legal_uniform", 3.))
    if "greedy_source" in G: pri["G1"] = contrast(metric("greedy_source", "success"), "human_bc", "human_legal_uniform", 3.)
    for k, h in zip(pri, holm([v["t_p"] for v in pri.values()])): pri[k]["holm_t_p"] = h
    out["primary"] = pri; sec = {}
    for fam in G:
        for m in ("success", "revisit_rate", "trapped", "contact_rate", "visited_source_goal", "pose_top1"):
            S = metric(fam, m)
            for a, b in (("human_bc", "human_legal_uniform"), ("human_bc", "cold_ppo"), ("rat_bc", "rat_legal_uniform")):
                if a in S and b in S and len(set(S[a]) & set(S[b])) > 2: sec[f"{fam}/{m}/{a}-{b}"] = contrast(S, a, b, 3. if m == "success" else .1)
    out["secondary"] = sec
    # FtUu: is the human gain only less looping? Success among non-trapped trials, and the seed-level regression of the
    # success gain on the trapped-rate gain (intercept = gain at equal looping; descriptive, post hoc).
    lm = {}
    for fam in ("registered_sample_source", "sample_moved"):
        if fam not in G: continue
        ss = sorted(set(G[fam]["human_bc"]) & set(G[fam]["human_legal_uniform"]))
        ds = np.array([G[fam]["human_bc"][s]["success"] - G[fam]["human_legal_uniform"][s]["success"] for s in ss]) * 100 / 30
        dt = np.array([G[fam]["human_bc"][s]["trapped"] - G[fam]["human_legal_uniform"][s]["trapped"] for s in ss]) * 100
        X = np.c_[np.ones_like(dt), dt]; beta, res, *_ = np.linalg.lstsq(X, ds, rcond=None)
        sig2 = float(((ds - X @ beta) ** 2).sum() / (len(ds) - 2)); se = np.sqrt(np.diag(sig2 * np.linalg.inv(X.T @ X)))
        from scipy import stats as st
        q = st.t.ppf(.975, len(ds) - 2)
        lm[fam] = dict(n=len(ss), intercept_pp=float(beta[0]), intercept_ci=[float(beta[0] - q * se[0]), float(beta[0] + q * se[0])],
                       slope=float(beta[1]), mean_dsuccess=float(ds.mean()), mean_dtrapped=float(dt.mean()))
    out["loop_adjusted"] = lm
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, fam in out["means"].items():
        print(k, {a: {m: round(v[m], 3) for m in ("success", "revisit_rate", "trapped", "trapped_share_of_failures", "visited_source_goal", "pose_top1", "pose_mean_p_true")} for a, v in fam.items()})
    print(json.dumps({k: {q: v[q] for q in ("mean", "ci95", "t_p", "holm_t_p")} for k, v in pri.items()}, indent=1))


def check():
    w = np.ones((10, 10), bool); assert moved_goal(w, (6, 6), [(0, 9)]) == (6, 6)
    w[6, 6] = False; assert moved_goal(w, (6, 6), [(0, 9)]) == (5, 6)
    e = dict(path=[[0, 0]] + [[0, 1], [0, 0]] * 22 + [[0, 1]], success=False, decisions=45, spl=0.)
    m = episode_metrics(e, (6, 6)); assert m["trapped"] and m["unique_cells"] == 2 and abs(m["revisit_rate"] - 44 / 45) < 1e-9
    print("check ok")


if __name__ == "__main__":
    cmd = sys.argv[1]; torch.set_num_threads(1)
    if cmd == "check": check()
    elif cmd == "prereg": prereg()
    elif cmd == "probe": probe(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "analyse": analyse()
