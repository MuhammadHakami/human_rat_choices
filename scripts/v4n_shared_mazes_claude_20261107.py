"""Shared mazes (camera-ready addition, author 2026-11-07: "i want our layout of the maps also to follow some human/rat ... mazes
so we can compare more geometrically for the exact same area between them while retaining our accepted criteria"; "ours is about
OOD generalization on unseen so our paper still within the accepted criteria yet we adapt the other for the same models").
Registered before any agent outcome in these rooms. No training: the SAME frozen and learned networks as the paper are probed.

Rooms: the original Tartarus layouts (de Cothi et al. 2022, Zenodo 10.5281/zenodo.6734981) that both species walked and that no
network has seen. Imitation used choices from configurations 1-5 only (docs/WACV_NATIVE_BEHAVIOR_PRIOR_SOURCE_PROTOCOL_20261012.json,
native_configurations), so configurations 6-25 (20 layouts) are used. Each keeps its OWN 10 native starts (the humans' and
rats' starts). Accepted room criteria are retained and checked: 10 x 10, familiar goal (3,3), every open cell connected to it,
every start within 30 steps (holds for all 20; blocked cells 9-19 vs 18 in our generator). Moved goal: the paper's rule
(nearest walkable cell to (6,6), not a start, not the familiar goal, reachable from every start) with each layout's own starts.
Layouts are read at run time from data/unified (raw-derived; never written to the outputs or the export).
Probe: the registered frozen probe (F.probe) unchanged except the schedule (own 10 starts x 3 floor rotations = 30 trials, 45
decisions, act ~ softmax(logits), weights verified frozen). Both goals, every layout.
Agents (30 seeds each): frozen networks trained on human / rat actions; after learning the moved goal (paper's test records):
PPO, stage-3 brain circuit, stage-7 goal map (ungated), from human_bc, human_legal_uniform, cold_ppo.
Outcomes (descriptive; no primary): success per goal; on the familiar goal, the same layouts and starts as the recorded humans
and rats (configurations 6-25, all subjects, paths truncated at 45 moves): success within 45 moves, revisit rate (moves into an
already visited cell / moves), excess path (moves / shortest, successful trials), occupancy similarity per layout
(Bhattacharyya coefficient of visited-cell distributions) to humans and to rats, with human-vs-rat as the reference.
  check | prereg | run W N | analyse
"""
import json, os, sys, time, hashlib, collections
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_brainlike_claude_20261103 as B
G = B.G; M = B.M; F = M.F; CR = M.CR; worlds = M.worlds
KC0 = B.KC
OUT = M.ROOT / "data/audit/v4n_shared_mazes_20261107"; PROTOCOL = OUT / "PROTOCOL.json"
TT = M.ROOT / "data/unified/tartarus"; TRAJ = M.ROOT / "data/unified/tartarus_trajectories.json"
PRIOR_PROTOCOL = M.ROOT / "docs/WACV_NATIVE_BEHAVIOR_PRIOR_SOURCE_PROTOCOL_20261012.json"
CONFIGS = tuple(range(6, 26)); FAM = (3, 3); SEEDS = tuple(range(1000, 1030)); LPRI = ("human_bc", "human_legal_uniform", "cold_ppo")
REC = {"ppo": "data/audit/v4n_moved_goal_learning_20261031/test2/ppo/registered/{p}/seed{s}.final.pt",
       "bl3": "data/audit/v4n_brainlike3_20261105/test/bl3_none/{p}/seed{s}.final.pt",
       "bl7": "data/audit/v4n_brainlike7_20261107/test/bl7_none/{p}/seed{s}.final.pt"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def layouts():
    """{cfg: (walkable 10x10 bool, own starts)} for the Tartarus configurations, read from the unified import."""
    T = json.loads((TT / "tasks.json").read_text())["tasks"]; Z = np.load(TT / "tasks.npz"); out = {}
    for i, t in enumerate(T):
        c = int(t["id"].split("/")[1]); w = Z[f"walkable_{i}"].astype(bool)
        if c not in out: out[c] = (w, [])
        assert np.array_equal(out[c][0], w); out[c][1].append(tuple(t["spawn"]))
    return out


def rooms():
    L = layouts(); rows = []
    for c in CONFIGS:
        w, st = L[c]; d = worlds.fresh.source.goal_distances(w, FAM)
        assert len(d) == int(w.sum()) and all(s in d and d[s] <= 30 for s in st) and len(st) == 10, f"criteria fail {c}"
        rows.append(dict(configuration=200 + c, tartarus=c, pair=0, stage="A", walkable=w.astype(int).tolist(), A_goal=list(FAM), B_goal=list(FAM),
                         starts=[list(s) for s in st], moved=list(map(int, CR.moved_goal(w, M.TARGET, st))), blocked=int((~w).sum())))
    return rows


def env_for(man, goal, seed, schedule):
    _, _, source = worlds.fresh.load(); cache = worlds.RoomCache(source, man); cache.goals.update({k: tuple(v) for k, v in goal.items()})
    phys = worlds.ContinuousRooms(cache, n=1, seed=82000 + seed, phase_frames=45 * len(schedule) // 2, pair=0, schedule=schedule)
    return worlds.VisualRooms(phys, source, "clean", noise_seed=[81013, seed, 0])


def probe(agent, man, row, goal, seed):
    """F.probe with the layout's own starts (the only change)."""
    sched = [dict(configuration=row["configuration"], spawn=s, rotation_index=rot) for rot in range(3) for s in row["starts"]]
    m = __import__("copy").deepcopy(agent); device = "cuda:0" if isinstance(m, F.ProtectedAC) else "cpu"
    m.tg = torch.Generator(device=device).manual_seed(230017 + seed)
    env = env_for(man, {row["configuration"]: goal}, seed, sched); obs = env.reset()
    before = {k: p.clone() for k, p in F.tensor_parameters(m).items()}; probs, vals = [], []
    while not env.exhausted and env.frames < env.limit:
        a, p, v = F.sample(m, obs); probs.append(p[0].cpu().tolist()); vals.append(float(v[0])); obs, _, _ = env.step(a.cpu().numpy())
    assert all(torch.equal(p, before[k]) for k, p in F.tensor_parameters(m).items())
    eps = [e for e in F.study.ppo_source.controls.parent.base.episodes(env, probs, vals) if "spawn" in e]
    return [dict(spawn=list(e["spawn"]), success=bool(e["success"]), path=[list(x) for x in e["path"]]) for e in eps]


def cfg3():
    c = dict(json.loads((M.ROOT / "data/audit/v4n_brainlike3_20261105/BEST.json").read_text())["configs"]["none"])
    c["eta_v"] = c["rho"] * (1 - M.GAMMA * c["lam"]); return c


def agent(kind, prior, seed):
    """The paper's networks: frozen imitation networks, or the test-phase state after learning the moved goal."""
    if kind == "frozen": return M.build(prior, seed)
    f = M.ROOT / REC[kind].format(p=prior, s=seed); sd = torch.load(f, map_location="cpu", weights_only=False)
    if kind == "ppo": m = M.build(prior, seed)
    elif kind == "bl3": B.KC = KC0; m = B.build(prior, seed, cfg3())
    else:
        import v4n_brainlike7_claude_20261107 as B7
        B.KC = B7.KC7; m = B7.build7(prior, seed, json.loads((B7.OUT / "BEST.json").read_text())["configs"]["none"])
    dev = next(m.net.parameters()).device; m.net.load_state_dict({k: v.to(dev) for k, v in sd.items()})
    if kind == "bl7":                                    # goal memory G, rebuilt exactly from its peaks (max write) and checked
        rec = json.loads(f.with_name(f.name.replace(".final.pt", ".json")).read_text()); c = rec["cfg"]; g = np.asarray(rec["goal_memory"]); Gm = np.zeros(100)
        for x in np.flatnonzero(np.isclose(g, round(c["eta_g"], 4))): Gm = np.maximum(Gm, c["eta_g"] * B7.kern(int(x), c["sigma"]))
        assert np.allclose(Gm.round(4), g, atol=1e-4); m.net.pi.hippo.G = Gm
    return m


def jobs():
    J = [("frozen", p, s) for p in ("human_bc", "rat_bc") for s in SEEDS]
    return J + [(k, p, s) for k in ("ppo", "bl3", "bl7") for p in LPRI for s in SEEDS]


def run_one(job, rows=None, dry=False):
    kind, prior, seed = job; out = OUT / ("dry" if dry else "runs") / kind / prior / f"seed{seed}.json"
    if out.exists(): return json.loads(out.read_text())
    torch.set_num_threads(1); G.CTX.update(c=0, mode="G0"); rows = rows or rooms(); rows = rows[:2] if dry else rows
    man = dict(starts=rows[0]["starts"], rows=rows); m = agent(kind, prior, seed); t0 = time.perf_counter(); res = {}
    for r in rows:
        for gname, goal in (("familiar", FAM), ("moved", tuple(r["moved"]))):
            eps = probe(m, man, r, goal, seed + r["tartarus"] + (0 if gname == "familiar" else 100))
            res[f"{r['tartarus']}/{gname}"] = dict(goals=sum(e["success"] for e in eps), paths=[e["path"] for e in eps] if gname == "familiar" else None)
    rec = dict(kind=kind, prior=prior, seed=seed, res=res, wall_s=time.perf_counter() - t0, protocol_sha256=None if dry else sha(PROTOCOL))
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec)); return rec


# ---------- geometry ----------
def metrics(paths, w, starts_d):
    """paths: lists of cells from a start; success = reaches (3,3) within 45 moves (truncated); revisit; excess; occupancy."""
    succ, rev, exc, occ = [], [], [], np.zeros(100)
    for p in paths:
        p = [tuple(x) for x in p if w[tuple(x)]][:46]                       # ADDENDUM1: drop off-maze tracking cells (rats, 0.1%)
        if not p: continue
        hit = FAM in p; q = p[:p.index(FAM) + 1] if hit else p; succ.append(hit)
        seen = {q[0]}; r = 0
        for x in q[1:]: r += x in seen; seen.add(x)
        rev.append(r / max(len(q) - 1, 1))
        if hit and starts_d.get(q[0]): exc.append((len(q) - 1) / starts_d[q[0]])  # ADDENDUM1: skip starts with no distance
        for x in q: occ[x[0] * 10 + x[1]] += 1
    return dict(success=float(np.mean(succ)), revisit=float(np.mean(rev)), excess=float(np.mean(exc)) if exc else None, occ=occ / max(occ.sum(), 1))


def bc(a, b): return float(np.sum(np.sqrt(a * b)))


def animals(rows):
    T = json.loads(TRAJ.read_text())["trials"]; by = collections.defaultdict(list)
    for t in T:
        if t["configuration"] in CONFIGS: by[(t["species"], t["configuration"])].append(t["cells"])
    return by


def check():
    rows = rooms(); P = json.loads(PRIOR_PROTOCOL.read_text()); assert P["native_configurations"] == [1, 2, 3, 4, 5]
    assert not set(CONFIGS) & set(P["native_configurations"])
    A = animals(rows); n = {k: len(v) for k, v in A.items()}
    assert all(n.get(("human", c), 0) > 0 and n.get(("rat", c), 0) > 0 for c in CONFIGS)
    for job in (("bl7", "human_bc", 1000), ("frozen", "rat_bc", 1000)): r = run_one(job, rows, dry=True); print(job, {k: v["goals"] for k, v in r["res"].items()}, round(r["wall_s"], 1), "s")
    print(json.dumps(dict(rooms=len(rows), blocked=[r["blocked"] for r in rows], moved=[r["moved"] for r in rows],
                          human_trials=sum(n[("human", c)] for c in CONFIGS), rat_trials=sum(n[("rat", c)] for c in CONFIGS))))


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=sha(__file__),
        rooms=f"Tartarus configurations {CONFIGS[0]}-{CONFIGS[-1]} (no network saw them; imitation used 1-5), own native starts; layouts hash "
              + hashlib.sha256(json.dumps(rooms(), sort_keys=True).encode()).hexdigest(),
        criteria="10x10, familiar goal (3,3), connected, every start <= 30 steps (all pass); moved goal = paper's rule with own starts",
        agents="frozen human_bc, rat_bc; learned ppo, bl3_none, bl7_none from human_bc / human_legal_uniform / cold_ppo; 30 seeds; no training",
        probe="F.probe, own 10 starts x 3 rotations = 30 trials per layout and goal, weights verified frozen",
        outcomes="descriptive: success per goal; familiar goal vs recorded humans and rats on the same layouts and starts: success within 45, "
                 "revisit rate, excess path, occupancy Bhattacharyya coefficient per layout (to humans, to rats; human-vs-rat reference)",
        role="addition to the camera-ready; the paper's registered claims stay on the 16 generated rooms",
        not_included="stage-7 gated and ablation arms (compute); moved-goal animal data do not exist",
        outcomes_seen="check (dry, timing) before this file, layouts 6-7, seed 1000: bl7/human_bc familiar 10/30, 15/30, moved 24/30, 16/30; frozen rat_bc familiar 9/30, 10/30, moved 0, 0",
        data="layouts and animal paths are read from data/unified at run time; outputs hold agent paths and aggregate animal metrics only"), indent=1) + "\n")
    print(PROTOCOL.read_text()[:400])


def run(w, n):
    assert PROTOCOL.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    J = jobs()[w::n]; rows = rooms()
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 12))) as ex:
        for i, r in enumerate(ex.map(run_one, J, [rows] * len(J))): print(i + 1, len(J), r["kind"], r["prior"], r["seed"], round(r["wall_s"]), flush=True)


def ci(v):
    v = np.asarray(v, float); from scipy import stats
    return [round(float(v.mean()), 2), round(float(stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))), 2) if len(v) > 1 else None, len(v)]


def analyse():
    rows = rooms(); A = animals(rows); W = {r["tartarus"]: (np.asarray(r["walkable"], bool), r["starts"]) for r in rows}
    dist = {c: worlds.fresh.source.goal_distances(w, FAM) for c, (w, _) in W.items()}
    ani = {sp: {c: metrics(A[(sp, c)], W[c][0], dist[c]) for c in CONFIGS} for sp in ("human", "rat")}
    R = [json.loads(f.read_text()) for f in sorted((OUT / "runs").glob("*/*/seed*.json"))]; groups = collections.defaultdict(list)
    for r in R: groups[(r["kind"], r["prior"])].append(r)
    out = dict(animals={sp: dict(success=ci([ani[sp][c]["success"] * 100 for c in CONFIGS]), revisit=ci([ani[sp][c]["revisit"] * 100 for c in CONFIGS]),
                                 excess=ci([ani[sp][c]["excess"] for c in CONFIGS if ani[sp][c]["excess"]])) for sp in ani},
               human_vs_rat_occupancy=ci([bc(ani["human"][c]["occ"], ani["rat"][c]["occ"]) for c in CONFIGS]), agents={})
    for (k, p), rs in sorted(groups.items()):
        fam = [np.mean([r["res"][f"{c}/familiar"]["goals"] for c in CONFIGS]) * 100 / 30 for r in rs]
        mov = [np.mean([r["res"][f"{c}/moved"]["goals"] for c in CONFIGS]) * 100 / 30 for r in rs]
        geo = collections.defaultdict(list)
        for r in rs:
            g = {c: metrics(r["res"][f"{c}/familiar"]["paths"], W[c][0], dist[c]) for c in CONFIGS}
            geo["revisit"].append(np.mean([g[c]["revisit"] for c in CONFIGS]) * 100)
            geo["excess"].append(np.nanmean([g[c]["excess"] if g[c]["excess"] else np.nan for c in CONFIGS]))
            for sp in ("human", "rat"): geo[f"occ_{sp}"].append(np.mean([bc(g[c]["occ"], ani[sp][c]["occ"]) for c in CONFIGS]))
        out["agents"][f"{k}/{p}"] = dict(familiar=ci(fam), moved=ci(mov), **{q: ci([x for x in v if np.isfinite(x)]) for q, v in geo.items()})
    out["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); out["records"] = len(R)
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check": check()
    elif cmd == "prereg": prereg()
    elif cmd == "run": run(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "analyse": analyse()
