"""E2 fresh-room confirmation (PROTOCOL.json + ADDENDUM1.json in data/audit/wacv_fresh_rooms_confirm_claude).
Frozen, already-audited checkpoints probed in 16 NEW rooms (registered generator rules, seed 20261014) with the source
goal (3,3). The probe is the unchanged E2 probe (wacv_continual_factorial_claude.probe, clean, 30 trials/room); only
the room cache is replaced. Camera-only checkpoints get their serialized input mask re-attached exactly as in
input_ablation_claude.ood. No learning, no room filtering after generation, no retraining.
  python scripts/fresh_rooms_confirm_claude.py check | generate | dry FAMILY | run FAMILY W N | analyse
Families: main (seeds30, 6 arms, GPU1 as in E2), camera (input study no_posterior, GPU seed%2), expert (human_bc full vs
T-calibrated all-optimal teacher, GPU seed%2).
"""
import gzip, hashlib, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wacv_ood_zero_shot_claude as O
import input_ablation_claude as IA
import alloptimal_expert_claude as AO

F = O.F; worlds = F.worlds; ROOT = F.ROOT
D = ROOT / "data/audit/wacv_fresh_rooms_confirm_claude"; MAN = D / "fresh_geometry_manifest.json"
EXPOSED = worlds.fresh.BASE / "geometry_manifest.json"
SEED, GOAL, BLOCKED, MAXD = 20261014, (3, 3), 18, 30
S30 = tuple(range(1000, 1030)); S12 = tuple(range(1200, 1210))
COHORTS = {  # family -> [(checkpoint root, case, seeds)]
    "main": [(ROOT / "data/wacv/seeds30/cohort", a, S30) for a in ("rat_bc", "rat_legal_uniform", "rat_sham", "human_bc", "human_legal_uniform", "cold_ppo")],
    "camera": [(IA.OUT / "cohort", f"{a}__no_posterior", S12) for a in ("human_bc", "human_legal_uniform", "cold_ppo")],
    "expert": [(IA.OUT / "cohort", "human_bc__full", S12), (AO.CAL, f"{AO.ARM}_T__full", S12)],
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gpu_for(family, seed): return 1 if family == "main" else seed % 2


def hashes():
    return dict(wrapper=sha(__file__), protocol=sha(D / "PROTOCOL.json"), addendum1=sha(D / "ADDENDUM1.json") if (D / "ADDENDUM1.json").exists() else None,
                manifest=sha(MAN) if MAN.exists() else None, factorial=F.binding(), zero_shot_runner=sha(O.__file__),
                input_ablation=sha(IA.__file__), alloptimal=sha(AO.__file__))


def checkpoints():
    return {f"{fam}/{c}/seed{s}": sha(root / c / f"seed{s}" / "checkpoint_source.pt") for fam, rows in COHORTS.items() for root, c, seeds in rows for s in seeds}


def rooms(rng, starts, exposed_blocked):
    """Registered rules: 18 blocked cells uniform over non-start, non-goal cells; all walkable cells connected to the goal;
    every native start within 30 steps. Deterministic in the seed; the first 16 valid rooms are kept, none filtered later."""
    reserved = set(starts) | {GOAL}; cand = [s for s in range(100) if divmod(s, 10) not in reserved]; seen = set(exposed_blocked); out = []; tries = 0
    while len(out) < 16:
        tries += 1; assert tries < 100000, "rejection budget exhausted"
        cells = sorted(map(int, rng.choice(cand, BLOCKED, replace=False)))
        if frozenset(cells) in seen: continue
        w = np.ones(100, bool); w[cells] = False; w = w.reshape(10, 10); d = worlds.fresh.source.goal_distances(w, GOAL)
        if len(d) != int(w.sum()) or not all(s in d and d[s] <= MAXD for s in starts): continue
        seen.add(frozenset(cells)); out.append((cells, w, tries))
    return out


def generate():
    ex = json.loads(EXPOSED.read_text()); starts = [tuple(s) for s in ex["starts"]]
    built = rooms(np.random.default_rng(SEED), starts, [frozenset(r["blocked"]) for r in ex["rows"]])
    rows = [dict(pair=i // 2, stage="AB"[i % 2], configuration=100 + i, blocked=c, walkable=w.astype(int).tolist(), A_goal=list(GOAL), B_goal=list(GOAL),
                 attempt=t) for i, (c, w, t) in enumerate(built)]
    m = dict(seed=SEED, starts=ex["starts"], rows=rows, exposed_manifest_sha256=sha(EXPOSED), generator_wrapper_sha256=sha(__file__),
             protocol_sha256=sha(D / "PROTOCOL.json"), addendum1_sha256=sha(D / "ADDENDUM1.json"), policy_results_seen=False)
    with MAN.open("x") as f: json.dump(m, f, indent=1); f.write("\n")
    print(json.dumps(dict(rooms=len(rows), attempts=rows[-1]["attempt"], manifest_sha256=sha(MAN))))


def env_fresh(pair, n, seed, tier, phase_frames, schedule=None):
    """Same construction as factorial.environment + zero-shot env_source_goal, with the fresh room cache."""
    _, _, source = worlds.fresh.load(); cache = worlds.RoomCache(source, json.loads(MAN.read_text()))
    assert all(g == GOAL for g in cache.goals.values()) and len(cache.worlds) == 16
    phys = worlds.ContinuousRooms(cache, n=n, seed=82000 + seed, phase_frames=phase_frames, pair=pair, schedule=schedule)
    return worlds.VisualRooms(phys, source, tier, noise_seed=[81013, seed, int(tier == "noisy")])


def probe_one(root, c, s, p, st, method):
    F.SEEDS30 = root; F.study.source_model = lambda x, c=c: F.source_checkpoint(c, x); m = F.study.build(method, s)
    hk = IA.mask_of(F.study.source_model(s))
    if hk is not None: m.net.body.register_forward_pre_hook(IA.InputMask(hk.condition))
    return F.probe(m, p, st, s, "clean"), (hk.condition if hk is not None else "full")


def run(family, worker, n, dry=False):
    gpu = int(os.environ["CUDA_VISIBLE_DEVICES"]); F.environment = env_fresh
    method = next(x for x in json.loads(F.study.MANIFEST.read_text())["methods"] if x["id"] == "ppo")
    if not dry:
        go = json.loads((D / "GO.json").read_text()); assert go["status"] == "GO" and go["hashes"] == hashes()
    jobs = [(root, c, s, p, st) for root, c, seeds in COHORTS[family] for s in seeds if gpu_for(family, s) == gpu for p in range(8) for st in "AB"]
    jobs = jobs[:1] if dry else jobs[worker::n]
    for root, c, s, p, st in jobs:
        ck = sha(root / c / f"seed{s}" / "checkpoint_source.pt")
        if not dry: assert ck == go["checkpoints"][f"{family}/{c}/seed{s}"]
        f = D / ("dry" if dry else "cohort") / family / c / f"seed{s}" / f"pair{p}_{st}.json"; full = f.with_suffix(".full.json.gz")
        if f.exists():
            old = json.loads(f.read_text()); assert old["checkpoint_sha256"] == ck and sha(full) == old["full_record_sha256"], f"invalid existing {f}"; continue
        rec, inp = probe_one(root, c, s, p, st, method); assert rec["trials"] == 30 and rec["weights_frozen_verified"]
        f.parent.mkdir(parents=True, exist_ok=True); full.write_bytes(gzip.compress(json.dumps(rec, default=float).encode()))
        f.write_text(json.dumps(dict(family=family, arm=c, input=inp, seed=s, pair=p, room=st, goal=list(GOAL), goals=rec["goals"], trials=rec["trials"],
                                     checkpoint_sha256=ck, full_record_sha256=sha(full), manifest_sha256=sha(MAN), gpu_name=torch.cuda.get_device_name(0))) + "\n")
    print(json.dumps(dict(family=family, worker=worker, done=len(jobs), dry=dry)), flush=True)


def scores(family):
    go = json.loads((D / "GO.json").read_text()); S = {}
    for root, c, seeds in COHORTS[family]:
        S[c] = {}
        for s in seeds:
            rows = [json.loads((D / "cohort" / family / c / f"seed{s}" / f"pair{p}_{st}.json").read_text()) for p in range(8) for st in "AB"]
            assert all(r["trials"] == 30 and r["checkpoint_sha256"] == go["checkpoints"][f"{family}/{c}/seed{s}"] and r["manifest_sha256"] == go["hashes"]["manifest"]
                       for r in rows)
            S[c][s] = float(np.mean([r["goals"] for r in rows]))
    return S


def analyse():
    from prior_seed_power_claude import contrast
    from analyse_seeds30_claude import holm
    go = json.loads((D / "GO.json").read_text()); assert go["status"] == "GO" and go["hashes"] == hashes()
    M, C, E = scores("main"), scores("camera"), scores("expert"); out = dict(means={k: {a: float(np.mean(list(v.values()))) for a, v in S.items()} for k, S in
                                                                                     (("main", M), ("camera", C), ("expert", E))})
    pri = dict(C1=contrast(M, "human_bc", "human_legal_uniform", 3.), C2=contrast(M, "human_bc", "cold_ppo", 3.))
    for k, h in zip(pri, holm([v["t_p"] for v in pri.values()])): pri[k]["holm_t_p"] = h
    out["primary"] = pri; out["confirmed"] = bool(all(v["mean"] > 0 and v["holm_t_p"] < .05 for v in pri.values()))
    I = dict(h={s: M["human_bc"][s] - M["human_legal_uniform"][s] for s in S30}, r={s: M["rat_bc"][s] - M["rat_legal_uniform"][s] for s in S30})
    out["secondary"] = dict(C3=contrast(M, "rat_bc", "rat_legal_uniform", 3.), C4_interaction=contrast(I, "h", "r", 3.))
    cam = dict(F1a=contrast(C, "human_bc__no_posterior", "human_legal_uniform__no_posterior", 3.), F1b=contrast(C, "human_bc__no_posterior", "cold_ppo__no_posterior", 3.))
    for k, h in zip(cam, holm([v["t_p"] for v in cam.values()])): cam[k]["holm_t_p"] = h
    out["camera_only"] = cam; out["expert"] = dict(F2=contrast(E, "human_bc__full", f"{AO.ARM}_T__full", 3.))
    (D / "analysis.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1)[:3000])


def check():
    ex = json.loads(EXPOSED.read_text()); starts = [tuple(s) for s in ex["starts"]]
    a = rooms(np.random.default_rng(1), starts, [frozenset(r["blocked"]) for r in ex["rows"]])
    b = rooms(np.random.default_rng(1), starts, [frozenset(r["blocked"]) for r in ex["rows"]])
    assert [x[0] for x in a] == [x[0] for x in b] and len(a) == 16
    exp = {frozenset(r["blocked"]) for r in ex["rows"]}; assert not exp & {frozenset(x[0]) for x in a}
    for cells, w, _ in a:
        d = worlds.fresh.source.goal_distances(w, GOAL); assert len(d) == w.sum() and max(d[s] for s in starts) <= MAXD and not w[divmod(cells[0], 10)]
    assert sum(len(v) for rows in COHORTS.values() for _, _, v in rows) == 230
    print("check ok: deterministic, connected, starts<=30, no exposed duplicate; 230 checkpoints")


if __name__ == "__main__":
    import importlib; M = importlib.import_module("fresh_rooms_confirm_claude"); torch.set_num_threads(1); c = sys.argv[1]
    if c == "check": M.check()
    elif c == "generate": M.generate()
    elif c == "dry": M.run(sys.argv[2], 0, 1, dry=True)
    elif c == "run": M.run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif c == "analyse": M.analyse()
