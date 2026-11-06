"""Joint-training reference for forgetting (camera-ready; reviewer 6g1M; author 2026-11-01 "absolute evidence"). GPU 1.
6g1M: the policy is feed-forward and gets no task signal, so two goals in one room give IDENTICAL inputs that need
different actions. Is the loss of the familiar goal after learning a moved goal (Fig. kpi a) sequential interference,
or a limit on doing both at once with these inputs? Reference: the same checkpoints, learners, rates, room and budget
(32,768 interactions) as v4n_moved_goal_learning test2, but episodes ALTERNATE between the moved goal and the familiar
goal (3,3) in that room (same geometry, same renders; the goal is invisible). Probed frozen after learning on both goals.
  A  aliasing, from geometry only: share of (cell) states in the 8 trained rooms whose shortest-path moves to the two
     goals have no move in common (a goal-blind memoryless policy cannot be optimal for both there).
  J1 retention: familiar goal, trained room, joint minus sequential (paired by seed; per learner, pooled over priors).
  J2 total: familiar + moved success, joint minus sequential. J2 <= 0 (not above sequential) means the sequential loss
     is mostly the constraint of simultaneous performance; J2 > 0 means interference beyond it.
  Learners ppo and td_head (registered rate from PICK2), priors M.PRIORS, seeds M.FR.S30.
  python scripts/v4n_joint_reference_claude_20261101.py check | prereg | aliasing | run W N | analyse
"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_moved_goal_learning_claude_20261031 as M
import v4n_kpi_claude_20261031 as K

OUT = M.ROOT / "data/audit/v4n_joint_reference_20261101"; PROTOCOL = OUT / "PROTOCOL.json"
LEARNERS = ("ppo", "td_head"); SHADOW = 1000; FAM = K.FAM
worlds = M.worlds


class JointRooms(worlds.ContinuousRooms):
    """Odd episodes use a shadow configuration: same walls and renders, familiar goal."""
    def _new_world(self):
        w = super()._new_world()
        if w is not None and w["episode"] % 2:
            w["configuration"] += SHADOW; assert w["spawn"] != self.cache.goals[w["configuration"]]
        return w


def joint_env(manifest, gm, pair, n, seed, tier, phase_frames, schedule=None):
    _, _, source = worlds.fresh.load(); cache = worlds.RoomCache(source, json.loads(Path(manifest).read_text())); cache.goals.update(gm)
    for c in list(cache.worlds): cache.worlds[c + SHADOW] = cache.worlds[c]; cache.goals[c + SHADOW] = FAM
    phys = JointRooms(cache, n=n, seed=82000 + seed, phase_frames=phase_frames, pair=pair, schedule=schedule)
    return worlds.VisualRooms(phys, source, tier, noise_seed=[81013, seed, int(tier == "noisy")])


def run(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and PROTOCOL.exists()
    rates = json.loads((M.OUT / "PICK2.json").read_text())["rates"]; real = M.env_for
    for l, p, s in [(l, p, s) for s in M.FR.S30 for p in M.PRIORS for l in LEARNERS][w::n]:
        f = OUT / "joint" / l / p / f"seed{s}.json"; fk = f.with_suffix(".familiar.json")
        if fk.exists(): continue
        first = [True]; cached = f.exists()
        def once(*a, **k):                                                    # only the training env is joint; probes untouched
            if first[0]: first[0] = False; return joint_env(*a, **k)
            return real(*a, **k)
        M.env_for = once
        try: r = M.run_one(l, p, s, rates.get(l), M.FR.MAN, f)
        finally: M.env_for = real
        assert cached or not first[0]                                        # a fresh fit used the joint env
        m = M.build(p, s); m.net.load_state_dict({k: v.to(next(m.net.parameters()).device) for k, v in torch.load(f.with_suffix(".final.pt")).items()})
        fam = K.probe(m, {c: FAM for c in M.goal_map(M.FR.MAN)}, s % 8, "A", s)
        fk.write_text(json.dumps(dict(learner=l, prior=p, seed=s, familiar_trained_room=fam, moved_trained_room={str(k): v for k, v in r["curve"].items()}[str(M.BUDGET)])) + "\n")
    print(json.dumps(dict(stage="run", worker=w, done=True)), flush=True)


def aliasing():
    from arc_fly.unimaze import bfs, NB
    man = json.loads(M.FR.MAN.read_text()); gm = M.goal_map(M.FR.MAN); rows = {r["configuration"]: np.asarray(r["walkable"], bool) for r in man["rows"]}
    res = {}
    for pair in range(8):
        cfg = 100 + 2 * pair; walk = rows[cfg]; d = {g: bfs(walk, g) for g in (FAM, tuple(gm[cfg]))}
        def best(c, g):                                                       # moves that start a shortest path to g
            step = {i: (c[0] + v[0], c[1] + v[1]) for i, v in enumerate(NB)}
            dist = {i: d[g][q] for i, q in step.items() if 0 <= q[0] < 10 and 0 <= q[1] < 10 and walk[q] and d[g][q] >= 0}
            return {i for i, x in dist.items() if x == min(dist.values())}
        cells = [(int(a), int(b)) for a, b in zip(*np.where(walk))]
        cells = [c for c in cells if c not in (FAM, tuple(gm[cfg])) and d[FAM][c] >= 0 and d[tuple(gm[cfg])][c] >= 0]
        res[cfg] = dict(cells=len(cells), conflict=sum(not (best(c, FAM) & best(c, tuple(gm[cfg]))) for c in cells))
    tot = sum(v["conflict"] for v in res.values()) / sum(v["cells"] for v in res.values())
    (OUT / "aliasing.json").write_text(json.dumps(dict(per_room=res, conflict_share=tot), indent=1) + "\n"); print("conflict share", tot, res)


def _paired(a, b):
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=float(d.mean()), ci=[float(d.mean() - h), float(d.mean() + h)], p=float(stats.ttest_1samp(d, 0).pvalue), n=len(d))


def analyse():
    rates = json.loads((M.OUT / "PICK2.json").read_text())["rates"]; res = dict(aliasing=json.loads((OUT / "aliasing.json").read_text())["conflict_share"])
    for l in LEARNERS:
        sub = f"rate{rates[l]:g}" if l in rates else "registered"; J, S = {}, {}
        for p in M.PRIORS:
            for s in M.FR.S30:
                j = json.loads((OUT / "joint" / l / p / f"seed{s}.familiar.json").read_text())
                seq_m = json.loads((M.OUT / "test2" / l / sub / p / f"seed{s}.json").read_text())["curve"][str(M.BUDGET)]
                seq_f = json.loads((K.OUT / "kpi" / l / p / f"seed{s}.json").read_text())["familiar_trained_room"]
                J[p, s] = (j["familiar_trained_room"] * 100 / 30, j["moved_trained_room"] * 100 / 30); S[p, s] = (seq_f * 100 / 30, seq_m * 100 / 30)
        keys = sorted(J); r = dict(J1_retention=_paired([J[k][0] for k in keys], [S[k][0] for k in keys]),
                                  J2_total=_paired([sum(J[k]) for k in keys], [sum(S[k]) for k in keys]),
                                  moved_joint_minus_seq=_paired([J[k][1] for k in keys], [S[k][1] for k in keys]), per_prior={})
        for p in M.PRIORS:
            ks = [k for k in keys if k[0] == p]
            r["per_prior"][p] = dict(joint=[float(np.mean([J[k][i] for k in ks])) for i in (0, 1)], sequential=[float(np.mean([S[k][i] for k in ks])) for i in (0, 1)])
        res[l] = r
    from analyse_seeds30_claude import holm
    fam = [(l, k) for l in LEARNERS for k in ("J1_retention", "J2_total")]
    for (l, k), h in zip(fam, holm([res[l][k]["p"] for l, k in fam])): res[l][k]["holm_p"] = h
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res, indent=1))


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), outcomes_seen=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), design=__doc__, learners=LEARNERS, priors=M.PRIORS,
        seeds=list(M.FR.S30), budget=M.BUDGET, rates=json.loads((M.OUT / "PICK2.json").read_text())["rates"],
        predictions=dict(A="a substantial share of states conflict (descriptive)", J1="joint training retains more of the familiar goal than sequential",
                         J2="familiar + moved success under joint training is not above sequential (constraint, not interference)"),
        family="J1, J2 x 2 learners, Holm", reading="J1 and J2 reported as supported or not; per-prior values descriptive"), indent=1) + "\n")
    print("registered")


def check():
    gm = M.goal_map(M.FR.MAN); env = joint_env(M.FR.MAN, gm, 0, 4, 1000, "clean", 256); obs = env.reset(); rng = np.random.default_rng(0)
    seen = {}
    while env.frames < 256:
        obs, r, d = env.step(rng.integers(4, size=4))
    for e in env.env.completed: seen.setdefault(e["configuration"], []).append(e["success"])
    assert set(seen) <= {100, 100 + SHADOW} and len(seen) == 2, seen
    c = env.cache; assert c.goals[100 + SHADOW] == FAM and c.goals[100] == tuple(gm[100]) and np.array_equal(c.worlds[100], c.worlds[100 + SHADOW])
    assert np.array_equal(c.frame(100, (0, 9), (1, 0), 0), c.frame(100 + SHADOW, (0, 9), (1, 0), 0))     # identical views
    n = {k: len(v) for k, v in seen.items()}; assert abs(n[100] - n[100 + SHADOW]) <= 4, n
    print("check ok", n)


if __name__ == "__main__":
    c = sys.argv[1]; torch.set_num_threads(1)
    {"check": check, "prereg": prereg, "aliasing": aliasing, "analyse": analyse, "run": lambda: run(int(sys.argv[2]), int(sys.argv[3]))}[c]()
