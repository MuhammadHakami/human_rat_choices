"""Stability-plasticity search for biologically plausible learners (camera-ready, author 2026-11-02: optimise the
bio-plausible learners). GPU 1. v4n_local_search2 optimised the new goal only and its winners forget the old goal
(familiar goal, trained room: 2-4% vs 19-32% PPO). Two local mechanisms are added to the search2 space:
  consolidation  each synapse is pulled toward its value before learning, q += -rate * anchor * (q - q0)
                 (a two-component synapse: fast weight + slow consolidated weight; Benna & Fusi 2016; local)
  old replay     before the switch the frozen prior acts in the familiar goal for F0 frames (counted inside the same
                 32,768-interaction budget); during learning each new chunk is followed by a replayed old chunk with
                 weight beta (complementary learning systems; McClelland, McNaughton & O'Reilly 1995)
Multi-objective NSGA-II over (moved-goal final, familiar-goal final) on the dev rooms (3 dev seeds x 3 priors).
Selection (fixed here): per family (dfa, ed, head) the trial with the largest dev sum moved + familiar; each tested once on
30 seeds x 3 priors in the fresh rooms + the three KPI probes. Primary contrasts as search2 (prior AUC, Holm over 6).
  check | prereg | search W TRIALS | best | test W N | kpi W N | analyse
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_local_search2_claude_20261102 as S2
M = S2.M

OUT = M.ROOT / "data/audit/v4n_local_stability_20261102"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"
FAM = (3, 3); NAMES = {"dfa": "cs_dfa", "ed": "cs_ed", "head": "cs_head"}


class Consolidating(S2.ReplayLocal):
    def __init__(self, net, cfg, seed):
        super().__init__(net, cfg, seed); self.q0 = [q.detach().clone() for q in self.params]

    @torch.no_grad()
    def learn(self, buf, v_last, frac_done, weight=1.):
        c0 = self.c; rate = self.c["rate"]
        if weight != 1.: self.c = dict(self.c, rate=rate * weight)
        try: super().learn(buf, v_last, frac_done)
        finally: self.c = c0
        if self.c["anchor"] > 0:
            for q, q0 in zip(self.params, self.q0): q -= rate * self.c["anchor"] * (q - q0)


def collect(L, env, obs, gen, T, frames):
    """Frozen-prior experience in the familiar goal, chunked as the learner's buffers."""
    chunks, buf, n = [], [], 0
    while n < frames:
        a, p, v = L.act(obs, gen); o0 = torch.as_tensor(obs, device=L.dev); obs, r, done = env.step(a.numpy()); n += M.N_ENV
        buf.append((o0, a.to(L.dev), torch.as_tensor(r, device=L.dev, dtype=torch.float32), torch.as_tensor(done, device=L.dev, dtype=torch.float32),
                    torch.log(p.gather(1, a.to(L.dev)[:, None]).squeeze(1).clamp_min(1e-8))))
        if len(buf) == T: chunks.append((buf, L.forward(obs)[3])); buf = []
    return chunks


def run_cfg(args):
    cfg, prior, seed, manifest, out = args
    import v4n_kpi_claude_20261031 as K
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1"); torch.set_num_threads(1); manifest, out = Path(manifest), Path(out)
    if out.exists(): return json.loads(out.read_text())
    pair, stage = M.room_of(seed); gm = M.goal_map(manifest); fam = {c: FAM for c in gm}; m = M.build(prior, seed); t0 = time.perf_counter(); diverged = None
    if cfg["expand"]: S2.expand(m.net, cfg["expand"], cfg["kwta"], seed)
    L = Consolidating(m.net, cfg, seed); gen = torch.Generator().manual_seed(77 + seed)
    old = []
    if cfg["old_frames"]:
        envo = M.env_for(manifest, fam, pair, M.N_ENV, seed + 500, "clean", cfg["old_frames"]); old = collect(L, envo, envo.reset(), gen, cfg["T"], cfg["old_frames"])
    budget = M.BUDGET - cfg["old_frames"]; rng = np.random.default_rng(seed)
    env = M.env_for(manifest, gm, pair, M.N_ENV, seed, "clean", budget); obs = env.reset()
    curve = {0: M.probe_now(m, manifest, gm, pair, stage, seed)}; frames = 0; buf = []
    while frames < budget:
        a, p, v = L.act(obs, gen); o0 = torch.as_tensor(obs, device=L.dev); obs, r, done = env.step(a.numpy()); frames += M.N_ENV
        buf.append((o0, a.to(L.dev), torch.as_tensor(r, device=L.dev, dtype=torch.float32), torch.as_tensor(done, device=L.dev, dtype=torch.float32),
                    torch.log(p.gather(1, a.to(L.dev)[:, None]).squeeze(1).clamp_min(1e-8))))
        if len(buf) == cfg["T"] or frames >= budget:
            L.learn(buf, L.forward(obs)[3], frames / budget); buf = []
            if old and cfg["beta"] > 0: ob, vl = old[rng.integers(len(old))]; L.learn(ob, vl, frames / budget, cfg["beta"])
            if not all(torch.isfinite(q).all() for q in m.net.parameters()): diverged = frames; break
        f = frames + cfg["old_frames"]                                       # curve indexed by total interactions
        if f in M.CURVE and f > 0: curve[f] = M.probe_now(m, manifest, gm, pair, stage, seed)
    for x in M.CURVE: curve.setdefault(x, curve[0] if x <= cfg["old_frames"] else dict(goals=0, episodes=[]))   # old phase: prior unchanged
    rec = dict(learner="stability", cfg=cfg, prior=prior, seed=seed, pair=pair, room=stage, goal=list(gm[100 + 2 * pair]), budget=M.BUDGET,
               curve={k: v["goals"] for k, v in curve.items()}, familiar_trained_room=K.probe(m, fam, pair, stage, seed) if diverged is None else 0,
               diverged_at=diverged, final_episodes=curve[M.BUDGET]["episodes"], wall_s=time.perf_counter() - t0,
               protocol_sha256=M.sha(PROTOCOL), gpu_name=torch.cuda.get_device_name(0))
    out.parent.mkdir(parents=True, exist_ok=True); torch.save({k: v.cpu() for k, v in m.net.state_dict().items()}, out.with_suffix(".final.pt"))
    out.write_text(json.dumps(rec) + "\n"); return rec


def objective(trial):
    s = trial.suggest_categorical; T = s("T", [8, 32, 128])
    cfg = dict(plastic=s("plastic", ["head", "last", "all"]), feedback=s("feedback", list(S2.FAMILIES)), expand=s("expand", [0, 512, 2048]),
               kwta=trial.suggest_float("kwta", .02, .3, log=True), T=T, epochs=s("epochs", [1, 2, 4, 8]), minibatches=s("minibatches", [1, 4]),
               clip=s("clip", [0., .1, .2, .3]), gae=trial.suggest_float("gae", .5, .99), ent=trial.suggest_float("ent", 0., .1),
               critic_ratio=trial.suggest_float("critic_ratio", .3, 3., log=True), fb_scale=trial.suggest_float("fb_scale", .1, 3., log=True),
               rate=trial.suggest_float("rate", 1e-5, 1e-2, log=True), adv_norm=s("adv_norm", [True, False]), decay=s("decay", [True, False]),
               anchor=s("anchor", [0., 1e-3, 1e-2, 3e-2, .1, .3, 1.]), old_frames=s("old_frames", [0, 4096, 8192]), beta=s("beta", [0., .25, .5, 1.]))
    jobs = [(cfg, p, sd, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}" / p / f"seed{sd}.json")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = list(ex.map(run_cfg, jobs))
    new = float(np.mean([{str(k): v for k, v in r["curve"].items()}[str(M.BUDGET)] for r in recs])); old = float(np.mean([r["familiar_trained_room"] for r in recs]))
    trial.set_user_attr("sum", new + old); return new, old


def search(w, trials):
    import optuna
    st = optuna.create_study(study_name="v4n_stability", storage=DB, directions=["maximize", "maximize"], load_if_exists=True,
                             sampler=optuna.samplers.NSGAIISampler(seed=20261102 + w, population_size=24))
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_stability", storage=DB); done = [t for t in st.trials if t.values is not None]
    fam = {"dfa": lambda t: t.params["plastic"] != "head" and t.params["feedback"] == "dfa",
           "ed": lambda t: t.params["plastic"] != "head" and t.params["feedback"] == "ed", "head": lambda t: t.params["plastic"] == "head"}
    pick = {k: max((t for t in done if f(t)), key=lambda t: sum(t.values)) for k, f in fam.items() if any(f(t) for t in done)}
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done),
        configs={k: t.params for k, t in pick.items()}, trial_numbers={k: t.number for k, t in pick.items()},
        dev_values={k: t.values for k, t in pick.items()}, pareto=[dict(number=t.number, values=t.values) for t in st.best_trials]), indent=1) + "\n")
    print(json.dumps({k: (t.number, t.values) for k, t in pick.items()}))


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"; pick = json.loads((OUT / "BEST.json").read_text())["configs"]
    jobs = [(name, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for name, cfg in sorted(pick.items())][w::n]
    for name, cfg, p, s in jobs: run_cfg((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / NAMES[name] / p / f"seed{s}.json")))
    print(json.dumps(dict(worker=w, done=len(jobs))), flush=True)


def kpi(w, n):
    import v4n_kpi_claude_20261031 as K
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    for f in sorted((OUT / "test").glob("*/*/seed*.json"))[w::n]:
        r = json.loads(f.read_text()); out = K.OUT / "kpi" / f.parts[-3] / r["prior"] / f.name
        if out.exists(): continue
        m = M.build(r["prior"], r["seed"])
        if r["cfg"]["expand"]: S2.expand(m.net, r["cfg"]["expand"], r["cfg"]["kwta"], r["seed"])
        m.net.load_state_dict({k: v.to(next(m.net.parameters()).device) for k, v in torch.load(f.with_suffix(".final.pt")).items()})
        pair = r["seed"] % 8; mg = M.goal_map(M.FR.MAN); fam = {c: FAM for c in mg}
        rec = dict(learner=f.parts[-3], prior=r["prior"], seed=r["seed"], source=str(f), familiar_trained_room=K.probe(m, fam, pair, "A", r["seed"]),
                   moved_heldout_room=K.probe(m, mg, pair, "B", r["seed"]), familiar_heldout_room=K.probe(m, fam, pair, "B", r["seed"]))
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec) + "\n")
    print(json.dumps(dict(stage="kpi", worker=w, done=True)), flush=True)


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        outcomes_seen="search2 test and KPI outcomes (winners forget the old goal); no outcome of this study",
        script_sha256=M.sha(__file__), design=__doc__, trials="144 NSGA-II trials (2 workers, population 24), dev only",
        primary="per selected learner: human_bc - human_legal_uniform and human_bc - cold_ppo AUC; Holm over the 6",
        secondary="moved + familiar (trained room) vs PPO, vs the search2 learner of the same family, and vs the joint-training reference; KPI probes",
        caveat="old replay uses the agent's own frozen-prior experience of the familiar goal, inside the same interaction budget; no task cue"), indent=1) + "\n")
    print("registered")


def check():
    net = torch.nn.Module(); net.body = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8), torch.nn.Tanh())
    net.pi = torch.nn.Linear(8, 4); net.v = torch.nn.Linear(8, 1)
    cfg = dict(plastic="all", feedback="dfa", rate=1e-2, gae=.9, ent=.01, critic_ratio=1., fb_scale=1., epochs=2, minibatches=2, clip=.2,
               adv_norm=True, decay=False, T=4, anchor=1e9)
    L = Consolidating(net, cfg, 0); gen = torch.Generator().manual_seed(0); buf = []
    for t in range(4):
        o = torch.randn(3, 5); a, p, v = L.act(o, gen); buf.append((o, a, torch.tensor([1., 0, 0]), torch.tensor([0., 0, 0]), torch.log(p.gather(1, a[:, None]).squeeze(1))))
    L.c["anchor"] = 1 / cfg["rate"]; L.learn(buf, torch.zeros(3), .1)                 # rate x anchor = 1: snaps back to q0
    assert all(torch.allclose(q, q0) for q, q0 in zip(L.params, L.q0))
    L.c["anchor"] = 0.; L.learn(buf, torch.zeros(3), .1, weight=.5); assert any(not torch.equal(q, q0) for q, q0 in zip(L.params, L.q0)) and L.c["rate"] == 1e-2
    print("check ok")


def analyse():
    """Registered primary (prior AUC, Holm over 6) + secondary paired contrasts vs PPO and the same-family search2 learner."""
    from prior_seed_power_claude import contrast
    from analyse_seeds30_claude import holm
    import v4n_kpi_claude_20261031 as K
    xs = np.array(M.CURVE, float); auc = lambda c: float(np.trapezoid([c[str(int(x))] for x in xs], xs) / (30 * M.BUDGET)) * 100
    def load(name):                                                       # {metric: {(prior, seed): %}}
        S = {"auc": {}, "new": {}, "old": {}, "ood_new": {}, "ood_old": {}}
        for f in (K.OUT / "kpi" / name).glob("*/seed*.json"):
            r = json.loads(f.read_text()); k = (r["prior"], r["seed"]); c = json.loads(Path(r["source"]).read_text())["curve"]
            S["auc"][k] = auc(c); S["new"][k] = c[str(M.BUDGET)] * 100 / 30
            for q, kk in (("old", "familiar_trained_room"), ("ood_new", "moved_heldout_room"), ("ood_old", "familiar_heldout_room")): S[q][k] = r[kk] * 100 / 30
        return S
    res, fam = {}, {}
    for fam_, name in NAMES.items():
        S = load(name); R = dict(means={q: float(np.mean(list(v.values()))) for q, v in S.items()}, n=len(S["auc"]))
        by = lambda q: {p: {k[1]: v for k, v in S[q].items() if k[0] == p} for p in M.PRIORS}
        for b in ("human_legal_uniform", "cold_ppo"): fam[f"{name}/{b}"] = R[f"auc/human_bc-{b}"] = contrast(by("auc"), "human_bc", b, 10.)
        for other in ("ppo", f"s2_{fam_}" if fam_ != "head" else "s2_head_strat"):
            O = load(other)
            for q in ("new", "old", "ood_new", "ood_old"):
                keys = sorted(set(S[q]) & set(O[q])); R[f"{q}/{name}-{other}"] = contrast({"a": {k: S[q][k] for k in keys}, "b": {k: O[q][k] for k in keys}}, "a", "b", 10.)
        res[name] = R
    for k, h in zip(fam, holm([v["t_p"] for v in fam.values()])): fam[k]["holm_t_p"] = h
    res["primary_auc"] = fam; (OUT / "analysis.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    for name in NAMES.values():
        R = res[name]; print(name, R["n"], {q: round(x, 1) for q, x in R["means"].items()})
        for q, v in R.items():
            if "/" in q: print("  ", q, round(v["mean"], 1), [round(x, 1) for x in v["ci95"]], "p=%.2g" % v["t_p"], ("holm=%.2g" % v["holm_t_p"]) if "holm_t_p" in v else "")


if __name__ == "__main__":
    c = sys.argv[1]; torch.set_num_threads(1)
    {"check": check, "prereg": prereg, "best": best, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])),
     "test": lambda: test(int(sys.argv[2]), int(sys.argv[3])), "kpi": lambda: kpi(int(sys.argv[2]), int(sys.argv[3])), "analyse": analyse}[c]()
