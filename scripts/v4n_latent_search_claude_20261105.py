"""Latent-cause inference search (camera-ready, author 2026-11-05: "unsupported claims we don't want to give up on easily;
use NAS/postNAS/Optuna to improve our gains"). GPU 1. Registered before any outcome of this search.
Seen before writing: v4n_context_gating G0/G1/G2 (familiar goal kept 4.7 / 57.5 / 42.7%; moved goal ~99.8% all).
The cue-free gate G2 loses 15 pp to the oracle G1; its runs switch 2.2x per familiar-goal probe and 1.2x per moved-goal
probe (false alarms) and end learning in the moved half in only 82% of runs: the gap is inference, not capacity.
Search (postNAS: the learner, expansion, budget, schedule and rooms are FIXED to G2; only the latent-cause rule changes):
  est     episodes before a context is established        3..30
  emin    expectation needed to be established           .05..0.5 log
  drop    hypothesised drop ratio q = e / drop            1.5..10 log
  bound   CUSUM bound                                     1..15 log
  dwell   minimum episodes after a switch                 3..40
  ref     cap on the frozen reference rate                .5..0.95
Bayesian-changepoint flavour of Gershman, Blei & Niv 2010; parameters are biological time constants of a switch circuit.
Objective: NSGA-II over (moved goal, familiar goal) in the trained room, dev rooms (FR.EXPOSED), 3 dev seeds x 3 priors.
Selection (fixed here): largest dev sum. Test once: 30 seeds x 3 priors, fresh rooms (FR.MAN), the four KPI probes.
Registered contrasts (paired by prior x seed, Holm over 4): G2s-G2 familiar, G2s-G2 moved, G2s-G0 familiar, G2s-G1 familiar.
  check | prereg | search W TRIALS | best | test W N | analyse
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_context_gating_claude_20261103 as G
M = G.M

OUT = M.ROOT / "data/audit/v4n_latent_search_20261105"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"
DEFAULT = dict(est=20, emin=.25, drop=4., bound=12., dwell=20, ref=.95)   # the registered G2 rule


def make_latent(c):
    class Latent(G.Latent):
        def update(self, success):
            k = self.c; self.n[k] += 1; self.dwell += 1
            if self.S > 0 or (self.n[k] > c["est"] and self.e[k] >= c["emin"] and self.dwell > c["dwell"]):
                if self.S == 0: self.ref = min(self.e[k], c["ref"])
                e = self.ref; q = e / c["drop"]
                self.S = max(0., self.S + (np.log(q / e) if success else np.log((1 - q) / (1 - e))))
                if self.S > c["bound"]: self.c, self.S, self.switches, self.dwell = 1 - k, 0., self.switches + 1, 0; return
            self.e[k] += (float(success) - self.e[k]) / min(self.n[k], 20)
    return Latent


def run(args):
    c, prior, seed, manifest, outdir = args
    torch.set_num_threads(1); f = Path(outdir) / "runs" / "G2" / prior / f"seed{seed}.json"
    if f.exists(): return json.loads(f.read_text())
    G.MAN, G.OUT, G.PROTOCOL, G.Latent = Path(manifest), Path(outdir), PROTOCOL, make_latent(c)
    r = G.run_one("G2", prior, seed); r["latent"] = c; f.write_text(json.dumps(r) + "\n"); return r


def objective(trial):
    s = trial.suggest_float
    c = dict(est=trial.suggest_int("est", 3, 30), emin=s("emin", .05, .5, log=True), drop=s("drop", 1.5, 10., log=True),
             bound=s("bound", 1., 15., log=True), dwell=trial.suggest_int("dwell", 3, 40), ref=s("ref", .5, .95))
    jobs = [(c, p, sd, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: R = list(ex.map(run, jobs))
    new, old = (float(np.mean([r[k] * 100 / 30 for r in R])) for k in ("moved_trained_room", "familiar_trained_room"))
    trial.set_user_attr("sum", new + old); return new, old


def search(w, trials):
    import optuna
    assert PROTOCOL.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1")
    st = optuna.create_study(study_name="v4n_latent", storage=DB, directions=["maximize", "maximize"], load_if_exists=True,
                             sampler=optuna.samplers.NSGAIISampler(seed=20261105 + w, population_size=24))
    if w == 0 and not st.trials: st.enqueue_trial(DEFAULT)                 # the registered rule is trial 0 (dev reference)
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_latent", storage=DB); done = [t for t in st.trials if t.values is not None]
    t = max(done, key=lambda t: sum(t.values))
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done), cfg=t.params,
        dev=t.values, default_dev=next((x.values for x in done if x.number == 0), None), pareto=[dict(n=x.number, v=x.values, p=x.params) for x in st.best_trials]), indent=1) + "\n")
    print(t.number, t.values, t.params)


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1"); c = json.loads((OUT / "BEST.json").read_text())["cfg"]
    for p, s in [(p, s) for s in M.FR.S30 for p in M.PRIORS][w::n]: run((c, p, s, str(M.FR.MAN), str(OUT / "test")))
    print(json.dumps(dict(worker=w, done=True)), flush=True)


def analyse():
    from analyse_seeds30_claude import holm
    from scipy import stats
    load = lambda d: {(r["prior"], r["seed"]): r for r in (json.loads(f.read_text()) for f in d.glob("*/seed*.json"))}
    R = dict(G2s=load(OUT / "test/runs/G2"), **{c: load(G.OUT / "runs" / c) for c in G.CONDS}); keys = sorted(set.intersection(*[set(v) for v in R.values()]))
    pct = lambda r, k: r[k] * 100 / 30; P = ("moved_trained_room", "familiar_trained_room", "moved_heldout_room", "familiar_heldout_room")
    res = dict(n=len(keys), means={c: {k: float(np.mean([pct(R[c][x], k) for x in keys])) for k in P} for c in R}, tests={})
    for a, b, k in (("G2s", "G2", P[1]), ("G2s", "G2", P[0]), ("G2s", "G0", P[1]), ("G2s", "G1", P[1])):
        d = np.array([pct(R[a][x], k) - pct(R[b][x], k) for x in keys]); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        res["tests"][f"{a}-{b}/{k}"] = dict(mean=float(d.mean()), ci=[float(d.mean() - h), float(d.mean() + h)], p=float(stats.ttest_1samp(d, 0).pvalue))
    for k, hp in zip(res["tests"], holm([v["p"] for v in res["tests"].values()])): res["tests"][k]["holm_p"] = hp
    res["per_prior"] = {p: {k: float(np.mean([pct(R["G2s"][x], k) for x in keys if x[0] == p])) for k in P[:2]} for p in M.PRIORS}
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res, indent=1))


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=M.sha(__file__), design=__doc__,
        outcomes_seen="v4n_context_gating G0/G1/G2 test results (fresh rooms); no outcome of this search", base_cfg=G.BASE_CFG, default=DEFAULT,
        dev_seeds=list(M.DEV_SEEDS), test_seeds=list(M.FR.S30), priors=M.PRIORS), indent=1) + "\n"); print("registered")


def check():
    a, b = G.Latent(), make_latent(DEFAULT)(); rng = np.random.default_rng(0)
    for x in rng.random(3000) < np.r_[np.full(800, .4), np.full(800, .02), np.full(1400, .9)]:
        a.update(x); b.update(x); assert (a.c, a.switches, a.S) == (b.c, b.switches, b.S)      # DEFAULT reproduces the registered G2 rule
    print("check ok")


if __name__ == "__main__":
    {"check": check, "prereg": prereg, "best": best, "analyse": analyse, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])),
     "test": lambda: test(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
