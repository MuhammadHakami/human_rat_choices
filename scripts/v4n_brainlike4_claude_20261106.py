"""Brain-constrained circuit, stage 4 (camera-ready, author 2026-11-06: "continue but use GPU id 1 only"). GPU 1 only.
Registered before any outcome of this stage.
Seen before writing: stage 3 test (v4n_brainlike3_20261105): ungated new goal 73.2 (+12.4 vs PPO), old 27.9 (TOST eq.),
new goal in the UNSEEN room 20.4 (-7.4 [-13.4, -1.5] vs PPO). Stage-2/3 dev searches: unseen-room new goal correlates
.91-.96 with trained-room new goal; neither objective included it; old-goal sleep replay (O) was rejected.
Stage 4 = stage 3 with O = 0 + one mechanism aimed at transfer of the new goal to an unseen room:
  lateral-horn pathway  a dense, non-expanded copy of the cortical input h (unit-normalised, gain w) reaches the same
                output neurons alongside the sparse KC code, with the same three-factor rule. In insects the lateral
                horn runs parallel to the mushroom body and carries generalising, non-sparse odour information to shared
                output/premotor neurons (Jefferis et al. 2007; Dolan et al. 2019; Eschbach et al. 2020). Sparse codes
                separate patterns (good for same-room learning), dense codes generalise. w = 0 is stage 3 (O = 0).
Lean (proofs/Proofs/Trace.lean): with |z| <= sqrt(1 + w^2) the trace bound holds with M = sqrt(1 + w^2); the value rate
stays searched in units of (1 - gamma lambda).
postNAS: K = 2048, c = 7, O = 0 fixed; searched as stage 3 (f, eta, tau, lam, rho, m, R, E, T, A, gate) + w {0, .25, .5, 1}.
Objectives (NSGA-II, 3): dev new goal, dev old goal, dev new goal in the unseen room (dev seeds 1000-1002 x 3 priors).
The stage-3 winners (w = 0) are enqueued first as references. 2 workers, both on GPU 1, 60 trials each.
Selection (fixed here): largest dev (new + old + unseen-room new), per gate. Test once on GPU 1 (30 seeds x 3 priors,
fresh rooms, four probes). Contrasts (descriptive): stage 4 - PPO, - stage 3.
  check | prereg | search W TRIALS | best | test W N | analyse
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_brainlike_claude_20261103 as B
import v4n_brainlike2_claude_20261105 as B2
import v4n_brainlike3_claude_20261105 as B3
M = B.M

OUT = M.ROOT / "data/audit/v4n_brainlike4_20261106"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"
_KC = B.KC
P4 = ("moved_trained_room", "familiar_trained_room", "moved_heldout_room", "familiar_heldout_room")


class KC4(_KC):
    """Sparse KC code plus a dense lateral-horn copy of h (gain w) on the same output synapses."""
    def __init__(self, base, H, cfg, seed, gated):
        super().__init__(base, H, cfg, seed, gated); self.w = float(cfg.get("w", 0.))
        self.R = torch.nn.Linear(cfg["K"] + H, base.out_features, bias=False); torch.nn.init.zeros_(self.R.weight)

    def code(self, h):
        z = super().code(h); return torch.cat([z, self.w * h / (h.norm(dim=-1, keepdim=True) + 1e-6)], -1)


def run_cfg(args):
    cfg = dict(args[0], O=0); cfg["eta_v"] = cfg["rho"] * (1 - M.GAMMA * cfg["lam"])
    B.KC, B.Circuit, B.PROTOCOL = KC4, B2.Circuit2, PROTOCOL; return B.run_cfg((cfg,) + tuple(args[1:]))


def objective(trial):
    s = trial.suggest_categorical; fl = trial.suggest_float
    cfg = dict(K=2048, c=7, f=fl("f", .02, .12, log=True), eta=fl("eta", 3., 200., log=True), tau=fl("tau", 1.5, 5., log=True),
               lam=fl("lam", .8, .99), rho=fl("rho", .1, 4., log=True), m=s("m", [0., 1., 10.]), R=s("R", [0, 1]), E=s("E", [1, 2, 4, 8]),
               T=s("T", [8, 16, 32]), A=s("A", [0, 1]), w=s("w", [0., .25, .5, 1.]), gate=s("gate", ["none", "inferred"]))
    jobs = [(cfg, p, sd, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}" / p / f"seed{sd}.json")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = list(ex.map(run_cfg, jobs))
    v = [float(np.mean([r[q] for r in recs])) for q in P4[:3]]
    trial.set_user_attr("sum", sum(v)); trial.set_user_attr("gpu", recs[0]["gpu_name"]); return tuple(v)


def search(w, trials):
    import optuna
    assert PROTOCOL.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    st = optuna.create_study(study_name="v4n_brainlike4", storage=DB, directions=["maximize"] * 3, load_if_exists=True,
                             sampler=optuna.samplers.NSGAIISampler(seed=20261107 + w, population_size=24))
    if w == 0 and not st.trials:
        for g, c3 in json.loads((B3.OUT / "BEST.json").read_text())["configs"].items():
            st.enqueue_trial({k: c3[k] for k in ("f", "eta", "tau", "lam", "rho", "m", "R", "E", "T", "A", "gate")} | dict(w=0.))
    else: time.sleep(5)
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_brainlike4", storage=DB); done = [t for t in st.trials if t.values is not None]
    pick = {g: max((t for t in done if t.params["gate"] == g), key=lambda t: sum(t.values)) for g in ("none", "inferred") if any(t.params["gate"] == g for t in done)}
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done),
        configs={g: dict(t.params, K=2048, c=7) for g, t in pick.items()}, numbers={g: t.number for g, t in pick.items()},
        dev_values={g: t.values for g, t in pick.items()}, pareto=[dict(n=t.number, v=t.values, p=t.params) for t in st.best_trials]), indent=1) + "\n")
    print(json.dumps({g: (t.number, [round(x * 100 / 30, 1) for x in t.values], t.params) for g, t in pick.items()}))


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"; pick = json.loads((OUT / "BEST.json").read_text())["configs"]
    for g, cfg, p, s in [(g, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for g, cfg in sorted(pick.items())][w::n]:
        run_cfg((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / f"bl4_{g}" / p / f"seed{s}.json")))
    print(json.dumps(dict(worker=w, done=True)), flush=True)


def analyse():
    import v4n_kpi_claude_20261031 as K
    from prior_seed_power_claude import contrast
    own = lambda d: {(r["prior"], r["seed"]): r for r in (json.loads(f.read_text()) for f in d.glob("*/seed*.json"))}
    ppo = {}
    for f in (K.OUT / "kpi/ppo").glob("*/seed*.json"):
        r = json.loads(f.read_text()); c = json.loads(Path(r["source"]).read_text())["curve"]; ppo[(r["prior"], r["seed"])] = dict(r, moved_trained_room=c[str(M.BUDGET)])
    res = {}
    for g in ("none", "inferred"):
        S = own(OUT / "test" / f"bl4_{g}"); refs = {"ppo": ppo, "bl3": own(B3.OUT / "test" / f"bl3_{g}")}
        R = dict(n=len(S), means={q: float(np.mean([r[q] for r in S.values()]) * 100 / 30) for q in P4}, diverged=sum(r["diverged"] is not None for r in S.values()))
        for o, O in refs.items():
            for q in P4:
                ks = sorted(set(S) & set(O)); R[f"{q}/{o}"] = contrast({"a": {k: S[k][q] * 100 / 30 for k in ks}, "b": {k: O[k][q] * 100 / 30 for k in ks}}, "a", "b", 10.)
        for b in ("human_legal_uniform", "cold_ppo"):
            for q in P4[:3]: R[f"prior/{q}/human_bc-{b}"] = contrast({p: {k[1]: S[k][q] * 100 / 30 for k in S if k[0] == p} for p in ("human_bc", b)}, "human_bc", b, 10.)
        res[g] = R; print(g, R["n"], {q[:14]: round(v, 1) for q, v in R["means"].items()}, "diverged", R["diverged"])
        for q, v in R.items():
            if "/" in q: print("  ", q, round(v["mean"], 1), [round(x, 1) for x in v["ci95"]], "p=%.2g tost=%.2g" % (v["t_p"], v["tost_p"]))
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=M.sha(__file__), design=__doc__,
        outcomes_seen="stages 1-3 dev and test (incl. unseen-room probes); none of stage 4", dev_seeds=list(M.DEV_SEEDS),
        test_seeds=list(M.FR.S30), priors=M.PRIORS), indent=1) + "\n"); print("registered")


def check():
    """w = 0 reproduces the stage-3 winner's dev record exactly (one prior x seed, same GPU model)."""
    import tempfile
    c3 = json.loads((B3.OUT / "BEST.json").read_text()); cfg = dict(c3["configs"]["none"], w=0.); n3 = c3["numbers"]["none"]
    ref = json.loads((B3.OUT / "search" / f"trial{n3}" / "cold_ppo" / "seed1000.json").read_text())
    global PROTOCOL
    P0, PROTOCOL = PROTOCOL, Path(tempfile.mkdtemp()) / "P.json"; PROTOCOL.write_text("{}")
    rec = run_cfg((cfg, "cold_ppo", 1000, str(M.FR.EXPOSED), str(PROTOCOL.parent / "r.json"))); PROTOCOL = P0
    assert all(rec[q] == ref[q] for q in P4), ([rec[q] for q in P4], [ref[q] for q in P4])
    print("check ok", [rec[q] for q in P4])


if __name__ == "__main__":
    {"check": check, "prereg": prereg, "best": best, "analyse": analyse, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])),
     "test": lambda: test(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
