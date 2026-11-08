"""Mushroom-body circuit, stage 6 (camera-ready, author 2026-11-07: "use the sota ones and improve our bio plausible one to reach
maximum OOD generalization, minimal catastrophic forgetting, and continue learning"). GPU 1 only. Registered before any outcome
of this stage.
Seen before writing: stages 1-5 dev and test (stage 3 test: unseen-room new goal 20.4%, -7.4 pp vs PPO; stage 4 dense pathway on
the backbone hidden layer rejected; stage 5 place drive INTO the Kenyon cells: ungated winner beta = 0, gated winner beta = 1 fell
to 11.1% on the unseen-room new goal). Descriptive diagnostics: same-state overlap trained vs never-seen room: hidden .86, Kenyon
code .71, place belief 1.00. Literature (reports/Bio plausible OOD navigation transfer.md): insects carry goal knowledge across
places in a central-complex vector memory on a path-integration / place estimate, not in the mushroom body, whose generalisation
equals Kenyon-code overlap (Stone 2017; Le Moel 2019; Campbell 2013; Lu & Webb 2026).
Lean (proofs_ml/ProofsML/Transfer.lean): transfer_identity (transfer = eta delta g <z_A, z_B>); concat_inner (a readout on the
joined code [z, g u] transfers by <z_A, z_B> + g^2 <u_A, u_B>); copies_orthogonal (context-gated copies of the place code are
orthogonal across contexts, so learning in one context cannot move the readout in the other).
Mechanism (one): a goal-vector pathway. The same plastic output synapses (KC -> action, KC -> value; same three-factor TD rule,
traces, metaplasticity, adaptive dopamine, replay) also read the agent's own place belief (obs[150:250]) directly, with gain g:
code = [Kenyon code, g * u], u = b (no gate) or u = b on the copy of the current latent cause, 0 on the other (gate: inferred;
the same latent switch that already gates the Kenyon cells). g = 0 is stage 3 exactly (no extra synapses are created).
postNAS: K = 2048, c = 7, O = 0 fixed; searched as stage 3 (f, eta, tau, lam, rho, m, R, E, T, A, gate) + g {0, .1, .25, .5, 1, 2}.
Objectives (NSGA-II, 3), as stages 4-5: dev new goal, dev old goal kept, dev new goal in the unseen room (dev seeds 1000-1002 x 3
priors). Stage-3 winners (g = 0) enqueued first. 2 workers, both GPU 1, 60 trials each.
Selection (fixed here): largest dev (new + old + unseen-room new), per gate. Test once on GPU 1 (30 seeds x 3 priors, fresh rooms,
four probes). Contrasts (descriptive): stage 6 - PPO, - stage 3.
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
G = B.G; M = B.M

OUT = M.ROOT / "data/audit/v4n_brainlike6_20261107"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"
_KC, _build = B.KC, B.build
P4 = ("moved_trained_room", "familiar_trained_room", "moved_heldout_room", "familiar_heldout_room")
PLACE = {}                                                               # the batch's place belief, set by a hook on the backbone
LEAN = M.ROOT / "proofs_ml/ProofsML/Transfer.lean"


class KC6(_KC):
    """Stage-3 Kenyon cells; with g > 0 the code is joined by g * (context copy of) the agent's place belief."""
    def __init__(self, base, H, cfg, seed, gated):
        super().__init__(base, H, cfg, seed, gated); self.g = float(cfg.get("g", 0.))
        if self.g:
            n = cfg["K"] + 100 * (2 if gated else 1)
            self.R = torch.nn.Linear(n, base.out_features, bias=False).to(self.P.device); torch.nn.init.zeros_(self.R.weight)

    def code(self, h):
        z = super().code(h)
        if not self.g: return z
        b = PLACE["b"].reshape(*h.shape[:-1], 100)                       # the probe feeds (1,1,252) while h is (1,64) (stage-5 ADDENDUM1)
        if self.gated: b = torch.cat([b, torch.zeros_like(b)] if G.CTX["c"] == 0 else [torch.zeros_like(b), b], -1)
        return torch.cat([z, self.g * b], -1)


def build6(prior, seed, cfg):
    m = _build(prior, seed, cfg)
    m.net.body.register_forward_pre_hook(lambda mod, a: PLACE.__setitem__("b", a[0][..., 150:250].float()))
    return m


def run_cfg(args):
    cfg = dict(args[0], O=0); cfg["eta_v"] = cfg["rho"] * (1 - M.GAMMA * cfg["lam"])
    B.KC, B.build, B.Circuit, B.PROTOCOL = KC6, build6, B2.Circuit2, PROTOCOL; return B.run_cfg((cfg,) + tuple(args[1:]))


def objective(trial):
    s = trial.suggest_categorical; fl = trial.suggest_float
    cfg = dict(K=2048, c=7, f=fl("f", .02, .12, log=True), eta=fl("eta", 3., 200., log=True), tau=fl("tau", 1.5, 5., log=True),
               lam=fl("lam", .8, .99), rho=fl("rho", .1, 4., log=True), m=s("m", [0., 1., 10.]), R=s("R", [0, 1]), E=s("E", [1, 2, 4, 8]),
               T=s("T", [8, 16, 32]), A=s("A", [0, 1]), g=s("g", [0., .1, .25, .5, 1., 2.]), gate=s("gate", ["none", "inferred"]))
    jobs = [(cfg, p, sd, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}" / p / f"seed{sd}.json")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = list(ex.map(run_cfg, jobs))
    v = [float(np.mean([r[q] for r in recs])) for q in P4[:3]]
    trial.set_user_attr("sum", sum(v)); trial.set_user_attr("gpu", recs[0]["gpu_name"]); return tuple(v)


def search(w, trials):
    import optuna
    assert PROTOCOL.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    st = optuna.create_study(study_name="v4n_brainlike6", storage=DB, directions=["maximize"] * 3, load_if_exists=True,
                             sampler=optuna.samplers.NSGAIISampler(seed=20261110 + w, population_size=24))
    if w == 0 and not st.trials:
        for g, c3 in json.loads((B3.OUT / "BEST.json").read_text())["configs"].items():
            st.enqueue_trial({k: c3[k] for k in ("f", "eta", "tau", "lam", "rho", "m", "R", "E", "T", "A", "gate")} | dict(g=0.))
    else: time.sleep(5)
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_brainlike6", storage=DB); done = [t for t in st.trials if t.values is not None]
    pick = {g: max((t for t in done if t.params["gate"] == g), key=lambda t: sum(t.values)) for g in ("none", "inferred") if any(t.params["gate"] == g for t in done)}
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done),
        configs={g: dict(t.params, K=2048, c=7) for g, t in pick.items()}, numbers={g: t.number for g, t in pick.items()},
        dev_values={g: t.values for g, t in pick.items()}, pareto=[dict(n=t.number, v=t.values, p=t.params) for t in st.best_trials]), indent=1) + "\n")
    print(json.dumps({g: (t.number, [round(x * 100 / 30, 1) for x in t.values], t.params) for g, t in pick.items()}))


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"; pick = json.loads((OUT / "BEST.json").read_text())["configs"]
    for g, cfg, p, s in [(g, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for g, cfg in sorted(pick.items())][w::n]:
        run_cfg((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / f"bl6_{g}" / p / f"seed{s}.json")))
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
        S = own(OUT / "test" / f"bl6_{g}")
        if not S: continue
        refs = {"ppo": ppo, "bl3": own(B3.OUT / "test" / f"bl3_{g}")}
        R = dict(n=len(S), means={q: float(np.mean([r[q] for r in S.values()]) * 100 / 30) for q in P4}, diverged=sum(r["diverged"] is not None for r in S.values()))
        for o, O in refs.items():
            for q in P4:
                ks = sorted(set(S) & set(O)); R[f"{q}/{o}"] = contrast({"a": {k: S[k][q] * 100 / 30 for k in ks}, "b": {k: O[k][q] * 100 / 30 for k in ks}}, "a", "b", 10.)
        res[g] = R; print(g, R["n"], {q[:14]: round(v, 1) for q, v in R["means"].items()}, "diverged", R["diverged"])
        for q, v in R.items():
            if "/" in q: print("  ", q, round(v["mean"], 1), [round(x, 1) for x in v["ci95"]], "p=%.2g tost=%.2g" % (v["t_p"], v["tost_p"]))
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=M.sha(__file__), design=__doc__,
        lean=dict(file="proofs_ml/ProofsML/Transfer.lean", sha256=M.sha(LEAN)),
        outcomes_seen="stages 1-5 dev and test (incl. unseen-room probes); descriptive code-overlap diagnostics; of stage 6 only the check smoke run (g = .5, stage-3 ungated config, cold_ppo seed 1000, dev rooms): new 30, old 6, unseen new 18, unseen familiar 5 of 30",
        dev_seeds=list(M.DEV_SEEDS), test_seeds=list(M.FR.S30), priors=M.PRIORS), indent=1) + "\n"); print("registered")


def check():
    """g = 0 reproduces the stage-3 winner's dev record exactly; g > 0 joins the place code (both gates) and the probe path runs."""
    import tempfile
    c3 = json.loads((B3.OUT / "BEST.json").read_text()); n3 = c3["numbers"]["none"]
    ref = json.loads((B3.OUT / "search" / f"trial{n3}" / "cold_ppo" / "seed1000.json").read_text())
    global PROTOCOL
    P0, PROTOCOL = PROTOCOL, Path(tempfile.mkdtemp()) / "P.json"; PROTOCOL.write_text("{}")
    rec = run_cfg((dict(c3["configs"]["none"], g=0.), "cold_ppo", 1000, str(M.FR.EXPOSED), str(PROTOCOL.parent / "r0.json")))
    assert all(rec[q] == ref[q] for q in P4), ([rec[q] for q in P4], [ref[q] for q in P4])
    for gate, width in (("none", 2148), ("inferred", 2248)):
        m = build6("cold_ppo", 1000, dict(c3["configs"]["none"], g=.5, gate=gate)); x = torch.rand(5, 252, device=next(m.net.parameters()).device)
        with torch.no_grad():
            G.CTX["c"] = 1; z = m.net.pi.code(m.net.body(x))
        assert z.shape[-1] == width and m.net.pi.R.weight.shape[1] == width and torch.allclose(z[:, -100:] if gate == "inferred" else z[:, 2048:], .5 * x[:, 150:250])
        G.CTX["c"] = 0
    m = build6("cold_ppo", 1000, dict(c3["configs"]["none"], g=.5)); G.MAN = M.FR.EXPOSED; G.CTX.update(c=0, mode="G0"); pair, stage = M.room_of(1000)
    goals, sw = G.probe(m, M.goal_map(M.FR.EXPOSED), pair, stage, 1000, G.Latent(), 1); assert 0 <= goals <= 30
    rec2 = run_cfg((dict(c3["configs"]["none"], g=.5), "cold_ppo", 1000, str(M.FR.EXPOSED), str(PROTOCOL.parent / "r1.json")))
    PROTOCOL = P0; print("check ok", [rec[q] for q in P4], "probe", goals, "g=.5 smoke", [rec2[q] for q in P4], "diverged", rec2["diverged"])


if __name__ == "__main__":
    {"check": check, "prereg": prereg, "best": best, "analyse": analyse, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])),
     "test": lambda: test(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
