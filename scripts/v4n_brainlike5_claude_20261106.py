"""Mushroom-body circuit, stage 5 (camera-ready, author 2026-11-06: "continue and make sure to do LEAN based for unseen over bio
plausible using GPU 1"). GPU 1 only. Registered before any outcome of this stage.
Seen before writing: stages 1-4 dev and test, incl. the unseen-room probes (stage 3 test: unseen-room new goal 20.4%, -7.4 pp
[-13.4, -1.5] vs PPO; stage 4: dense lateral-horn pathway rejected on dev). Descriptive diagnostics (browser port, human-choice
prior, seeds 1000-1004, not outcomes of this stage): at the same place and heading in the trained room and its never-seen
partner, backbone hidden codes are cos .86 (.72 along matched trajectories), Kenyon-cell codes .71 (.61; 58% of active cells
shared), and the agent's place belief 1.00 (same most likely cell in 100% of matched states).
Lean (proofs_ml/ProofsML/Transfer.lean, Mathlib, no sorry): transfer_identity -- after a three-factor update at state A the
readout at any state B moves by eta delta g <z_A, z_B>, so transfer to a never-seen room is proportional to Kenyon-code overlap;
self_interference -- identical views (old and new goal share them) get the full update, so sparseness cannot reduce that
interference; drive_inner/drive_diff -- a common drive beta*c added to both pre-activations leaves their difference unchanged
and adds beta(<a,c>+<c,b>) + beta^2 |c|^2 to their overlap.
Mechanism (one): place drive. Each Kenyon cell also sums cp random entries of the agent's own place belief (fixed, sparse,
signed, never trained; hippocampal / central-complex place input converging on a learning circuit), with gain beta, before the
global inhibition (k-WTA) and normalisation: pre = h P + beta * (b Q). beta = 0 is stage 3 exactly (O = 0).
postNAS: K = 2048, c = 7, O = 0 fixed; searched as stage 3 (f, eta, tau, lam, rho, m, R, E, T, A, gate) + beta {0, .5, 1, 2, 4}
+ cp {3, 7, 15}. Objectives (NSGA-II, 3), as stage 4: dev new goal, dev old goal, dev new goal in the unseen room (dev seeds
1000-1002 x 3 priors). The stage-3 winners (beta = 0) are enqueued first as references. 2 workers, both on GPU 1, 60 trials each.
Selection (fixed here): largest dev (new + old + unseen-room new), per gate. Test once on GPU 1 (30 seeds x 3 priors, fresh rooms,
four probes). Contrasts (descriptive): stage 5 - PPO, - stage 3.
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

OUT = M.ROOT / "data/audit/v4n_brainlike5_20261106"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"
_KC, _build = B.KC, B.build
P4 = ("moved_trained_room", "familiar_trained_room", "moved_heldout_room", "familiar_heldout_room")
PLACE = {}                                                               # the batch's place belief, set by a hook on the backbone


class KC5(_KC):
    """Stage-3 Kenyon cells plus a fixed sparse place drive from the agent's own place belief (obs[150:250])."""
    def __init__(self, base, H, cfg, seed, gated):
        super().__init__(base, H, cfg, seed, gated); self.beta = float(cfg.get("beta", 0.)); cp = int(cfg.get("cp", 7))
        g = torch.Generator().manual_seed(19000 + seed); Q = torch.zeros(100, cfg["K"])
        for j in range(cfg["K"]):
            v = torch.randn(cp, generator=g) / np.sqrt(cp); Q[torch.randperm(100, generator=g)[:cp], j] = v
        self.register_buffer("Q", Q.to(self.P.device))

    def code(self, h):                                                   # = Gated.code + KC normalisation, plus the place drive
        z = h @ self.P
        if self.beta: z = z + self.beta * (PLACE["b"].reshape(*h.shape[:-1], 100) @ self.Q)   # ADDENDUM1: the probe feeds (1,1,252), h is (1,64)
        if self.gated: z = z.masked_fill(~self.mask[G.CTX["c"]], float("-inf"))
        top = torch.topk(z, self.k, -1); z = torch.zeros_like(z).scatter(-1, top.indices, torch.relu(top.values))
        return z / (z.norm(dim=-1, keepdim=True) + 1e-6)


def build5(prior, seed, cfg):
    m = _build(prior, seed, cfg)
    m.net.body.register_forward_pre_hook(lambda mod, a: PLACE.__setitem__("b", a[0][..., 150:250].float()))
    return m


def run_cfg(args):
    cfg = dict(args[0], O=0); cfg["eta_v"] = cfg["rho"] * (1 - M.GAMMA * cfg["lam"])
    B.KC, B.build, B.Circuit, B.PROTOCOL = KC5, build5, B2.Circuit2, PROTOCOL; return B.run_cfg((cfg,) + tuple(args[1:]))


def objective(trial):
    s = trial.suggest_categorical; fl = trial.suggest_float
    cfg = dict(K=2048, c=7, f=fl("f", .02, .12, log=True), eta=fl("eta", 3., 200., log=True), tau=fl("tau", 1.5, 5., log=True),
               lam=fl("lam", .8, .99), rho=fl("rho", .1, 4., log=True), m=s("m", [0., 1., 10.]), R=s("R", [0, 1]), E=s("E", [1, 2, 4, 8]),
               T=s("T", [8, 16, 32]), A=s("A", [0, 1]), beta=s("beta", [0., .5, 1., 2., 4.]), cp=s("cp", [3, 7, 15]), gate=s("gate", ["none", "inferred"]))
    jobs = [(cfg, p, sd, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}" / p / f"seed{sd}.json")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = list(ex.map(run_cfg, jobs))
    v = [float(np.mean([r[q] for r in recs])) for q in P4[:3]]
    trial.set_user_attr("sum", sum(v)); trial.set_user_attr("gpu", recs[0]["gpu_name"]); return tuple(v)


def search(w, trials):
    import optuna
    assert PROTOCOL.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    st = optuna.create_study(study_name="v4n_brainlike5", storage=DB, directions=["maximize"] * 3, load_if_exists=True,
                             sampler=optuna.samplers.NSGAIISampler(seed=20261108 + w, population_size=24))
    if w == 0 and not st.trials:
        for g, c3 in json.loads((B3.OUT / "BEST.json").read_text())["configs"].items():
            st.enqueue_trial({k: c3[k] for k in ("f", "eta", "tau", "lam", "rho", "m", "R", "E", "T", "A", "gate")} | dict(beta=0., cp=7))
    else: time.sleep(5)
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_brainlike5", storage=DB); done = [t for t in st.trials if t.values is not None]
    pick = {g: max((t for t in done if t.params["gate"] == g), key=lambda t: sum(t.values)) for g in ("none", "inferred") if any(t.params["gate"] == g for t in done)}
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done),
        configs={g: dict(t.params, K=2048, c=7) for g, t in pick.items()}, numbers={g: t.number for g, t in pick.items()},
        dev_values={g: t.values for g, t in pick.items()}, pareto=[dict(n=t.number, v=t.values, p=t.params) for t in st.best_trials]), indent=1) + "\n")
    print(json.dumps({g: (t.number, [round(x * 100 / 30, 1) for x in t.values], t.params) for g, t in pick.items()}))


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"; pick = json.loads((OUT / "BEST.json").read_text())["configs"]
    for g, cfg, p, s in [(g, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for g, cfg in sorted(pick.items())][w::n]:
        run_cfg((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / f"bl5_{g}" / p / f"seed{s}.json")))
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
        S = own(OUT / "test" / f"bl5_{g}")
        if not S: continue
        refs = {"ppo": ppo, "bl3": own(B3.OUT / "test" / f"bl3_{g}")}
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
        lean=dict(file="proofs_ml/ProofsML/Transfer.lean", sha256=M.sha(M.ROOT / "proofs_ml/ProofsML/Transfer.lean")),
        outcomes_seen="stages 1-4 dev and test (incl. unseen-room probes); descriptive code-overlap diagnostics listed in the design; none of stage 5",
        dev_seeds=list(M.DEV_SEEDS), test_seeds=list(M.FR.S30), priors=M.PRIORS), indent=1) + "\n"); print("registered")


def check():
    """beta = 0 reproduces the stage-3 winner's dev record exactly; beta > 0 changes the code (one prior x seed, same GPU model)."""
    import tempfile
    c3 = json.loads((B3.OUT / "BEST.json").read_text()); n3 = c3["numbers"]["none"]
    ref = json.loads((B3.OUT / "search" / f"trial{n3}" / "cold_ppo" / "seed1000.json").read_text())
    global PROTOCOL
    P0, PROTOCOL = PROTOCOL, Path(tempfile.mkdtemp()) / "P.json"; PROTOCOL.write_text("{}")
    rec = run_cfg((dict(c3["configs"]["none"], beta=0., cp=7), "cold_ppo", 1000, str(M.FR.EXPOSED), str(PROTOCOL.parent / "r0.json")))
    assert all(rec[q] == ref[q] for q in P4), ([rec[q] for q in P4], [ref[q] for q in P4])
    m = build5("cold_ppo", 1000, dict(c3["configs"]["none"], beta=2., cp=7)); x = torch.rand(5, 252, device=next(m.net.parameters()).device)
    with torch.no_grad():
        h = m.net.body(x); z2 = m.net.pi.code(h); m.net.pi.beta = 0.; z0 = m.net.pi.code(h)
    assert not torch.equal(z2, z0) and int((m.net.pi.Q != 0).sum(0).max()) == 7
    m.net.pi.beta = m.net.v.beta = 2.; G.MAN = M.FR.EXPOSED; G.CTX.update(c=0, mode="G0"); pair, stage = M.room_of(1000)   # ADDENDUM1: the frozen probe path
    goals, sw = G.probe(m, M.goal_map(M.FR.EXPOSED), pair, stage, 1000, G.Latent(), 1); assert 0 <= goals <= 30
    PROTOCOL = P0; print("check ok", [rec[q] for q in P4], "probe with place drive", goals)


if __name__ == "__main__":
    {"check": check, "prereg": prereg, "best": best, "analyse": analyse, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])),
     "test": lambda: test(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
