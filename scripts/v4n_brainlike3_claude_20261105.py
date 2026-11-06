"""Brain-constrained circuit, stage 3 (camera-ready, author 2026-11-05: "use both GPUs and improve our significance,
especially the biologically plausible ones; utilize postNAS/Optuna and Lean"). Registered before any outcome of this stage.
Seen before writing: stage 2 test (v4n_brainlike2_20261105): ungated new 60.1 / old 24.6 (equivalent to PPO within
+/-10 pp); gated new 53.4 / old 37.4 (old +13.3 vs PPO, new -7.3 [-16.9, 2.2]); unseen-room new goal -7.9 vs PPO.
Stage 3 = stage 2 + one mechanism aimed at keeping the old goal while learning the new one:
  sleep replay  rewarded episodes of the familiar-goal phase are stored (hippocampal memory, at most 64) and, after every
                T moved-goal steps, O of them are replayed in reverse with the same three-factor rule, values and
                policy recomputed from the current synapses (interleaved replay / complementary learning systems;
                McClelland et al. 1995; offline replay of old experience, Wilson & McNaughton 1994). O = 0 is stage 2.
Lean (proofs/Proofs/Trace.lean): the eligibility trace is bounded by M / (1 - gamma lambda); the value rate is therefore
searched in these units, rho = eta_v / (1 - gamma lambda), which decouples it from lambda.
postNAS: K = 2048 and c = 7 fixed (both stage-2 winners); searched f [.02,.12] log, eta [3,200] log, tau [1.5,5] log,
lam [.8,.99], rho [.1,4] log, m {0,1,10}, R {0,1}, E {1,2,4,8}, T {8,16,32}, A {0,1}, O {0,1,2,4}, gate {none, inferred}.
The two stage-2 winners are enqueued first (O = 0) as references. NSGA-II, 2 workers (one per GPU), 60 trials each;
objective and dev rooms as stages 1-2. Selection (fixed here): largest dev sum per gate. Test once on GPU 1 only
(30 seeds x 3 priors, fresh rooms, four probes). Contrasts (descriptive): stage 3 - stage 2, - PPO, - G2.
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
M = B.M

OUT = M.ROOT / "data/audit/v4n_brainlike3_20261105"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"
CAP = 64


class Circuit3(B2.Circuit2):
    def __init__(self, net, cfg):
        super().__init__(net, cfg); self.t = 0; self.mem = []; self.cur = None; self.g = torch.Generator().manual_seed(4242)

    @torch.no_grad()
    def step(self, z, p, v, a, r, done, v_next):
        c = self.c; fam = self.t * M.N_ENV < B.G.PRE; self.t += 1
        if c.get("O") and fam:                                                  # store familiar-goal episodes
            lb = c["tau"] * torch.log(p.clamp_min(1e-8)) - z @ self.Wa.T; vb = v - z @ self.Wv[0]
            if self.cur is None: self.cur = [[] for _ in range(len(z))]
            for i in range(len(z)):
                self.cur[i].append((z[i].clone(), lb[i].clone(), float(vb[i]), a[i].to(self.dev), float(r[i])))
                if done[i]:
                    if r[i] > .5 and len(self.mem) < CAP: self.mem.append(self.cur[i])
                    self.cur[i] = []
        delta = super().step(z, p, v, a, r, done, v_next)
        if c.get("O") and not fam and self.mem and self.t % c["T"] == 0:
            for j in torch.randint(len(self.mem), (c["O"],), generator=self.g).tolist(): self.replay_old(self.mem[j])
        return delta

    @torch.no_grad()
    def replay_old(self, ep):
        c = self.c; gl = M.GAMMA * c["lam"]; K = ep[0][0].shape[0]; ea = torch.zeros(4, K, device=self.dev); ev = torch.zeros(K, device=self.dev); vn = 0.
        for z, lb, vb, a, r in reversed(ep):
            pc = torch.softmax((lb + self.Wa @ z) / c["tau"], -1); v = vb + float(self.Wv[0] @ z)
            ea = gl * ea + (torch.nn.functional.one_hot(a, 4).float() - pc)[:, None] * z[None]; ev = gl * ev + z
            delta = r + M.GAMMA * vn - v; vn = v
            if c.get("A") and self.rms is not None: delta = delta / float(self.rms.mean().sqrt().clamp_min(1e-3))
            self.apply(delta * ea, delta * ev[None])


def run_cfg(args):
    cfg = dict(args[0]); cfg.setdefault("eta_v", cfg["rho"] * (1 - M.GAMMA * cfg["lam"]) if "rho" in cfg else cfg.get("eta_v"))
    B.Circuit, B.PROTOCOL = Circuit3, PROTOCOL; return B.run_cfg((cfg,) + tuple(args[1:]))


def objective(trial):
    s = trial.suggest_categorical; fl = trial.suggest_float
    cfg = dict(K=2048, c=7, f=fl("f", .02, .12, log=True), eta=fl("eta", 3., 200., log=True), tau=fl("tau", 1.5, 5., log=True),
               lam=fl("lam", .8, .99), rho=fl("rho", .1, 4., log=True), m=s("m", [0., 1., 10.]), R=s("R", [0, 1]), E=s("E", [1, 2, 4, 8]),
               T=s("T", [8, 16, 32]), A=s("A", [0, 1]), O=s("O", [0, 1, 2, 4]), gate=s("gate", ["none", "inferred"]))
    jobs = [(cfg, p, sd, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}" / p / f"seed{sd}.json")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = list(ex.map(run_cfg, jobs))
    new = float(np.mean([r["moved_trained_room"] for r in recs])); old = float(np.mean([r["familiar_trained_room"] for r in recs]))
    trial.set_user_attr("sum", new + old); trial.set_user_attr("gpu", recs[0]["gpu_name"]); return new, old


def search(w, trials):
    import optuna
    assert PROTOCOL.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1")
    st = optuna.create_study(study_name="v4n_brainlike3", storage=DB, directions=["maximize", "maximize"], load_if_exists=True,
                             sampler=optuna.samplers.NSGAIISampler(seed=20261106 + w, population_size=24))
    if w == 0 and not st.trials:
        for g, c2 in json.loads((B2.OUT / "BEST.json").read_text())["configs"].items():
            st.enqueue_trial({k: c2[k] for k in ("f", "eta", "tau", "lam", "m", "R", "E", "T", "A", "gate")} | dict(rho=c2["eta_v"] / (1 - M.GAMMA * c2["lam"]), O=0))
    else: time.sleep(5)
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_brainlike3", storage=DB); done = [t for t in st.trials if t.values is not None]
    pick = {g: max((t for t in done if t.params["gate"] == g), key=lambda t: sum(t.values)) for g in ("none", "inferred") if any(t.params["gate"] == g for t in done)}
    cfgs = {g: dict(t.params, K=2048, c=7) for g, t in pick.items()}
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done),
        configs=cfgs, numbers={g: t.number for g, t in pick.items()}, dev_values={g: t.values for g, t in pick.items()},
        pareto=[dict(n=t.number, v=t.values, p=t.params) for t in st.best_trials]), indent=1) + "\n")
    print(json.dumps({g: (t.number, t.values, t.params) for g, t in pick.items()}))


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"; pick = json.loads((OUT / "BEST.json").read_text())["configs"]
    for g, cfg, p, s in [(g, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for g, cfg in sorted(pick.items())][w::n]:
        run_cfg((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / f"bl3_{g}" / p / f"seed{s}.json")))
    print(json.dumps(dict(worker=w, done=True)), flush=True)


def analyse():
    import v4n_kpi_claude_20261031 as K
    from prior_seed_power_claude import contrast
    own = lambda d: {(r["prior"], r["seed"]): r for r in (json.loads(f.read_text()) for f in d.glob("*/seed*.json"))}
    ppo = {}
    for f in (K.OUT / "kpi/ppo").glob("*/seed*.json"):
        r = json.loads(f.read_text()); c = json.loads(Path(r["source"]).read_text())["curve"]; ppo[(r["prior"], r["seed"])] = dict(r, moved_trained_room=c[str(M.BUDGET)])
    P = ("moved_trained_room", "familiar_trained_room", "moved_heldout_room", "familiar_heldout_room"); res = {}
    for g in ("none", "inferred"):
        S = own(OUT / "test" / f"bl3_{g}"); refs = {"ppo": ppo, "bl2": own(B2.OUT / "test" / f"bl2_{g}"), "G2": own(B.G.OUT / "runs/G2")}
        R = dict(n=len(S), means={q: float(np.mean([r[q] for r in S.values()]) * 100 / 30) for q in P}, diverged=sum(r["diverged"] is not None for r in S.values()))
        for o, O in refs.items():
            for q in P:
                ks = sorted(set(S) & set(O)); R[f"{q}/{o}"] = contrast({"a": {k: S[k][q] * 100 / 30 for k in ks}, "b": {k: O[k][q] * 100 / 30 for k in ks}}, "a", "b", 10.)
        for b in ("human_legal_uniform", "cold_ppo"):
            for q in P[:2]: R[f"prior/{q}/human_bc-{b}"] = contrast({p: {k[1]: S[k][q] * 100 / 30 for k in S if k[0] == p} for p in ("human_bc", b)}, "human_bc", b, 10.)
        res[g] = R; print(g, R["n"], {q[:12]: round(v, 1) for q, v in R["means"].items()}, "diverged", R["diverged"])
        for q, v in R.items():
            if "/" in q: print("  ", q, round(v["mean"], 1), [round(x, 1) for x in v["ci95"]], "p=%.2g tost=%.2g" % (v["t_p"], v["tost_p"]))
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=M.sha(__file__), design=__doc__,
        outcomes_seen="stages 1-2 dev and test; gating G0/G1/G2; latent search; none of stage 3", lean="proofs/Proofs/Trace.lean (lake build ok)",
        dev_seeds=list(M.DEV_SEEDS), test_seeds=list(M.FR.S30), priors=M.PRIORS), indent=1) + "\n"); print("registered")


def check():
    """O = 0 reproduces stage 2 on a synthetic stream; O > 0 stores and replays without error."""
    import copy
    class Net: pass
    torch.manual_seed(0); K, n = 32, 3; net = Net(); net.pi = Net(); net.v = Net()
    net.pi.R = torch.nn.Linear(K, 4, bias=False); net.v.R = torch.nn.Linear(K, 1, bias=False)
    for q in (net.pi.R.weight, net.v.R.weight): torch.nn.init.zeros_(q); q.requires_grad_(False)
    cfg = dict(eta=1., eta_v=.1, m=1., lam=.9, tau=1.3, R=0, E=1, T=4, A=1, O=0)
    n2, n3 = copy.deepcopy(net), copy.deepcopy(net); L2, L3, L4 = B2.Circuit2(net, cfg), Circuit3(n2, cfg), Circuit3(n3, dict(cfg, O=2))
    for t in range(400):
        z = torch.relu(torch.randn(n, K)); p = torch.softmax(torch.randn(n, 4), -1); v = torch.randn(n); a = torch.randint(0, 4, (n,))
        r = (np.random.rand(n) > .7).astype(float); d = np.random.rand(n) > .8
        for L in (L2, L3, L4): L.step(z, p, v, a, r, d, v)
    assert torch.allclose(net.pi.R.weight, n2.pi.R.weight) and len(L4.mem) > 0 and torch.isfinite(n3.pi.R.weight).all()
    assert not torch.allclose(n3.pi.R.weight, n2.pi.R.weight); print("check ok", len(L4.mem))


if __name__ == "__main__":
    {"check": check, "prereg": prereg, "best": best, "analyse": analyse, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])),
     "test": lambda: test(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
