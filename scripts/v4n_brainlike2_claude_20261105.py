"""Brain-constrained circuit, stage 2 (camera-ready, author 2026-11-05: "unsupported claims we don't want to give up on
easily; use NAS/postNAS/Optuna"). GPU 1. Registered before any outcome of this stage.
Seen before writing: stage 1 (v4n_brainlike_20261103) test: new goal 15% (PPO 61%), old goal 55%; dev new goal ~24%.
Diagnosis: stage 1 learns strictly online (one three-factor update per step); every learner here that learns the moved
goal reuses each experience several times. Stage 2 keeps the stage-1 circuit and rule and adds two biological mechanisms:
  awake replay  after every T steps, the last T steps (all animals) are replayed E times with the SAME three-factor rule,
                values and policy recomputed from the current synapses (awake sharp-wave-ripple replay of recent experience;
                Foster & Wilson 2006; Ambrose, Pfeiffer & Foster 2016). E = 0 is stage 1.
  adaptive DA   optional: the dopamine signal is divided by a slow running RMS of past TD errors (adaptive coding of
                reward-prediction errors; Tobler, Fiorillo & Schultz 2005). A slow per-animal state, no batch statistics.
Still none of: backprop through the prior, batches statistics, adaptive optimisers, clipping, ratios.
Space: stage-1 space + E {0, 1, 2, 4, 8}, T {8, 16, 32}, A {0, 1}. NSGA-II, 96 trials, dev rooms, 3 dev seeds x 3 priors.
Selection (fixed here): largest dev sum per gate setting. Test once: 30 seeds x 3 priors, fresh rooms, four probes.
Contrasts (descriptive, as stage 1): stage2 - stage1 and stage2 - PPO on the four probes.
  check | prereg | search W TRIALS | best | test W N | analyse
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_brainlike_claude_20261103 as B
M = B.M

OUT = M.ROOT / "data/audit/v4n_brainlike2_20261105"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"


class Circuit2(B.Circuit):
    def __init__(self, net, cfg):
        super().__init__(net, cfg); self.buf = []; self.rms = None

    def dopamine(self, delta):
        if not self.c.get("A"): return delta
        if self.rms is None: self.rms = torch.ones_like(delta)
        self.rms = .999 * self.rms + .001 * delta.detach() ** 2; return delta / self.rms.sqrt().clamp_min(1e-3)

    @torch.no_grad()
    def step(self, z, p, v, a, r, done, v_next):
        c = self.c; g = torch.nn.functional.one_hot(a.to(self.dev), 4).float() - p; keep = 1. - torch.as_tensor(done, device=self.dev, dtype=torch.float32)
        if self.ea is None: self.ea, self.ev, self.kp = torch.zeros(len(z), 4, z.shape[1], device=self.dev), torch.zeros(len(z), z.shape[1], device=self.dev), torch.ones(len(z), device=self.dev)
        gl = M.GAMMA * c["lam"]; r = torch.as_tensor(r, device=self.dev, dtype=torch.float32)
        self.ea = gl * self.ea * self.kp[:, None, None] + g[:, :, None] * z[:, None, :]; self.ev = gl * self.ev * self.kp[:, None] + z; self.kp = keep
        delta = self.dopamine(r + M.GAMMA * v_next * keep - v)
        self.apply((delta[:, None, None] * self.ea).mean(0), (delta[:, None] * self.ev).mean(0)[None])
        if c.get("E"):
            lb = c["tau"] * torch.log(p.clamp_min(1e-8)) - z @ self.Wa.T; vb = v - z @ self.Wv[0]        # prior (cortical) parts
            self.buf.append((z, lb, vb, a.to(self.dev), r, keep, v_next))
            if len(self.buf) == c["T"]: self.replay_recent(); self.buf = []
        return delta

    @torch.no_grad()
    def replay_recent(self):
        c = self.c; gl = M.GAMMA * c["lam"]
        for _ in range(c["E"]):
            n, K = self.buf[0][0].shape; ea = torch.zeros(n, 4, K, device=self.dev); ev = torch.zeros(n, K, device=self.dev); kp = torch.ones(n, device=self.dev)
            for t, (z, lb, vb, a, r, keep, vn_stored) in enumerate(self.buf):
                pc = torch.softmax((lb + z @ self.Wa.T) / c["tau"], -1); v = vb + z @ self.Wv[0]
                if t + 1 < len(self.buf): zn, _, vbn = self.buf[t + 1][:3]; vn = vbn + zn @ self.Wv[0]
                else: vn = vn_stored
                ea = gl * ea * kp[:, None, None] + (torch.nn.functional.one_hot(a, 4).float() - pc)[:, :, None] * z[:, None, :]; ev = gl * ev * kp[:, None] + z; kp = keep
                delta = self.dopamine(r + M.GAMMA * vn * keep - v)
                self.apply((delta[:, None, None] * ea).mean(0), (delta[:, None] * ev).mean(0)[None])
            if not (torch.isfinite(self.Wa).all() and torch.isfinite(self.Wv).all()): return


def run_cfg(args):
    B.Circuit, B.PROTOCOL = Circuit2, PROTOCOL; return B.run_cfg(args)


def objective(trial):
    s = trial.suggest_categorical
    cfg = dict(K=s("K", [1024, 2048, 4096]), c=s("c", [4, 7, 12]), f=trial.suggest_float("f", .02, .2, log=True), eta=trial.suggest_float("eta", .1, 300., log=True), tau=trial.suggest_float("tau", 1., 5., log=True),
               lam=trial.suggest_float("lam", .5, .99), eta_v=trial.suggest_float("eta_v", .003, .3, log=True), m=s("m", [0., 1., 10., 100.]),
               R=s("R", [0, 1, 2]), gate=s("gate", ["none", "inferred"]), E=s("E", [0, 1, 2, 4, 8]), T=s("T", [8, 16, 32]), A=s("A", [0, 1]))
    jobs = [(cfg, p, sd, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}" / p / f"seed{sd}.json")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = list(ex.map(run_cfg, jobs))
    new = float(np.mean([r["moved_trained_room"] for r in recs])); old = float(np.mean([r["familiar_trained_room"] for r in recs]))
    trial.set_user_attr("sum", new + old); return new, old


def search(w, trials):
    import optuna
    assert PROTOCOL.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1")
    st = optuna.create_study(study_name="v4n_brainlike2", storage=DB, directions=["maximize", "maximize"], load_if_exists=True,
                             sampler=optuna.samplers.NSGAIISampler(seed=20261105 + w, population_size=24))
    if w == 0 and not st.trials:                                                # stage-1 winners as references (E = 0)
        for g, c1 in json.loads((B.OUT / "BEST.json").read_text())["configs"].items(): st.enqueue_trial(dict(c1, E=0, T=16, A=0))
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_brainlike2", storage=DB); done = [t for t in st.trials if t.values is not None]
    pick = {g: max((t for t in done if t.params["gate"] == g), key=lambda t: sum(t.values)) for g in ("none", "inferred") if any(t.params["gate"] == g for t in done)}
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done),
        configs={g: t.params for g, t in pick.items()}, dev_values={g: t.values for g, t in pick.items()}, pareto=[dict(n=t.number, v=t.values, p=t.params) for t in st.best_trials]), indent=1) + "\n")
    print(json.dumps({g: (t.number, t.values, t.params) for g, t in pick.items()}))


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1"); pick = json.loads((OUT / "BEST.json").read_text())["configs"]
    for g, cfg, p, s in [(g, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for g, cfg in sorted(pick.items())][w::n]:
        run_cfg((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / f"bl2_{g}" / p / f"seed{s}.json")))
    print(json.dumps(dict(worker=w, done=True)), flush=True)


def analyse():
    G = B.G; load = lambda d: {(r["prior"], r["seed"]): r for r in (json.loads(f.read_text()) for f in d.glob("*/seed*.json"))}
    R = {f"bl2_{g}": load(OUT / "test" / f"bl2_{g}") for g in ("none", "inferred")}
    R.update({f"bl1_{g}": load(B.OUT / "test" / f"bl_{g}") for g in ("none", "inferred")}, G2=load(G.OUT / "runs" / "G2"))
    R = {k: v for k, v in R.items() if v}; keys = sorted(set.intersection(*[set(v) for v in R.values()]))
    P = ("moved_trained_room", "familiar_trained_room", "moved_heldout_room", "familiar_heldout_room"); pct = lambda r, k: r[k] * 100 / 30
    res = dict(n=len(keys), means={c: {k: float(np.mean([pct(R[c][x], k) for x in keys])) for k in P} for c in R}, contrasts={})
    for a in [k for k in R if k.startswith("bl2")]:
        for b in [k for k in R if not k.startswith("bl2")]:
            for k in P:
                d = np.array([pct(R[a][x], k) - pct(R[b][x], k) for x in keys]); h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
                res["contrasts"][f"{a}-{b}/{k}"] = dict(mean=float(d.mean()), ci=[float(d.mean() - h), float(d.mean() + h)])
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res["means"], indent=1))


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=M.sha(__file__), design=__doc__,
        outcomes_seen="stage-1 brainlike test and dev; context gating G0/G1/G2 tests; latent search test; none of stage 2",
        dev_seeds=list(M.DEV_SEEDS), test_seeds=list(M.FR.S30), priors=M.PRIORS), indent=1) + "\n"); print("registered")


def check():
    """E = 0, A = 0 reproduces stage 1 exactly on a synthetic stream."""
    torch.manual_seed(0); K, n = 64, 3
    class Net: pass
    net = Net(); net.pi = Net(); net.v = Net(); net.pi.R = torch.nn.Linear(K, 4, bias=False); net.v.R = torch.nn.Linear(K, 1, bias=False)
    for q in (net.pi.R.weight, net.v.R.weight): torch.nn.init.zeros_(q); q.requires_grad_(False)
    cfg = dict(eta=1., eta_v=.1, m=1., lam=.9, tau=1.3, R=0, E=0, T=4, A=0)
    import copy; net2 = copy.deepcopy(net); L1, L2 = B.Circuit(net, cfg), Circuit2(net2, cfg)
    for t in range(50):
        z = torch.relu(torch.randn(n, K)); p = torch.softmax(torch.randn(n, 4), -1); v = torch.randn(n); a = torch.randint(0, 4, (n,)); r = np.random.rand(n) > .8; d = np.random.rand(n) > .9
        L1.step(z, p, v, a, r.astype(float), d, torch.randn(n) * 0 + v); L2.step(z, p, v, a, r.astype(float), d, v)
    assert torch.allclose(net.pi.R.weight, net2.pi.R.weight) and torch.allclose(net.v.R.weight, net2.v.R.weight)
    L3 = Circuit2(copy.deepcopy(net), dict(cfg, E=2, A=1)); L3.step(z, p, v, a, r.astype(float), d, v); [L3.step(z, p, v, a, r.astype(float), d, v) for _ in range(4)]
    assert torch.isfinite(L3.Wa).all(); print("check ok")


if __name__ == "__main__":
    {"check": check, "prereg": prereg, "best": best, "analyse": analyse, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])),
     "test": lambda: test(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
