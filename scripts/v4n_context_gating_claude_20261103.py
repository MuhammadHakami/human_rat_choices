"""Context-gated local plasticity: is forgetting here a failure of locality or a missing context? (camera-ready
diagnostic, author 2026-11-03). GPU 1. Registered; no search (hyperparameters = v4n_local_search2 BEST_head).
Same states, different choices: the two goals give identical views (Rosetta stage: views do not say which action), so a
context-free learner must rewrite the synapses that hold the old goal (v4n_forgetting_diagnosis: local rules move 52-94%
of synapses and retain less than PPO at matched learning). Recent bio-plausible remedies gate plasticity by context
(dendritic / inhibitory gating, Neuron 2026; evidence-gated plasticity, 2026; astrocyte-gated factor, 2026; XdG, Masse 2018).
Learner (all conditions): body and the prior's heads FROZEN; only the readout of a fixed random k-WTA expansion
(granule-like, 2048 units) learns, with the search2 replay three-factor rule. The prior is unchanged at the start.
Budget 32,768 interactions: 2,048 in the familiar goal (3,3), then 30,720 in the moved goal (same room).
  G0 ungated    one code for all experience
  G1 oracle     the expansion is split into two disjoint halves; the goal identity selects the half (upper bound)
  G2 inferred   no cue: a latent-cause rule picks the half. Each half keeps an expected success e_c (EMA, .05, from the
                episodes it ran); once e_c >= .25 over >= 10 episodes, a CUSUM on episode outcomes against a 5% rate
                (bound 4.5) detects that the context has stopped paying and switches to the other half; a half with no
                established expectation is never abandoned. The same rule runs during the frozen probes (no weights change).
Outcomes (30 seeds x 3 priors, fresh rooms, as the KPI probes): moved goal and familiar goal in the trained room,
moved and familiar goal in the unseen room. Probes start from the context state at the end of learning (G2).
Registered contrasts (paired by prior x seed, Holm over 4): familiar-goal retention G1-G0 and G2-G0; moved-goal final
G1-G0 and G2-G0 (cost of gating). Prediction: G1 and G2 retain more with little cost.
  check | prereg | run W N | analyse
"""
import copy, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_local_search2_claude_20261102 as S2
M = S2.M

OUT = M.ROOT / "data/audit/v4n_context_gating_20261103"; PROTOCOL = OUT / "PROTOCOL.json"
FAM, PRE = (3, 3), 2048; MAN = M.FR.MAN; CONDS = ("G0", "G1", "G2"); CTX = dict(c=0, mode="G0")   # module-level: survives the probe's deepcopy
BASE_CFG = json.loads((S2.OUT / "BEST_head.json").read_text())["configs"]["head"] if (S2.OUT / "BEST_head.json").exists() else None


class Latent:
    """Two-slot latent-cause switch on episode outcomes. A context is established after 20 episodes; a CUSUM tests its
    success rate e (held fixed while a change is suspected) against a drop to e/4, bound 12; after a switch the new
    context dwells >= 20 episodes; a context with no established expectation is never abandoned."""
    def __init__(self): self.e, self.n, self.c, self.S, self.switches, self.dwell = [0., 0.], [0, 0], 0, 0., 0, 0

    def update(self, success):
        c = self.c; self.n[c] += 1; self.dwell += 1
        if self.S > 0 or (self.n[c] > 20 and self.e[c] >= .25 and self.dwell > 20):   # an open test runs to its end
            if self.S == 0: self.ref = min(self.e[c], .95)                     # reference frozen while a change is suspected
            e = self.ref; q = e / 4
            self.S = max(0., self.S + (np.log(q / e) if success else np.log((1 - q) / (1 - e))))
            if self.S > 12: self.c, self.S, self.switches, self.dwell = 1 - c, 0., self.switches + 1, 0; return
        self.e[c] += (float(success) - self.e[c]) / min(self.n[c], 20)          # running mean, then EMA (1/20)


class Gated(S2.Expanded):
    def __init__(self, base, P, k, gated):
        super().__init__(base, P, k); K = P.shape[1]; g = torch.Generator().manual_seed(31337); perm = torch.randperm(K, generator=g)
        mask = torch.zeros(2, K, dtype=torch.bool); mask[0, perm[:K // 2]] = True; mask[1, perm[K // 2:]] = True
        self.register_buffer("mask", mask.to(P.device)); self.gated = gated; self.k = max(1, k // 2) if gated else k

    def code(self, h):
        z = h @ self.P
        if self.gated: z = z.masked_fill(~self.mask[CTX["c"]], float("-inf"))
        top = torch.topk(z, self.k, -1); return torch.zeros_like(z).scatter(-1, top.indices, torch.relu(top.values))


def build(prior, seed, gated):
    m = M.build(prior, seed); net = m.net; dev = net.pi.weight.device; H = net.pi.in_features; c = BASE_CFG
    g = torch.Generator().manual_seed(9000 + seed); P = (torch.randn(H, c["expand"], generator=g) / np.sqrt(H)).to(dev); k = max(1, int(round(c["kwta"] * c["expand"])))
    net.pi = Gated(net.pi, P, k, gated).to(dev); net.v = Gated(net.v, P, k, gated).to(dev); return m


class ReadoutOnly(S2.ReplayLocal):
    def __init__(self, net, cfg, seed):
        super().__init__(net, dict(cfg, plastic="head"), seed); self.params = [net.pi.R.weight, net.v.R.weight]
        self.m = [torch.zeros_like(q) for q in self.params]; self.s2 = [torch.zeros_like(q) for q in self.params]

    def dirs(self, hs, z, gout, verr): return [gout.T @ z / len(verr), verr[None] @ z / len(verr)]


class Watched:
    """Env wrapper: feeds episode outcomes to the latent-cause switch (G2 only)."""
    def __init__(self, env, lat): self.env, self.lat = env, lat

    def __getattr__(self, k): return getattr(self.env, k)

    def step(self, a):
        obs, r, d = self.env.step(a)
        if CTX["mode"] == "G2":
            for i in np.where(d)[0]: self.lat.update(r[i] > .5)
            CTX["c"] = self.lat.c
        return obs, r, d


def probe(m, gm, pair, stage, seed, lat, ctx_oracle):
    real = M.env_for; saved = copy.deepcopy(lat)
    def wrapped(*a, **k): return Watched(real(*a, **k), saved)
    CTX["c"] = ctx_oracle if CTX["mode"] == "G1" else (saved.c if CTX["mode"] == "G2" else 0)
    M.env_for = wrapped
    try: r = M.probe_now(m, MAN, gm, pair, stage, seed)
    finally: M.env_for = real
    return r["goals"], saved.switches


def run_one(cond, prior, seed):
    out = OUT / "runs" / cond / prior / f"seed{seed}.json"
    if out.exists(): return json.loads(out.read_text())
    CTX.update(c=0, mode=cond); pair, stage = M.room_of(seed); gm = M.goal_map(MAN); fam = {c: FAM for c in gm}
    m = build(prior, seed, cond != "G0"); L = ReadoutOnly(m.net, BASE_CFG, seed); lat = Latent(); gen = torch.Generator().manual_seed(77 + seed)
    t0 = time.perf_counter(); log = []
    for phase, goals, frames, oracle in (("familiar", fam, PRE, 0), ("moved", gm, M.BUDGET - PRE, 1)):
        if cond == "G1": CTX["c"] = oracle
        env = Watched(M.env_for(MAN, goals, pair, M.N_ENV, seed + (500 if phase == "familiar" else 0), "clean", frames), lat); obs = env.reset(); n, buf = 0, []
        while n < frames:
            c_now = CTX["c"]; a, p, v = L.act(obs, gen); o0 = torch.as_tensor(obs, device=L.dev); obs, r, done = env.step(a.numpy()); n += M.N_ENV
            buf.append((o0, a.to(L.dev), torch.as_tensor(r, device=L.dev, dtype=torch.float32), torch.as_tensor(done, device=L.dev, dtype=torch.float32),
                        torch.log(p.gather(1, a.to(L.dev)[:, None]).squeeze(1).clamp_min(1e-8))))
            if len(buf) == BASE_CFG["T"] or n >= frames or CTX["c"] != c_now:      # one context per replayed chunk
                c_new = CTX["c"]; CTX["c"] = c_now; L.learn(buf, L.forward(obs)[3], .0); CTX["c"] = c_new; buf = []
        log.append(dict(phase=phase, context_end=CTX["c"], switches=lat.switches, e=list(lat.e)))
    res = {}
    for key, goals, st, orc in (("moved_trained_room", gm, stage, 1), ("familiar_trained_room", fam, stage, 0),
                                ("moved_heldout_room", gm, "B", 1), ("familiar_heldout_room", fam, "B", 0)):
        res[key], res[key + "_switches"] = probe(m, goals, pair, st, seed, lat, orc)
    rec = dict(cond=cond, prior=prior, seed=seed, pair=pair, **res, phases=log, wall_s=time.perf_counter() - t0, cfg=BASE_CFG,
               protocol_sha256=M.sha(PROTOCOL), gpu_name=torch.cuda.get_device_name(0))
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec) + "\n"); return rec


def run(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and PROTOCOL.exists(); torch.set_num_threads(1)
    for c, p, s in [(c, p, s) for s in M.FR.S30 for p in M.PRIORS for c in CONDS][w::n]: run_one(c, p, s)
    print(json.dumps(dict(stage="run", worker=w, done=True)), flush=True)


def analyse():
    from analyse_seeds30_claude import holm
    from scipy import stats
    R = {c: {(r["prior"], r["seed"]): r for r in (json.loads(f.read_text()) for f in (OUT / "runs" / c).glob("*/seed*.json"))} for c in CONDS}
    keys = sorted(set.intersection(*[set(v) for v in R.values()])); pct = lambda r, k: r[k] * 100 / 30
    res = dict(n=len(keys), means={c: {k: float(np.mean([pct(R[c][x], k) for x in keys])) for k in ("moved_trained_room", "familiar_trained_room", "moved_heldout_room", "familiar_heldout_room")} for c in CONDS},
               per_prior={c: {p: {k: float(np.mean([pct(R[c][x], k) for x in keys if x[0] == p])) for k in ("moved_trained_room", "familiar_trained_room")} for p in M.PRIORS} for c in CONDS},
               switches={c: float(np.mean([R[c][x]["phases"][-1]["switches"] for x in keys])) for c in CONDS}, tests={})
    for c in ("G1", "G2"):
        for k in ("familiar_trained_room", "moved_trained_room"):
            d = np.array([pct(R[c][x], k) - pct(R["G0"][x], k) for x in keys]); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            res["tests"][f"{c}-G0/{k}"] = dict(mean=float(d.mean()), ci=[float(d.mean() - h), float(d.mean() + h)], p=float(stats.ttest_1samp(d, 0).pvalue))
    for k, hp in zip(res["tests"], holm([v["p"] for v in res["tests"].values()])): res["tests"][k]["holm_p"] = hp
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res, indent=1))


def prereg():
    assert not PROTOCOL.exists() and BASE_CFG is not None; OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), outcomes_seen="pilots: 3 runs on test seed human_bc/1001 (debugging an oscillating switch; not part of the registered runs, which are recomputed) and 4 dev-room runs; search2/stability/diagnosis outcomes seen",
        script_sha256=M.sha(__file__), design=__doc__, cfg=BASE_CFG, conditions=CONDS, seeds=list(M.FR.S30), priors=M.PRIORS), indent=1) + "\n")
    print("registered")


def check():
    lat = Latent()
    rng = np.random.default_rng(0)
    for _ in range(400): lat.update(rng.random() < .4)
    assert lat.switches <= 1                                                                            # rare false alarm at a steady 40% (0.8 per 2000 episodes measured)
    s0 = lat.switches; lat.dwell = 99; [lat.update(False) for _ in range(40)]; assert lat.switches == s0 + 1   # established context stops paying -> switch
    c1 = lat.c; [lat.update(False) for _ in range(60)]; assert lat.c == c1                              # fresh context is never abandoned
    s1 = lat.switches
    for _ in range(400): lat.update(rng.random() < .9)
    assert lat.switches <= s1 + 1                                                                       # steady 90%: no oscillation
    m = build("human_bc", 1000, True); x = torch.randn(5, 252, device=next(m.net.parameters()).device)
    base = M.build("human_bc", 1000).net
    with torch.no_grad():
        p0 = torch.softmax(base.pi(base.body(x)), -1)
        for c in (0, 1):
            CTX["c"] = c; z = m.net.pi.code(m.net.body(x)); assert not (z[:, ~m.net.pi.mask[c]] > 0).any()
            assert torch.allclose(torch.softmax(m.net.pi(m.net.body(x)), -1), p0, atol=1e-6)               # prior unchanged at start
    L = ReadoutOnly(m.net, BASE_CFG, 0); assert len(L.params) == 2
    print("check ok")


if __name__ == "__main__":
    c = sys.argv[1]
    {"check": check, "prereg": prereg, "analyse": analyse, "run": lambda: run(int(sys.argv[2]), int(sys.argv[3]))}[c]()
