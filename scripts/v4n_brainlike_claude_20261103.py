"""Brain-constrained learner: a mushroom-body / cerebellum-like circuit on the frozen prior (camera-ready, author
2026-11-03: "more scientifically similar to how the brain works, given the NeuroAI literature"). GPU 1.
Circuit (every element has a biological counterpart; nothing uses batch statistics, adaptive optimisers or ratios):
  cortex        the frozen prior network (body + its policy/value heads): the innate / previously learnt mapping
  granule/KC    K cells, each wired to c random hidden units (sparse, signed), global inhibition keeps the top fraction f
                active (APL / Golgi-like k-WTA) [Litwin-Kumar et al. 2017; Shen, Dasgupta & Navlakha 2023]
  output        KC -> action (MBON / Purkinje-like) and KC -> value synapses, zero at start (the prior is unchanged)
  dopamine      TD error delta = r + gamma V(s') - V(s) broadcast to all output synapses [Schultz 1997]
  plasticity    dW = eta_ij * delta * e_ij, e_ij = gamma lambda e_ij + z_j (onehot(a) - pi)_i: three-factor rule with an
                eligibility trace [Fremaux & Gerstner 2016; Gerstner et al. 2018]; only synapses of ACTIVE cells change
  metaplasticity eta_ij = eta / (1 + m * M_ij), M_ij = accumulated |dW| (synapses that changed a lot become stable;
                Benna & Fusi 2016; Laborieux et al. 2021)
  exploration   while learning, actions are drawn from softmax(logits / tau), tau >= 1 (tonic noradrenergic exploration;
                Aston-Jones & Cohen 2005); the frozen probes act at tau = 1. The eligibility uses the behaving policy.
  normalisation the active cells' rates are divisively normalised to unit length (global inhibition; Carandini & Heeger 2012)
  replay        after a rewarded episode, its own steps are replayed once in reverse (reverse hippocampal replay;
                Foster & Wilson 2006), same rule, R times
  context       optional: the latent-cause switch of v4n_context_gating (G2; Gershman et al. 2010) selects one of two KC
                halves (compartment / remapping analogue); no cue
Search: NSGA-II over (moved-goal final, familiar-goal final) on the dev rooms (3 dev seeds x 3 priors), as the
stability search. Space: K {1024, 2048, 4096}, c {4, 7, 12}, f [.02, .2], actor eta [.1, 300] log, value eta_v [.003, .3] log (separate synapse populations; TD stability needs
eta_v (1 - gamma lambda)^-1 < ~1 for unit-norm codes), tau [1, 5] log, lambda [.5, .99], m {0, 1, 10, 100}, R {0, 1, 2}, gate {none, inferred}. Budget and schedule as the gating
study: 2,048 familiar-goal interactions, then 30,720 moved-goal (32,768 in all).
Selection (fixed here): the trial with the largest dev sum, per gate setting; each tested once on 30 seeds x 3 priors
(fresh rooms), with the four probes (moved/familiar x trained/unseen room); spiking cost counted per decision.
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

OUT = M.ROOT / "data/audit/v4n_brainlike_20261103"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"; FAM = G.FAM


class KC(G.Gated):
    """Sparse wiring: each of K cells sees c hidden units."""
    def __init__(self, base, H, cfg, seed, gated):
        g = torch.Generator().manual_seed(9000 + seed); K, c = cfg["K"], cfg["c"]; P = torch.zeros(H, K)
        for j in range(K): P[torch.randperm(H, generator=g)[:c], j] = torch.randn(c, generator=g) / np.sqrt(c)
        super().__init__(base, P.to(base.weight.device), max(1, int(round(cfg["f"] * K))), gated)

    def code(self, h):                                                       # divisive normalisation of the active cells (APL-like)
        z = super().code(h); return z / (z.norm(dim=-1, keepdim=True) + 1e-6)


def build(prior, seed, cfg):
    m = M.build(prior, seed); net = m.net; dev = net.pi.weight.device; H = net.pi.in_features; gated = cfg["gate"] == "inferred"
    net.pi = KC(net.pi, H, cfg, seed, gated).to(dev); net.v = KC(net.v, H, cfg, seed, gated).to(dev); net.v.P = net.pi.P; return m


class Circuit:
    def __init__(self, net, cfg):
        self.net, self.c = net, cfg; self.Wa, self.Wv = net.pi.R.weight, net.v.R.weight; self.dev = self.Wa.device
        self.Ma, self.Mv = torch.zeros_like(self.Wa), torch.zeros_like(self.Wv); self.ea = self.ev = None; self.ep = {}

    @torch.no_grad()
    def forward(self, obs):
        h = self.net.body(torch.as_tensor(obs, device=self.dev)); z = self.net.pi.code(h)
        return z, torch.softmax(self.net.pi(h) / self.c["tau"], -1), self.net.v(h).squeeze(-1)

    @torch.no_grad()
    def apply(self, da, dv):
        c = self.c; ra = c["eta"] / (1 + c["m"] * self.Ma); rv = c["eta_v"] / (1 + c["m"] * self.Mv)
        ua, uv = ra * da, rv * dv; self.Wa += ua; self.Wv += uv; self.Ma += ua.abs(); self.Mv += uv.abs()

    @torch.no_grad()
    def step(self, z, p, v, a, r, done, v_next):
        """Online three-factor update for n parallel animals (traces per animal, synapse change averaged over them)."""
        g = torch.nn.functional.one_hot(a.to(self.dev), 4).float() - p; keep = 1. - torch.as_tensor(done, device=self.dev, dtype=torch.float32)
        if self.ea is None: self.ea, self.ev, self.kp = torch.zeros(len(z), 4, z.shape[1], device=self.dev), torch.zeros(len(z), z.shape[1], device=self.dev), torch.ones(len(z), device=self.dev)
        gl = M.GAMMA * self.c["lam"]
        self.ea = gl * self.ea * self.kp[:, None, None] + g[:, :, None] * z[:, None, :]; self.ev = gl * self.ev * self.kp[:, None] + z; self.kp = keep
        delta = torch.as_tensor(r, device=self.dev, dtype=torch.float32) + M.GAMMA * v_next * keep - v
        self.apply((delta[:, None, None] * self.ea).mean(0), (delta[:, None] * self.ev).mean(0)[None])
        return delta

    @torch.no_grad()
    def replay(self, steps):
        """Reverse replay of one rewarded episode: (z, a, p, r) from end to start, values recomputed from the current synapses."""
        for _ in range(self.c["R"]):
            ea = torch.zeros(4, steps[0][0].shape[0], device=self.dev); ev = torch.zeros(steps[0][0].shape[0], device=self.dev); v_next = 0.
            for z, a, p, r, vb in reversed(steps):
                v = float(vb + self.Wv[0] @ z); delta = r + M.GAMMA * v_next - v; v_next = v
                ea = M.GAMMA * self.c["lam"] * ea + (torch.nn.functional.one_hot(a, 4).float() - p)[:, None] * z[None]; ev = M.GAMMA * self.c["lam"] * ev + z
                self.apply(delta * ea, delta * ev[None])


def run_cfg(args):
    cfg, prior, seed, manifest, out = args
    torch.set_num_threads(1); out = Path(out)
    if out.exists(): return json.loads(out.read_text())
    G.MAN = Path(manifest); G.CTX.update(c=0, mode="G2" if cfg["gate"] == "inferred" else "G0")
    pair, stage = M.room_of(seed); gm = M.goal_map(G.MAN); fam = {c: FAM for c in gm}; m = build(prior, seed, cfg); L = Circuit(m.net, cfg)
    lat = G.Latent(); gen = torch.Generator().manual_seed(77 + seed); t0 = time.perf_counter(); n_active = []; diverged = None
    for phase, goals, frames in (("familiar", fam, G.PRE), ("moved", gm, M.BUDGET - G.PRE)):
        env = G.Watched(M.env_for(G.MAN, goals, pair, M.N_ENV, seed + (500 if phase == "familiar" else 0), "clean", frames), lat); obs = env.reset(); n = 0
        z, p, v = L.forward(obs); L.ea = None; ep = [[] for _ in range(M.N_ENV)]
        while n < frames:
            a = torch.multinomial(p.cpu(), 1, generator=gen).squeeze(-1); c_now = G.CTX["c"]
            obs, r, done = env.step(a.numpy()); n += M.N_ENV; c_new = G.CTX["c"]; G.CTX["c"] = c_now
            z2, p2, v2 = L.forward(obs); G.CTX["c"] = c_new; L.step(z, p, v, a, r, done, v2)
            if L.c["R"]:
                for i in range(M.N_ENV):
                    ep[i].append((z[i], a[i].to(L.dev), p[i], float(r[i]), float(v[i] - L.Wv[0] @ z[i])))          # prior (cortical) value part
                    if done[i]:
                        if r[i] > .5: L.replay(ep[i])
                        ep[i] = []
            if c_new != c_now: z2, p2, v2 = L.forward(obs); L.ea = None                  # new context: fresh code and traces
            n_active.append(float((z > 0).sum(-1).float().mean())); z, p, v = z2, p2, v2
            if not (torch.isfinite(L.Wa).all() and torch.isfinite(L.Wv).all()): diverged = n; break   # divergence = failure (as the learning study)
        if diverged is not None: break
    res = {}
    for key, goals, st, orc in (("moved_trained_room", gm, stage, 1), ("familiar_trained_room", fam, stage, 0),
                                ("moved_heldout_room", gm, "B", 1), ("familiar_heldout_room", fam, "B", 0)):
        res[key], res[key + "_switches"] = G.probe(m, goals, pair, st, seed, lat, orc) if diverged is None else (0, 0)
    k = m.net.pi.k
    rec = dict(cfg=cfg, prior=prior, seed=seed, pair=pair, **res, diverged=diverged, switches=lat.switches, active_cells=float(np.mean(n_active)),
               synapses=dict(kc_in=cfg["K"] * cfg["c"], kc_out_active=k * 5), wall_s=time.perf_counter() - t0,
               protocol_sha256=M.sha(PROTOCOL), gpu_name=torch.cuda.get_device_name(0))
    out.parent.mkdir(parents=True, exist_ok=True); torch.save({kk: vv.cpu() for kk, vv in m.net.state_dict().items()}, out.with_suffix(".final.pt"))
    out.write_text(json.dumps(rec) + "\n"); return rec


def objective(trial):
    s = trial.suggest_categorical
    cfg = dict(K=s("K", [1024, 2048, 4096]), c=s("c", [4, 7, 12]), f=trial.suggest_float("f", .02, .2, log=True), eta=trial.suggest_float("eta", .1, 300., log=True), tau=trial.suggest_float("tau", 1., 5., log=True),
               lam=trial.suggest_float("lam", .5, .99), eta_v=trial.suggest_float("eta_v", .003, .3, log=True), m=s("m", [0., 1., 10., 100.]),
               R=s("R", [0, 1, 2]), gate=s("gate", ["none", "inferred"]))
    jobs = [(cfg, p, sd, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}" / p / f"seed{sd}.json")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = list(ex.map(run_cfg, jobs))
    new = float(np.mean([r["moved_trained_room"] for r in recs])); old = float(np.mean([r["familiar_trained_room"] for r in recs]))
    trial.set_user_attr("sum", new + old); return new, old


def search(w, trials):
    import optuna
    st = optuna.create_study(study_name="v4n_brainlike", storage=DB, directions=["maximize", "maximize"], load_if_exists=True,
                             sampler=optuna.samplers.NSGAIISampler(seed=20261103 + w, population_size=24))
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_brainlike", storage=DB); done = [t for t in st.trials if t.values is not None]
    pick = {g: max((t for t in done if t.params["gate"] == g), key=lambda t: sum(t.values)) for g in ("none", "inferred") if any(t.params["gate"] == g for t in done)}
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done),
        configs={g: t.params for g, t in pick.items()}, dev_values={g: t.values for g, t in pick.items()}, pareto=[dict(n=t.number, v=t.values) for t in st.best_trials]), indent=1) + "\n")
    print(json.dumps({g: (t.number, t.values, t.params) for g, t in pick.items()}))


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1"); pick = json.loads((OUT / "BEST.json").read_text())["configs"]
    for g, cfg, p, s in [(g, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for g, cfg in sorted(pick.items())][w::n]:
        run_cfg((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / f"bl_{g}" / p / f"seed{s}.json")))
    print(json.dumps(dict(worker=w, done=True)), flush=True)


def analyse():
    """Descriptive (no primary was registered): four probes per gate setting, paired (prior, seed) contrasts vs PPO,
    the inferred-gate expansion learner (G2) and between gate settings; human_bc - other priors on each probe."""
    from prior_seed_power_claude import contrast
    import v4n_kpi_claude_20261031 as K
    P = ("moved_trained_room", "familiar_trained_room", "moved_heldout_room", "familiar_heldout_room")
    def own(d): return {(r["prior"], r["seed"]): r for r in (json.loads(f.read_text()) for f in d.glob("*/seed*.json") if f.stat().st_size)}
    def ppo():
        out = {}
        for f in (K.OUT / "kpi/ppo").glob("*/seed*.json"):
            r = json.loads(f.read_text()); c = json.loads(Path(r["source"]).read_text())["curve"]; out[(r["prior"], r["seed"])] = dict(r, moved_trained_room=c[str(M.BUDGET)])
        return out
    refs = {"ppo": ppo(), "G2": own(G.OUT / "runs/G2")}; res = {}
    for g in ("none", "inferred"):
        S = own(OUT / "test" / f"bl_{g}"); R = dict(n=len(S), means={q: float(np.mean([r[q] for r in S.values()]) * 100 / 30) for q in P},
                   synapses=float(np.mean([r["synapses"]["kc_in"] + r["synapses"]["kc_out_active"] for r in S.values()])), active=float(np.mean([r["active_cells"] for r in S.values()])))
        for o, O in list(refs.items()) + ([("bl_none", own(OUT / "test/bl_none"))] if g == "inferred" else []):
            for q in P:
                ks = sorted(set(S) & set(O)); R[f"{q}/{o}"] = contrast({"a": {k: S[k][q] * 100 / 30 for k in ks}, "b": {k: O[k][q] * 100 / 30 for k in ks}}, "a", "b", 10.)
        for b in ("human_legal_uniform", "cold_ppo"):
            for q in P[:2]: R[f"prior/{q}/human_bc-{b}"] = contrast({p: {k[1]: S[k][q] * 100 / 30 for k in S if k[0] == p} for p in ("human_bc", b)}, "human_bc", b, 10.)
        res[g] = R; print(g, R["n"], {q[:12]: round(v, 1) for q, v in R["means"].items()}, "synapses", round(R["synapses"]))
        for q, v in R.items():
            if "/" in q: print("  ", q, round(v["mean"], 1), [round(x, 1) for x in v["ci95"]], "p=%.2g" % v["t_p"])
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1) + "\n")


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), outcomes_seen="search2, stability (partial), gating (partial) outcomes; dev-room pilots of this circuit (cold_ppo/1000; used to fix divergence, add exploration gain and separate value rate)",
        script_sha256=M.sha(__file__), design=__doc__, trials="96 NSGA-II trials (2 workers, population 24), dev only"), indent=1) + "\n"); print("registered")


def check():
    cfg = dict(K=256, c=7, f=.1, eta=.1, eta_v=.05, tau=1., lam=.9, m=10., R=1, gate="inferred")
    m = build("human_bc", 1000, cfg); base = M.build("human_bc", 1000).net; x = torch.randn(6, 252, device=next(m.net.parameters()).device)
    assert int((m.net.pi.P != 0).sum(0).max()) == 7                                                       # sparse wiring
    with torch.no_grad(): zz = m.net.pi.code(m.net.body(x)); assert torch.allclose(zz.norm(dim=-1), torch.ones(6, device=zz.device), atol=1e-4)
    with torch.no_grad(): assert torch.allclose(torch.softmax(m.net.pi(m.net.body(x)), -1), torch.softmax(base.pi(base.body(x)), -1), atol=1e-6)
    L = Circuit(m.net, cfg); z, p, v = L.forward(x.cpu().numpy()); a = torch.tensor([0, 1, 2, 3, 0, 1])
    W0 = L.Wa.clone(); L.step(z, p, v, a, np.ones(6), np.zeros(6, bool), torch.zeros(6, device=L.dev))
    moved = (L.Wa - W0).abs().sum(0) > 0; assert torch.equal(moved, (z > 0).any(0))                       # only synapses of active cells change
    L.replay([(z[i], a[i].to(L.dev), p[i], 1. if i == 5 else 0., float(v[i])) for i in range(6)]); assert torch.isfinite(L.Wa).all()
    print("check ok")


if __name__ == "__main__":
    c = sys.argv[1]
    {"check": check, "prereg": prereg, "best": best, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])), "test": lambda: test(int(sys.argv[2]), int(sys.argv[3])), "analyse": analyse}[c]()
