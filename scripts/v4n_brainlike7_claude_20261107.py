"""Brain-constrained circuit, stage 7: a hippocampal goal map (camera-ready, author 2026-11-07: "find a way for our bio plausible
to excel in transfer to unseen rooms ... adapt evidence based bio plausible strategies and prove them working with LEAN before
commiting to them"). GPU 1 only. Registered before any outcome of this stage's learner.
Seen before writing: stages 1-6 dev and test (stage 6 gated test: new 83.7 / old 57.7 / unseen-room new 7.5; PPO unseen new 27.8).
Diagnosis (descriptive, no learner): the agent's own place belief (obs[150:250], the filter every agent receives) peaks on the true
cell at 100% of steps in all 16 fresh rooms, trained and never-seen alike. Every earlier circuit stores the goal as
view -> action habits (striatal-like), which are layout-specific. Mammals store goals on the hippocampal place map
(Tolman 1948; O'Keefe & Nadel 1978; goal over-representation, Hok 2007, Dupret 2010) and read paths out of it by recurrent
propagation / replay (Pfeiffer & Foster 2013; wavefront model, Ponulak & Hopfield 2013).
Ceiling diagnostics ON THE FRESH TEST ROOMS with a SUPPLIED memory (no learning; disclosed): wavefront planner + bump depression
+ inhibition of return with both goals blurred (sigma .5): moved A 240/240, familiar A 240/240, moved B 228/240, familiar B 239/240;
an SR (mean-over-actions) value instead gave 205/177/221/150 (residual blur traps), which is why the wavefront form is used.
Lean (proofs_ml/ProofsML/Wavefront.lean, standard axioms): wave_ascent (away from a goal source the wavefront value always has a
strictly better open neighbour, in any layout) and greedy_reaches_source (climbing it reaches a remembered goal within |S| steps).
Mechanism (added to the stage-6 dev winners, MB parameters fixed):
  goal memory   G over the 100 place cells, written ONLY during learning by a one-shot dopamine-gated Hebbian rule at reward
                (BTSP-like, Bittner 2017): G <- max(G, eta_g K_sigma(place reached)), K = place-field overlap (width sigma); a plateau
                sets a fixed-shape field and repetition does not widen it (pre-registration fix: an additive write saturated the
                field over ~3 cells in the smoke run, 3,936 writes).
                The reached place is the believed place moved by the taken action (self-motion). Frozen in the probes.
  map           place-cell lattice links (innate metric, grid-cell-like); a bump (contact) depresses the believed link for the
                rest of the episode (short-term depression; stp in {0, 1}).
  value         recurrent wavefront v = max(G_e, gamma_h max_open v(next)), 100 sweeps (gamma_h).
  inhibition    at a remembered goal with no reward, G_e <- G_e (1 - K_sigma(place)) for the episode (dopamine dip; ior in {0,1}).
  steering      logits = stage-6 circuit logits + kappa D, D_a = log(q_a / max q) / log(1 / gamma_h) >= -10 (extra steps to the
                goal; q_a = sum_x b(x) v(next(x, a))). Pre-registration fix: a ratio-scaled D (one step = gamma_h - 1) was outvoted
                by habit logits of ~10 in the smoke run (agent bumped one wall 40 times).
  All within-episode state (map, G_e) resets at each episode start; no weights or G change in the probes.
kappa = 0 is stage 6 exactly (check). Search (NSGA-II, 3 objectives as stages 4-6, dev seeds 1000-1002 x 3 priors, exposed rooms):
base in {none, inferred} (stage-6 BEST configs), kappa {0, 2, 5, 10, 20, 50}, sigma {.5, .75, 1, 1.5}, gamma_h {.8, .9, .95},
eta_g {.25, .5, 1}, thr {.3, .5, .7}, stp {0, 1}, ior {0, 1}. Both kappa = 0 bases enqueued first. 2 workers, GPU 1, 40 trials each.
Selection (fixed here): largest dev (new + old + unseen-room new), per base. Test once (30 seeds x 3 priors, fresh rooms, four probes).
Registered ablation arm: the selected config with the larger dev sum, with stp = ior = 0 (goal map only). Contrasts (descriptive):
stage 7 - PPO, - stage 6.
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
import v4n_brainlike6_claude_20261107 as B6
from arc_fly.unimaze import NB
G = B.G; M = B.M

OUT = M.ROOT / "data/audit/v4n_brainlike7_20261107"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"
P4 = B6.P4; LEAN = M.ROOT / "proofs_ml/ProofsML/Wavefront.lean"; OBS = {}
RC = np.divmod(np.arange(100), 10); NXT = np.zeros((100, 4), int)
for _s in range(100):
    for _a, (_dr, _dc) in enumerate(NB):
        _q = (RC[0][_s] + _dr, RC[1][_s] + _dc); NXT[_s, _a] = _q[0] * 10 + _q[1] if 0 <= _q[0] < 10 and 0 <= _q[1] < 10 else _s
OPEN0 = NXT != np.arange(100)[:, None]                                   # lattice links inside the 10 x 10 arena


def kern(c, sigma): return np.exp(-((RC[0] - c // 10) ** 2 + (RC[1] - c % 10) ** 2) / (2 * sigma ** 2))


def wave(open_, Ge, gamma, sweeps=100):
    v = Ge.copy()
    if Ge.max() <= 0: return v
    for _ in range(sweeps):
        nv = np.maximum(Ge, gamma * np.where(open_, v[NXT], 0.).max(1))
        if np.array_equal(nv, v): break
        v = nv
    return v


class Hippo:
    """Goal memory G (learned), per-episode map / inhibited copy, wavefront value, steering drive. numpy state, not parameters."""
    def __init__(self, cfg):
        self.k, self.sig, self.gam, self.eta, self.thr = cfg["kappa"], cfg["sigma"], cfg["gamma_h"], cfg["eta_g"], cfg["thr"]
        self.stp, self.ior = int(cfg["stp"]), int(cfg["ior"]); self.G = np.zeros(100); self.n = 0; self.prev = None; self.writes = 0

    def _reset(self, i): self.open[i] = OPEN0.copy(); self.Ge[i] = self.G.copy(); self.v[i] = wave(self.open[i], self.Ge[i], self.gam)

    def drive(self, obs):
        n = len(obs)
        if n != self.n: self.n = n; self.open, self.Ge, self.v = [None] * n, [None] * n, [None] * n; self.prev = None; [self._reset(i) for i in range(n)]
        if self.prev is not None and np.array_equal(self.prev, obs): return self.D          # same observation seen again (context re-read)
        D = np.zeros((n, 4), np.float32)
        for i, x in enumerate(obs):
            b = x[150:250].astype(np.float64); c = int(b.argmax())
            if x[146:150].sum() == 0: self._reset(i)                                     # episode start (no previous action)
            else:
                ch = False
                if self.stp and x[144] > .5:
                    a = int(x[146:150].argmax())
                    if self.open[i][c, a]: self.open[i][c, a] = False; ch = True
                if self.ior and self.Ge[i].max() > 0 and self.Ge[i][c] > self.thr * self.Ge[i].max():
                    self.Ge[i] = self.Ge[i] * (1 - kern(c, self.sig)); ch = True
                if ch: self.v[i] = wave(self.open[i], self.Ge[i], self.gam)
            v = self.v[i]
            if v.max() > 0:
                q = np.array([b @ np.where(self.open[i][:, a], v[NXT[:, a]], v) for a in range(4)]); D[i] = np.maximum(np.log(np.maximum(q, 1e-12) / q.max()) / -np.log(self.gam), -10.)
        self.prev = obs.copy(); self.D = D; return D

    def learn(self, obs, actions, rewards):
        """Dopamine-gated one-shot Hebbian write of the reached place (believed place + self-motion)."""
        for x, a, r in zip(obs, actions, rewards):
            if r > .5: self.G = np.maximum(self.G, self.eta * kern(int(NXT[int(x[150:250].argmax()), int(a)]), self.sig)); self.writes += 1


class KC7(B6.KC6):
    def forward(self, h):
        out = super().forward(h); hp = getattr(self, "hippo", None)
        if hp is None or not hp.k: return out
        D = hp.drive(OBS["x"].reshape(-1, 252).detach().cpu().numpy())
        return out + hp.k * torch.as_tensor(D, device=out.device).reshape(out.shape)


def build7(prior, seed, cfg):
    m = B6.build6(prior, seed, cfg)
    m.net.body.register_forward_pre_hook(lambda mod, a: OBS.__setitem__("x", a[0]))
    m.net.pi.hippo = Hippo(cfg); return m


def run7(args):
    """B.run_cfg with one added line (goal-memory write after each env step); probes via G.probe (weights verified frozen)."""
    cfg, prior, seed, manifest, out = args; cfg = dict(cfg, O=0); cfg["eta_v"] = cfg["rho"] * (1 - M.GAMMA * cfg["lam"])
    B.KC, B.build, B.Circuit, B.PROTOCOL = KC7, build7, B2.Circuit2, PROTOCOL
    torch.set_num_threads(1); out = Path(out)
    if out.exists(): return json.loads(out.read_text())
    G.MAN = Path(manifest); G.CTX.update(c=0, mode="G2" if cfg["gate"] == "inferred" else "G0")
    pair, stage = M.room_of(seed); gm = M.goal_map(G.MAN); fam = {c: B.FAM for c in gm}; m = B.build(prior, seed, cfg); L = B.Circuit(m.net, cfg)
    hp = m.net.pi.hippo; lat = G.Latent(); gen = torch.Generator().manual_seed(77 + seed); t0 = time.perf_counter(); n_active = []; diverged = None
    for phase, goals, frames in (("familiar", fam, G.PRE), ("moved", gm, M.BUDGET - G.PRE)):
        env = G.Watched(M.env_for(G.MAN, goals, pair, M.N_ENV, seed + (500 if phase == "familiar" else 0), "clean", frames), lat); obs = env.reset(); n = 0
        z, p, v = L.forward(obs); L.ea = None; ep = [[] for _ in range(M.N_ENV)]
        while n < frames:
            a = torch.multinomial(p.cpu(), 1, generator=gen).squeeze(-1); c_now = G.CTX["c"]; o_t = obs
            obs, r, done = env.step(a.numpy()); n += M.N_ENV; c_new = G.CTX["c"]; G.CTX["c"] = c_now
            hp.learn(o_t, a.numpy(), r)                                                   # stage 7: the only added line
            z2, p2, v2 = L.forward(obs); G.CTX["c"] = c_new; L.step(z, p, v, a, r, done, v2)
            if L.c["R"]:
                for i in range(M.N_ENV):
                    ep[i].append((z[i], a[i].to(L.dev), p[i], float(r[i]), float(v[i] - L.Wv[0] @ z[i])))
                    if done[i]:
                        if r[i] > .5: L.replay(ep[i])
                        ep[i] = []
            if c_new != c_now: z2, p2, v2 = L.forward(obs); L.ea = None
            n_active.append(float((z > 0).sum(-1).float().mean())); z, p, v = z2, p2, v2
            if not (torch.isfinite(L.Wa).all() and torch.isfinite(L.Wv).all()): diverged = n; break
        if diverged is not None: break
    res = {}
    for key, goals, st, orc in (("moved_trained_room", gm, stage, 1), ("familiar_trained_room", fam, stage, 0),
                                ("moved_heldout_room", gm, "B", 1), ("familiar_heldout_room", fam, "B", 0)):
        res[key], res[key + "_switches"] = G.probe(m, goals, pair, st, seed, lat, orc) if diverged is None else (0, 0)
    k = m.net.pi.k
    rec = dict(cfg=cfg, prior=prior, seed=seed, pair=pair, **res, diverged=diverged, switches=lat.switches, active_cells=float(np.mean(n_active)),
               synapses=dict(kc_in=cfg["K"] * cfg["c"], kc_out_active=k * 5), goal_memory=hp.G.round(4).tolist(), goal_writes=hp.writes,
               wall_s=time.perf_counter() - t0, protocol_sha256=M.sha(PROTOCOL), gpu_name=torch.cuda.get_device_name(0))
    out.parent.mkdir(parents=True, exist_ok=True); torch.save({kk: vv.cpu() for kk, vv in m.net.state_dict().items()}, out.with_suffix(".final.pt"))
    out.write_text(json.dumps(rec) + "\n"); return rec


def base_cfgs(): return json.loads((B6.OUT / "BEST.json").read_text())["configs"]


def objective(trial):
    s = trial.suggest_categorical; base = s("base", ["none", "inferred"])
    cfg = dict(base_cfgs()[base], kappa=s("kappa", [0., 2., 5., 10., 20., 50.]), sigma=s("sigma", [.5, .75, 1., 1.5]), gamma_h=s("gamma_h", [.8, .9, .95]),
               eta_g=s("eta_g", [.25, .5, 1.]), thr=s("thr", [.3, .5, .7]), stp=s("stp", [0, 1]), ior=s("ior", [0, 1]))
    jobs = [(cfg, p, sd, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}" / p / f"seed{sd}.json")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = list(ex.map(run7, jobs))
    v = [float(np.mean([r[q] for r in recs])) for q in P4[:3]]
    trial.set_user_attr("sum", sum(v)); trial.set_user_attr("gpu", recs[0]["gpu_name"]); return tuple(v)


def search(w, trials):
    import optuna
    assert PROTOCOL.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    st = optuna.create_study(study_name="v4n_brainlike7", storage=DB, directions=["maximize"] * 3, load_if_exists=True,
                             sampler=optuna.samplers.NSGAIISampler(seed=20261120 + w, population_size=20))
    if w == 0 and not st.trials:
        for base in ("none", "inferred"): st.enqueue_trial(dict(base=base, kappa=0., sigma=.5, gamma_h=.9, eta_g=1., thr=.5, stp=1, ior=1))
    else: time.sleep(5)
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_brainlike7", storage=DB); done = [t for t in st.trials if t.values is not None]
    pick = {g: max((t for t in done if t.params["base"] == g), key=lambda t: sum(t.values)) for g in ("none", "inferred") if any(t.params["base"] == g for t in done)}
    cf = {g: dict(base_cfgs()[g], **{k: v for k, v in t.params.items() if k != "base"}) for g, t in pick.items()}
    top = max(pick, key=lambda g: sum(pick[g].values)); cf["ablation"] = dict(cf[top], stp=0, ior=0)
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done), configs=cf,
        numbers={g: t.number for g, t in pick.items()}, ablation_of=top, dev_values={g: t.values for g, t in pick.items()},
        pareto=[dict(n=t.number, v=t.values, p=t.params) for t in st.best_trials]), indent=1) + "\n")
    print(json.dumps({g: (t.number, [round(x * 100 / 30, 1) for x in t.values], t.params) for g, t in pick.items()}), "ablation of", top)


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"; pick = json.loads((OUT / "BEST.json").read_text())["configs"]
    for g, cfg, p, s in [(g, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for g, cfg in sorted(pick.items())][w::n]:
        run7((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / f"bl7_{g}" / p / f"seed{s}.json")))
    print(json.dumps(dict(worker=w, done=True)), flush=True)


def analyse():
    import v4n_kpi_claude_20261031 as K
    from prior_seed_power_claude import contrast
    own = lambda d: {(r["prior"], r["seed"]): r for r in (json.loads(f.read_text()) for f in d.glob("*/seed*.json"))}
    ppo = {}
    for f in (K.OUT / "kpi/ppo").glob("*/seed*.json"):
        r = json.loads(f.read_text()); c = json.loads(Path(r["source"]).read_text())["curve"]; ppo[(r["prior"], r["seed"])] = dict(r, moved_trained_room=c[str(M.BUDGET)])
    res = {}; pick = json.loads((OUT / "BEST.json").read_text())
    for g in ("none", "inferred", "ablation"):
        S = own(OUT / "test" / f"bl7_{g}")
        if not S: continue
        b6 = pick["ablation_of"] if g == "ablation" else g
        refs = {"ppo": ppo, "bl6": own(B6.OUT / "test" / f"bl6_{b6}")}
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
        lean=dict(file="proofs_ml/ProofsML/Wavefront.lean", sha256=M.sha(LEAN)),
        outcomes_seen="stages 1-6 dev and test; supplied-memory planner ceilings on the fresh test rooms (in the design text; no learner); "
                      "of stage 7 only the check runs (kappa = 0 reproduction and kappa = 10 smoke runs, dev rooms, cold_ppo seed 1000, gated base): additive-write smoke new 30, old 23, unseen new 12, unseen familiar 17 of 30 (rule then changed to the max write); max-write ratio-drive smoke 30, 23, 1, 19 with one debug probe of its paths and logits (drive then rescaled to steps); the final check values are in CHECK.json",
        dev_seeds=list(M.DEV_SEEDS), test_seeds=list(M.FR.S30), priors=M.PRIORS), indent=1) + "\n"); print("registered")


def check():
    """kappa = 0 reproduces the stage-6 winners' dev records exactly; the Lean claim holds on random layouts; probe path runs."""
    import tempfile
    rng = np.random.default_rng(0)
    for _ in range(200):                                                 # wave_ascent / greedy_reaches_source, numerically
        op = OPEN0 & (rng.random((100, 4)) > .3); Ge = np.zeros(100); Ge[rng.integers(100)] = 1.; v = wave(op, Ge, .9)
        for x in np.flatnonzero(v > Ge): assert max(v[NXT[x, a]] for a in range(4) if op[x, a]) > v[x]
    b6 = json.loads((B6.OUT / "BEST.json").read_text()); global PROTOCOL
    P0, PROTOCOL = PROTOCOL, Path(tempfile.mkdtemp()) / "P.json"; PROTOCOL.write_text("{}"); recs = {}
    for g in ("none", "inferred"):
        ref = json.loads((B6.OUT / "search" / f"trial{b6['numbers'][g]}" / "cold_ppo" / "seed1000.json").read_text())
        rec = run7((dict(b6["configs"][g], kappa=0., sigma=.5, gamma_h=.9, eta_g=1., thr=.5, stp=1, ior=1), "cold_ppo", 1000, str(M.FR.EXPOSED), str(PROTOCOL.parent / f"{g}.json")))
        assert all(rec[q] == ref[q] for q in P4), (g, [rec[q] for q in P4], [ref[q] for q in P4]); recs[g] = [rec[q] for q in P4]
    rec2 = run7((dict(b6["configs"]["inferred"], kappa=10., sigma=.5, gamma_h=.9, eta_g=1., thr=.5, stp=1, ior=1), "cold_ppo", 1000, str(M.FR.EXPOSED), str(PROTOCOL.parent / "k10.json")))
    PROTOCOL = P0; OUT.mkdir(parents=True, exist_ok=True)
    rep = dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), kappa0=recs, kappa10_smoke=[rec2[q] for q in P4], goal_writes=rec2["goal_writes"],
               goal_memory_peaks=[int(i) for i in np.argsort(rec2["goal_memory"])[-3:]], diverged=rec2["diverged"], wall_s=rec2["wall_s"])
    (OUT / "CHECK.json").write_text(json.dumps(rep, indent=1) + "\n"); print("check ok", rep)


if __name__ == "__main__":
    {"check": check, "prereg": prereg, "best": best, "analyse": analyse, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])),
     "test": lambda: test(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
