"""Second, larger search for biologically plausible learners on the moved-goal task (camera-ready; author 2026-11-02:
"do more optuna/nas/other optimization for the bio-plausible to reach even higher performance"). GPU 1.
Same task, budget (32,768 interactions, 16 envs), priors, probes and DEV/TEST split as v4n_local_search (v1, 30 trials).
New ingredients, all keeping every weight change local (third factor x presynaptic x postsynaptic, no weight transport):
  replay   hippocampal-style offline replay of the last T steps (16 envs x T), E passes in M minibatches; the third factor
           is the GAE(lambda) advantage from the learnt critic [Schulman 2016], optionally PPO-style clipped (a per-sample
           scalar gate: adv x ratio, zeroed outside [1-eps, 1+eps] in the improving direction) [Schulman 2017]
  expand   cerebellum-like fixed random expansion of the last hidden layer (K units, k-winners-take-all sparsity) with a
           plastic, zero-initialised readout into both heads [Marr 1969; Albus 1971; Litwin-Kumar et al. 2017]
  credit   hidden layers {none (head), last, all} via fixed random feedback (DFA, Nokland 2016) or ED modulo routing
           (Yamada et al. 2026, routing only); critic error trains the heads only
  rule     per-synapse Adam (local state), optional linear rate decay, entropy bonus, advantage normalisation
NAS x Optuna TPE over all of the above, 3 dev seeds x 3 priors per trial in the exposed rooms (never the test rooms).
Selection, test and contrasts as v1: best trial per hidden-credit family (dfa, ed; plastic != head) and best head-only
trial, each tested once on 30 seeds x 3 priors in the fresh confirmation rooms.
  check | prereg | search W TRIALS | best | test W N | kpi W N | analyse
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_moved_goal_learning_claude_20261031 as M

OUT = M.ROOT / "data/audit/v4n_local_search2_20261102"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"
SUF = "_" + os.environ["S2_PLASTIC"] if os.environ.get("S2_PLASTIC") else ""
FAMILIES = ("dfa", "ed"); NAMES = {"dfa": "s2_dfa", "ed": "s2_ed", "head": "s2_head" + ("_strat" if SUF else "")}


class Expanded(torch.nn.Module):
    """base(h) + R z(h); z = k-WTA of a fixed random projection (granule-like). R starts at zero: the prior is unchanged."""
    def __init__(self, base, P, k):
        super().__init__(); self.base, self.k = base, k
        self.register_buffer("P", P); self.R = torch.nn.Linear(P.shape[1], base.out_features, bias=False); torch.nn.init.zeros_(self.R.weight)

    def code(self, h):
        z = h @ self.P; top = torch.topk(z, self.k, -1)
        return torch.zeros_like(z).scatter(-1, top.indices, torch.relu(top.values))

    def forward(self, h): return self.base(h) + self.R(self.code(h))

    @property
    def out_features(self): return self.base.out_features


def expand(net, K, frac, seed):
    """In place: wrap pi and v (same fixed expansion code shared by both heads)."""
    g = torch.Generator().manual_seed(9000 + seed); dev = net.pi.weight.device; H = net.pi.in_features
    P = (torch.randn(H, K, generator=g) / np.sqrt(H)).to(dev); k = max(1, int(round(frac * K)))
    net.pi = Expanded(net.pi, P, k).to(dev); net.v = Expanded(net.v, P, k).to(dev)


class ReplayLocal:
    """Buffered local actor-critic. Weight changes per synapse = (per-sample third factor) x (pre) x (post feedback)."""
    def __init__(self, net, cfg, seed):
        self.net, self.c = net, cfg; dev = net.pi.weight.device if not isinstance(net.pi, Expanded) else net.pi.base.weight.device
        self.dev = dev; self.L = [m for m in net.body if isinstance(m, torch.nn.Linear)]; A = 4
        self.hid = {"head": [], "last": [len(self.L) - 1], "all": list(range(len(self.L)))}[cfg["plastic"]]
        g = torch.Generator().manual_seed(4242 + seed)
        if cfg["feedback"] == "ed":
            self.B = [(torch.arange(l.out_features)[:, None] % A == torch.arange(A)[None]).float().to(dev) * cfg["fb_scale"] for l in self.L]
        else:
            self.B = [torch.randn(l.out_features, A, generator=g).to(dev) / np.sqrt(A) * cfg["fb_scale"] for l in self.L]
        self.X = isinstance(net.pi, Expanded); pi, v = (net.pi.base, net.v.base) if self.X else (net.pi, net.v)
        self.params = [pi.weight, pi.bias, v.weight, v.bias] + ([net.pi.R.weight, net.v.R.weight] if self.X else []) + \
                      [q for i in self.hid for q in (self.L[i].weight, self.L[i].bias)]
        self.m = [torch.zeros_like(q) for q in self.params]; self.s2 = [torch.zeros_like(q) for q in self.params]; self.t = 0

    @torch.no_grad()
    def forward(self, x):
        hs = [torch.as_tensor(x, device=self.dev)]
        for l in self.L: hs.append(torch.tanh(l(hs[-1])))
        z = self.net.pi.code(hs[-1]) if self.X else None
        return hs, z, torch.softmax(self.net.pi(hs[-1]), -1), self.net.v(hs[-1]).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, gen):
        hs, z, p, v = self.forward(obs); a = torch.multinomial(p.cpu(), 1, generator=gen).squeeze(-1)
        return a, p, v

    @torch.no_grad()
    def dirs(self, hs, z, gout, verr):
        """gout (n,4): per-sample modulated output error; verr (n,): critic error. Batch-averaged local outer products."""
        n = len(verr); h = hs[-1]
        out = [gout.T @ h / n, gout.mean(0), verr[None] @ h / n, verr.mean(0, keepdim=True)]
        if self.X: out += [gout.T @ z / n, verr[None] @ z / n]
        for i in self.hid:
            d = (gout @ self.B[i].T) * (1 - hs[i + 1] ** 2); out += [d.T @ hs[i] / n, d.mean(0)]
        return out

    @torch.no_grad()
    def learn(self, buf, v_last, frac_done):
        c = self.c; obs, act, rew, done, logp0 = (torch.stack(x) for x in zip(*buf))     # (T, n, ...)
        T, n = rew.shape; hs, z, p, v = self.forward(obs.reshape(T * n, -1)); v = v.view(T, n)
        adv = torch.zeros(T, n, device=self.dev); last = torch.zeros(n, device=self.dev); nxt = v_last
        for t in reversed(range(T)):                                          # GAE(lambda), episode ends cut the trace
            keep = 1. - done[t]; d = rew[t] + M.GAMMA * nxt * keep - v[t]; last = d + M.GAMMA * c["gae"] * keep * last; adv[t] = last; nxt = v[t]
        ret = (adv + v).reshape(-1); adv = adv.reshape(-1)
        if c["adv_norm"]: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs, act, logp0 = obs.reshape(T * n, -1), act.reshape(-1), logp0.reshape(-1)
        rate = c["rate"] * ((1 - frac_done) if c["decay"] else 1.)
        for _ in range(c["epochs"]):
            for idx in torch.randperm(T * n, device=self.dev).chunk(c["minibatches"]):
                hs, z, p, v = self.forward(obs[idx]); a = act[idx]; A = adv[idx]
                ratio = (torch.log(p.gather(1, a[:, None]).squeeze(1).clamp_min(1e-8)) - logp0[idx]).exp()
                f = A * ratio
                if c["clip"] > 0: f = f * ~(((A > 0) & (ratio > 1 + c["clip"])) | ((A < 0) & (ratio < 1 - c["clip"])))
                logp = torch.log(p.clamp_min(1e-8)); H = -(p * logp).sum(-1, keepdim=True)
                g = f[:, None] * (torch.nn.functional.one_hot(a, 4).float() - p) + c["ent"] * (-p * (logp + H))
                ds = self.dirs(hs, z, g, (ret[idx] - v) * c["critic_ratio"])
                self.t += 1
                for q, d, m, s2 in zip(self.params, ds, self.m, self.s2):
                    d = d.view_as(q); m.mul_(.9).add_(.1 * d); s2.mul_(.999).add_(.001 * d * d)
                    q += rate * (m / (1 - .9 ** self.t)) / ((s2 / (1 - .999 ** self.t)).sqrt() + 1e-5)


def run_cfg(args):
    """One learning run (same env, probes and record as M.run_one; learner replaced)."""
    cfg, prior, seed, manifest, out = args
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1"); torch.set_num_threads(1); manifest, out = Path(manifest), Path(out)
    if out.exists(): return json.loads(out.read_text())
    pair, stage = M.room_of(seed); gm = M.goal_map(manifest); m = M.build(prior, seed); t0 = time.perf_counter(); diverged = None
    if cfg["expand"]: expand(m.net, cfg["expand"], cfg["kwta"], seed)
    env = M.env_for(manifest, gm, pair, M.N_ENV, seed, "clean", M.BUDGET); obs = env.reset()
    curve = {0: M.probe_now(m, manifest, gm, pair, stage, seed)}
    L = ReplayLocal(m.net, cfg, seed); gen = torch.Generator().manual_seed(77 + seed); frames = 0; buf = []
    while frames < M.BUDGET:
        a, p, v = L.act(obs, gen); o0 = torch.as_tensor(obs, device=L.dev)
        obs, r, done = env.step(a.numpy()); frames += M.N_ENV
        buf.append((o0, a.to(L.dev), torch.as_tensor(r, device=L.dev, dtype=torch.float32), torch.as_tensor(done, device=L.dev, dtype=torch.float32),
                    torch.log(p.gather(1, a.to(L.dev)[:, None]).squeeze(1).clamp_min(1e-8))))
        if len(buf) == cfg["T"] or frames >= M.BUDGET:
            L.learn(buf, L.forward(obs)[3], frames / M.BUDGET); buf = []
            if not all(torch.isfinite(q).all() for q in m.net.parameters()): diverged = frames; break
        if frames in M.CURVE: curve[frames] = M.probe_now(m, manifest, gm, pair, stage, seed)
    for x in M.CURVE: curve.setdefault(x, dict(goals=0, episodes=[]))
    rec = dict(learner="search2", cfg=cfg, prior=prior, seed=seed, pair=pair, room=stage, goal=list(gm[100 + 2 * pair]), budget=M.BUDGET,
               curve={k: v["goals"] for k, v in curve.items()}, diverged_at=diverged, final_episodes=curve[M.BUDGET]["episodes"],
               wall_s=time.perf_counter() - t0, protocol_sha256=M.sha(PROTOCOL), gpu_name=torch.cuda.get_device_name(0))
    out.parent.mkdir(parents=True, exist_ok=True); torch.save({k: v.cpu() for k, v in m.net.state_dict().items()}, out.with_suffix(".final.pt"))
    out.write_text(json.dumps(rec) + "\n"); return rec


def objective(trial):
    s = trial.suggest_categorical
    fix = os.environ.get("S2_PLASTIC")                                       # amendment 1: family-stratified study
    cfg = dict(plastic=s("plastic", [fix] if fix else ["head", "last", "all"]), feedback=s("feedback", list(FAMILIES)), expand=s("expand", [0, 512, 2048]),
               kwta=trial.suggest_float("kwta", .02, .3, log=True), T=s("T", [8, 32, 128]), epochs=s("epochs", [1, 2, 4, 8]),
               minibatches=s("minibatches", [1, 4]), clip=s("clip", [0., .1, .2, .3]), gae=trial.suggest_float("gae", .5, .99),
               ent=trial.suggest_float("ent", 0., .1), critic_ratio=trial.suggest_float("critic_ratio", .3, 3., log=True),
               fb_scale=trial.suggest_float("fb_scale", .1, 3., log=True), rate=trial.suggest_float("rate", 1e-5, 1e-2, log=True),
               adv_norm=s("adv_norm", [True, False]), decay=s("decay", [True, False]))
    jobs = [(cfg, p, sd, str(M.FR.EXPOSED), str(OUT / ("search" + SUF) / f"trial{trial.number}" / p / f"seed{sd}.json")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = list(ex.map(run_cfg, jobs))
    cur = lambda r, x: r["curve"][str(x)] if str(x) in r["curve"] else r["curve"][x]
    xs = np.array(M.CURVE, float); final = float(np.mean([cur(r, M.BUDGET) for r in recs]))
    auc = float(np.mean([np.trapezoid([cur(r, int(x)) for x in xs], xs) / (30 * M.BUDGET) for r in recs]))
    trial.set_user_attr("final", final); trial.set_user_attr("auc", auc); return final + auc       # as v1


def search(w, trials):
    import optuna
    st = optuna.create_study(study_name="v4n_local2" + SUF, storage=DB, direction="maximize", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=20261102 + w, n_startup_trials=12, multivariate=True, group=True))
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_local2" + SUF, storage=DB); done = [t for t in st.trials if t.value is not None]
    pick = {}
    for fam in FAMILIES:
        tt = [t for t in done if t.params["feedback"] == fam and t.params["plastic"] != "head"]
        if tt: pick[fam] = max(tt, key=lambda t: t.value)
    hh = [t for t in done if t.params["plastic"] == "head"]
    if hh: pick["head"] = max(hh, key=lambda t: t.value)
    (OUT / f"BEST{SUF}.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done),
        configs={k: t.params for k, t in pick.items()}, trial_numbers={k: t.number for k, t in pick.items()},
        dev_scores={k: t.value for k, t in pick.items()}, dev_final={k: t.user_attrs["final"] for k, t in pick.items()}), indent=1) + "\n")
    print(json.dumps({k: (t.number, round(t.value, 2), t.params) for k, t in pick.items()}, indent=1))


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    pick = json.loads((OUT / f"BEST{SUF}.json").read_text())["configs"]
    jobs = [(name, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for name, cfg in sorted(pick.items())][w::n]
    for name, cfg, p, s in jobs: run_cfg((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / NAMES[name] / p / f"seed{s}.json")))
    print(json.dumps(dict(worker=w, done=len(jobs))), flush=True)


def kpi(w, n):
    """The three frozen KPI probes of v4n_kpi on the tested networks (expansion rebuilt from the config, then loaded)."""
    import v4n_kpi_claude_20261031 as K
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    for f in sorted((OUT / "test").glob("*/*/seed*.json"))[w::n]:
        r = json.loads(f.read_text()); name = f.parts[-3]; out = K.OUT / "kpi" / name / r["prior"] / f.name
        if out.exists(): continue
        m = M.build(r["prior"], r["seed"])
        if r["cfg"]["expand"]: expand(m.net, r["cfg"]["expand"], r["cfg"]["kwta"], r["seed"])
        m.net.load_state_dict({k: v.to(next(m.net.parameters()).device) for k, v in torch.load(f.with_suffix(".final.pt")).items()})
        pair = r["seed"] % 8; mg = M.goal_map(M.FR.MAN); fam = {c: K.FAM for c in mg}
        rec = dict(learner=name, prior=r["prior"], seed=r["seed"], source=str(f), familiar_trained_room=K.probe(m, fam, pair, "A", r["seed"]),
                   moved_heldout_room=K.probe(m, mg, pair, "B", r["seed"]), familiar_heldout_room=K.probe(m, fam, pair, "B", r["seed"]))
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec) + "\n")
    print(json.dumps(dict(stage="kpi", worker=w, done=True)), flush=True)


def analyse():
    from prior_seed_power_claude import contrast
    from analyse_seeds30_claude import holm
    xs = np.array(M.CURVE, float); auc = lambda c: float(np.trapezoid([c[str(int(x))] for x in xs], xs) / (30 * M.BUDGET)) * 100
    res, fam = dict(best=json.loads((OUT / "BEST.json").read_text())), {}
    ref = {"ppo": M.OUT / "test2/ppo/registered", "v1_head": M.ROOT / "data/audit/v4n_local_search_20261031/test3/head",
           "v1_dfa": M.ROOT / "data/audit/v4n_local_search_20261031/test3/dfa", "v1_ed": M.ROOT / "data/audit/v4n_local_search_20261031/test3/ed"}
    for name, d in {**{v: OUT / "test" / v for v in NAMES.values()}, **ref}.items():
        S = {"auc": {}, "final": {}}
        for p in M.PRIORS:
            for k in S: S[k][p] = {}
            for f in sorted((d / p).glob("seed*.json")):
                r = json.loads(f.read_text()); S["auc"][p][r["seed"]] = auc(r["curve"]); S["final"][p][r["seed"]] = r["curve"][str(M.BUDGET)] * 100 / 30
        R = dict(means={k: {p: float(np.mean(list(v[p].values()))) for p in M.PRIORS if v[p]} for k, v in S.items()}, n={p: len(S["auc"][p]) for p in M.PRIORS})
        for k in ("auc", "final"):
            for b in ("human_legal_uniform", "cold_ppo"):
                if len(S[k]["human_bc"]) > 2 and len(S[k][b]) > 2: R[f"{k}/human_bc-{b}"] = contrast(S[k], "human_bc", b, 10.)
        if name.startswith("s2_"):
            for b in ("human_legal_uniform", "cold_ppo"): fam[f"{name}/{b}"] = R[f"auc/human_bc-{b}"]
            for other in ("ppo", f"v1_{name[3:]}"):                         # secondary: gain over v1 and gap to PPO, all seeds x priors
                o = {"final": {}}
                for p in M.PRIORS:
                    for f in sorted((ref[other] / p).glob("seed*.json")):
                        r = json.loads(f.read_text()); o["final"][(p, r["seed"])] = r["curve"][str(M.BUDGET)] * 100 / 30
                keys = sorted(set(o["final"]) & {(p, s) for p in M.PRIORS for s in S["final"][p]})
                if len(keys) > 2: R[f"final/{name}-{other}"] = contrast({"a": {k: S["final"][k[0]][k[1]] for k in keys}, "b": o["final"]}, "a", "b", 10.)
        res[name] = R
    for k, h in zip(fam, holm([v["t_p"] for v in fam.values()])): fam[k]["holm_t_p"] = h
    res["primary_auc"] = fam; (OUT / "analysis.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    for k, v in res.items():
        if isinstance(v, dict) and "means" in v: print(k, v["n"], {p: round(x, 1) for p, x in v["means"]["final"].items()},
                                                    {q: [round(v[q]["mean"], 1), [round(x, 1) for x in v[q]["ci95"]]] for q in v if q.startswith("final/s2")})


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), outcomes_seen="no test outcome of this search; v1 dev and test seen",
        script_sha256=M.sha(__file__), design=__doc__, trials="160 TPE trials (2 parallel workers), dev only; stop early only for time, the number run is reported",
        objective="mean final dev success (/30) over 3 priors x 3 dev seeds + AUC tie-break (as v1)",
        selection="best trial per hidden-credit family (dfa, ed; plastic != head) and best head-only trial",
        test="30 seeds x 3 priors, fresh confirmation rooms, as v4n_moved_goal_learning; then the three frozen KPI probes (as v4n_kpi)",
        primary="per selected learner: human_bc - human_legal_uniform and human_bc - cold_ppo learning-curve AUC; Holm over the 6",
        secondary="final success vs the matching v1 learner and vs PPO (paired by prior x seed); KPI probes; descriptive",
        caveat="learners start from backprop-trained priors; buffered replay is offline reuse of experience, not online; ED = routing only"), indent=1) + "\n")
    print("registered")


def check():
    torch.manual_seed(0); net = torch.nn.Module(); net.body = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8), torch.nn.Tanh())
    net.pi = torch.nn.Linear(8, 4); net.v = torch.nn.Linear(8, 1); x = torch.randn(6, 5); p0 = torch.softmax(net.pi(net.body(x)), -1)
    expand(net, 32, .1, 0); assert torch.allclose(p0, torch.softmax(net.pi(net.body(x)), -1))           # zero-init: prior unchanged
    assert int((net.pi.code(net.body(x)) > 0).sum(-1).max()) <= 3
    for plastic in ("head", "last", "all"):
        cfg = dict(plastic=plastic, feedback="dfa", rate=1e-2, gae=.9, ent=.01, critic_ratio=1., fb_scale=1., epochs=2, minibatches=2, clip=.2, adv_norm=True, decay=True, T=4)
        L = ReplayLocal(net, cfg, 0); gen = torch.Generator().manual_seed(0); buf = []
        for t in range(4):
            o = torch.randn(3, 5); a, p, v = L.act(o, gen)
            buf.append((o, a, torch.tensor([1., 0, 0]), torch.tensor([float(t == 3), 0, 0]), torch.log(p.gather(1, a[:, None]).squeeze(1))))
        w0 = [q.clone() for q in L.params]; L.learn(buf, L.forward(torch.randn(3, 5))[3], .5)
        assert all(not torch.equal(a, b) for a, b in zip(w0, L.params)), plastic
    # head-only, no clip, one pass: actor increment equals the exact policy gradient of the readout (locality is exact at the head)
    net2 = torch.nn.Module(); net2.body = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8), torch.nn.Tanh())
    net2.pi = torch.nn.Linear(8, 4); net2.v = torch.nn.Linear(8, 1); L = ReplayLocal(net2, dict(plastic="head", feedback="dfa", fb_scale=1.), 0)
    hs, z, p, v = L.forward(x); a = torch.tensor([0, 1, 2, 3, 0, 1]); A = torch.randn(6)
    W = net2.pi.weight.detach().clone().requires_grad_(True)
    (A * torch.log_softmax(hs[-1] @ W.T + net2.pi.bias.detach(), -1)[torch.arange(6), a]).mean().backward()
    d = L.dirs(hs, z, A[:, None] * (torch.nn.functional.one_hot(a, 4).float() - p), torch.zeros(6))
    assert torch.allclose(d[0], W.grad, atol=1e-6)
    print("check ok")


if __name__ == "__main__":
    c = sys.argv[1]; torch.set_num_threads(1)
    {"check": check, "prereg": prereg, "best": best, "analyse": analyse, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])),
     "test": lambda: test(int(sys.argv[2]), int(sys.argv[3])), "kpi": lambda: kpi(int(sys.argv[2]), int(sys.argv[3]))}[c]()
