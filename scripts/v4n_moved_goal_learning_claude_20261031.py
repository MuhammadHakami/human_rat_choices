"""Moved-goal LEARNING (camera-ready; author 2026-10-31: PPO plus biologically plausible local learners, GPU 1).
Each frozen seeds30 source checkpoint (prior: human choices / human legal labels / scratch) learns ONE moved goal in
ONE room online, then is probed frozen (30 trials, sampled actions) on that room and goal.
Learners (same initial weights, same interactions):
  ppo      registered E-family PPO (wacv_continual_geometry_development manifest 'ppo'), unchanged
  td_head  local TD(lambda) actor-critic on the readout (three-factor: TD error x eligibility trace), body frozen
           [Sutton & Barto 13.6; Fremaux, Sprekeler & Gerstner 2013]
  td_fa    the same three-factor rule in every layer, hidden credit by FIXED random feedback (no weight transport)
           [direct feedback alignment, Nokland 2016; reward-modulated three-factor form as in e-prop's random-feedback variant]
Calibration (dev): exposed rooms (cfg 100-115 of the continual study), dev seeds 1000-1002, pick the actor rate per
local learner by mean final success over priors; test: fresh confirmation rooms, all 30 seeds, rates frozen.
  prereg | calib W N | pick | test W N | analyse
"""
import gzip, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_camera_ready_claude_20261031 as CR

FR, F, worlds, ROOT = CR.FR, CR.F, CR.worlds, CR.ROOT
OUT = ROOT / "data/audit/v4n_moved_goal_learning_20261031"; PROTOCOL = OUT / "PROTOCOL.json"
PRIORS = ("human_bc", "human_legal_uniform", "cold_ppo"); LEARNERS = ("ppo", "td_head", "td_fa")
BUDGET, N_ENV, CURVE = 32768, 16, (0, 4096, 8192, 16384, 32768)
GAMMA, LAM, ETA_V_RATIO = .95, .9, 1.                     # gamma as the registered PPO; critic rate = actor rate
RATES = (1e-4, 3e-4, 1e-3, 3e-3)        # v2 (amendment 3): Adam step sizes
V = "2"; ENT = .05                       # v2: outputs in calib2/test2, PICK2.json; entropy bonus as the registered PPO
DEV_SEEDS, TARGET = (1000, 1001, 1002), (6, 6)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def room_of(seed): p = seed % 8; return p, "A"                       # one room per seed, fixed before outcomes


def goal_map(rooms_manifest, target=TARGET):
    m = json.loads(Path(rooms_manifest).read_text()); starts = [tuple(s) for s in m["starts"]]
    return {r["configuration"]: CR.moved_goal(np.asarray(r["walkable"], bool), target, starts) for r in m["rows"]}


def env_for(manifest, gm, pair, n, seed, tier, phase_frames, schedule=None):
    _, _, source = worlds.fresh.load(); cache = worlds.RoomCache(source, json.loads(Path(manifest).read_text())); cache.goals.update(gm)
    phys = worlds.ContinuousRooms(cache, n=n, seed=82000 + seed, phase_frames=phase_frames, pair=pair, schedule=schedule)
    return worlds.VisualRooms(phys, source, tier, noise_seed=[81013, seed, int(tier == "noisy")])


class LocalTD:
    """Online TD(lambda) actor-critic with per-synapse eligibility traces; every update is (TD error) x (local trace).
    td_head: readout only. td_fa: hidden layers get the actor's output error through fixed random matrices B."""
    def __init__(self, net, rate, fa, seed):
        self.net, self.rate, self.fa = net, rate, fa; dev = net.pi.weight.device
        self.L = [m for m in net.body if isinstance(m, torch.nn.Linear)]
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.B = [torch.randn(l.out_features, net.pi.out_features, generator=g).to(dev) / np.sqrt(net.pi.out_features) for l in self.L]
        self.params = [net.pi.weight, net.pi.bias, net.v.weight, net.v.bias] + ([p for l in self.L for p in (l.weight, l.bias)] if fa else [])
        self.e = None; self.t = 0
        self.m = [torch.zeros_like(q) for q in self.params]; self.v2 = [torch.zeros_like(q) for q in self.params]   # per-synapse Adam state (as e-prop, Bellec 2020)

    @torch.no_grad()
    def act(self, obs, gen):
        x = torch.as_tensor(obs, device=self.net.pi.weight.device); hs = [x]
        for l in self.L: hs.append(torch.tanh(l(hs[-1])))
        p = torch.softmax(self.net.pi(hs[-1]), -1); v = self.net.v(hs[-1]).squeeze(-1)
        a = torch.multinomial(p.cpu(), 1, generator=gen).squeeze(-1)
        return a, p, v, hs

    @torch.no_grad()
    def grads(self, a, p, hs):
        """Per-env local 'presynaptic x postsynaptic' terms (n, *param.shape), the eligibility increments.
        Order: pi.w, pi.b, v.w, v.b, then (td_fa) L0.w, L0.b, L1.w, L1.b."""
        g = torch.nn.functional.one_hot(a.to(p.device), p.shape[-1]).float() - p; h = hs[-1]; n = len(h)
        out = [g[:, :, None] * h[:, None, :], g, h[:, None, :].clone(), torch.ones(n, 1, device=h.device)]
        if self.fa:
            for i in range(len(self.L)):                       # direct random feedback of the output error x local derivative
                d = (g @ self.B[i].T) * (1 - hs[i + 1] ** 2); out += [d[:, :, None] * hs[i][:, None, :], d]
        return out

    @torch.no_grad()
    def step(self, a, p, v, hs, r, done, v_next):
        """Three-factor direction per synapse: TD error x eligibility trace (+ the policy's own entropy term), applied with
        per-synapse Adam scaling (local state only). Params order: pi.w, pi.b, v.w, v.b, [L0.w, L0.b, L1.w, L1.b]."""
        inc = self.grads(a, p, hs); dev = v.device
        keep = torch.as_tensor(1. - done.astype(np.float32), device=dev)
        bc = lambda t, x: x.view(-1, *[1] * (t.dim() - 1))
        if self.e is None: self.e, self.keep_prev = [torch.zeros_like(t) for t in inc], torch.ones_like(keep)
        self.e = [GAMMA * LAM * e * bc(e, self.keep_prev) + t for e, t in zip(self.e, inc)]   # traces restart after an episode ends
        self.keep_prev = keep
        delta = torch.as_tensor(r, device=dev) + GAMMA * v_next * keep - v                     # third factor (TD error)
        logp = torch.log(p.clamp_min(1e-8)); H = -(p * logp).sum(-1, keepdim=True); gH = -p * (logp + H)   # dH/dlogits, local to each output unit
        ent = [gH[:, :, None] * hs[-1][:, None, :], gH]
        if self.fa:
            for i in range(len(self.L)):
                d = (gH @ self.B[i].T) * (1 - hs[i + 1] ** 2); ent += [d[:, :, None] * hs[i][:, None, :], d]
        d_actor = [(bc(e, delta) * e).mean(0) + ENT * h.mean(0) for e, h in zip(self.e[:2] + self.e[4:], ent[:2] + ent[2:])]
        d_critic = [(bc(e, delta) * e).mean(0) for e in self.e[2:4]]
        dirs = d_actor[:2] + d_critic + d_actor[2:]
        self.t += 1; b1, b2 = .9, .999
        for q, g, m, v2 in zip(self.params, dirs, self.m, self.v2):
            g = g.view_as(q); m.mul_(b1).add_((1 - b1) * g); v2.mul_(b2).add_((1 - b2) * g * g)
            q += self.rate * (m / (1 - b1 ** self.t)) / ((v2 / (1 - b2 ** self.t)).sqrt() + 1e-5)   # ascent


def build(prior, seed):
    method = next(x for x in json.loads(F.study.MANIFEST.read_text())["methods"] if x["id"] == "ppo")
    F.SEEDS30 = ROOT / "data/wacv/seeds30/cohort"; F.study.source_model = lambda s: F.source_checkpoint(prior, s)
    return F.study.build(method, seed)


def probe_now(m, manifest, gm, pair, stage, seed):
    F.environment = lambda pr, n, s, tier, pf, sch=None: env_for(manifest, gm, pr, n, s, tier, pf, sch)
    r = F.probe(m, pair, stage, seed, "clean"); cfg = 100 + 2 * pair + int(stage == "B")
    return dict(goals=r["goals"], episodes=[CR.episode_metrics(e, gm[cfg]) for e in r["episodes"] if "spawn" in e],
                paths=[dict(spawn=e["spawn"], rotation_index=e["rotation_index"], path=e["path"], actions=e["actions"]) for e in r["episodes"] if "spawn" in e])


def run_one(learner, prior, seed, rate, manifest, out):
    if out.exists(): return json.loads(out.read_text())
    pair, stage = room_of(seed); gm = goal_map(manifest); m = build(prior, seed); t0 = time.perf_counter(); diverged_at = None
    env = env_for(manifest, gm, pair, N_ENV, seed, "clean", BUDGET); obs = env.reset(); curve = {0: probe_now(m, manifest, gm, pair, stage, seed)}
    if learner == "ppo":
        wrapped = F.LoggingRooms(env, obs); state = (torch.zeros(1, N_ENV, 0), torch.zeros(1, N_ENV, 0)); dprev = np.zeros(N_ENV, np.float32)
        while m.count["frames"] < BUDGET:
            B, s0, obs, state, dprev = m.rollout(wrapped, obs, state, dprev); m.update(B, s0)
            if m.count["frames"] in CURVE: curve[m.count["frames"]] = probe_now(m, manifest, gm, pair, stage, seed)
    else:
        L = LocalTD(m.net, rate, learner == "td_fa", 4242 + seed); gen = torch.Generator().manual_seed(77 + seed); frames = 0
        a, p, v, hs = L.act(obs, gen)
        while frames < BUDGET:
            obs, r, done = env.step(a.numpy()); frames += N_ENV
            if not all(torch.isfinite(q).all() for q in m.net.parameters()):          # amendment 1: divergence = failure
                diverged_at = frames; break
            a2, p2, v2, hs2 = L.act(obs, gen); L.step(a, p, v, hs, r, done, v2); a, p, v, hs = a2, p2, v2, hs2
            if frames in CURVE: curve[frames] = probe_now(m, manifest, gm, pair, stage, seed)
        if diverged_at is not None:
            for x in CURVE:
                if x not in curve: curve[x] = dict(goals=0, episodes=[])
    assert set(curve) == set(CURVE), sorted(curve)
    rec = dict(learner=learner, prior=prior, seed=seed, rate=rate, pair=pair, room=stage, goal=list(gm[100 + 2 * pair]), budget=BUDGET,
               curve={k: v["goals"] for k, v in curve.items()}, diverged_at=diverged_at, final_episodes=curve[BUDGET]["episodes"], wall_s=time.perf_counter() - t0,
               protocol_sha256=sha(PROTOCOL), gpu_name=torch.cuda.get_device_name(0))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in m.net.state_dict().items()}, out.with_suffix(".final.pt"))                 # amendment 2: for visual rollouts
    (out.with_suffix(".paths.json.gz")).write_bytes(gzip.compress(json.dumps({k: v.get("paths", []) for k, v in curve.items()}).encode()))
    out.write_text(json.dumps(rec) + "\n"); return rec


def jobs(phase):
    if phase == "calib" + V:
        return [(l, p, s, r) for s in DEV_SEEDS for p in PRIORS for l in ("td_head", "td_fa") for r in RATES] + [("ppo", p, s, None) for s in DEV_SEEDS for p in PRIORS]
    pick = json.loads((OUT / f"PICK{V}.json").read_text())["rates"]
    return [(l, p, s, pick.get(l)) for s in FR.S30 for p in PRIORS for l in LEARNERS]


def worker(phase, w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    manifest = FR.EXPOSED if phase.startswith("calib") else FR.MAN
    for l, p, s, r in jobs(phase)[w::n]:
        run_one(l, p, s, r, manifest, OUT / phase / l / (f"rate{r:g}" if r else "registered") / p / f"seed{s}.json")
    print(json.dumps(dict(phase=phase, worker=w, done=True)), flush=True)


def pick():
    assert not (OUT / f"PICK{V}.json").exists()
    score = {}
    for l in ("td_head", "td_fa"):
        for r in RATES:
            v = [json.loads((OUT / f"calib{V}" / l / f"rate{r:g}" / p / f"seed{s}.json").read_text())["curve"][str(BUDGET)] for p in PRIORS for s in DEV_SEEDS]
            score[f"{l}/{r:g}"] = float(np.mean(v))
    rates = {l: max(RATES, key=lambda r: (score[f"{l}/{r:g}"], -r)) for l in ("td_head", "td_fa")}
    (OUT / f"PICK{V}.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rule="max mean final dev success; ties -> smaller rate",
                                                   dev_scores=score, rates=rates), indent=1) + "\n"); print(rates, score)


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(
        written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), outcomes_seen=False, script_sha256=sha(__file__),
        question="Does a human-choice prior speed learning of a NEW goal, and does that depend on the learning rule (PPO vs local three-factor rules)?",
        priors=PRIORS, learners=LEARNERS, budget=BUDGET, n_envs=N_ENV, curve=CURVE, gamma=GAMMA, trace_lambda=LAM,
        rooms=dict(dev="exposed continual rooms (cfg 100-115), seeds 1000-1002", test="16 fresh confirmation rooms, seeds 1000-1029; room = pair seed%8, A"),
        moved_goal=f"target {TARGET}, camera-ready nearest-walkable rule", calibration=dict(rates=RATES, rule="max mean final dev success over priors; PPO untuned (registered)"),
        primary=dict(outcome="normalised learning-curve AUC (trapezoid over CURVE of successes/30, divided by 30 x budget) and final success",
                     contrasts="per learner: human_bc - human_legal_uniform, human_bc - cold_ppo (paired over 30 seeds); Holm over the 6 AUC contrasts"),
        secondary="final success; learner contrasts within prior; trapped/revisit at the final probe; dev seeds 1000-1002 also appear in test (different rooms)",
        caveat="local learners are biologically plausible credit-assignment rules on a backprop-trained (BC+PPO) starting network; not spiking, not a full bio-plausible agent"), indent=1) + "\n")
    print(PROTOCOL.read_text()[:600])


def analyse():
    from prior_seed_power_claude import contrast
    from analyse_seeds30_claude import holm
    xs = np.array(CURVE, float); out = dict(pick=json.loads((OUT / f"PICK{V}.json").read_text()), learners={})
    auc = lambda c: float(np.trapezoid([c[str(int(x))] for x in xs], xs) / (30 * BUDGET))
    fam = {}
    for l in LEARNERS:
        S = {"auc": {}, "final": {}, "trapped": {}}
        for p in PRIORS:
            for k in S: S[k][p] = {}
            for f in sorted((OUT / f"test{V}" / l).glob(f"*/{p}/seed*.json")):
                r = json.loads(f.read_text()); s = r["seed"]
                S["auc"][p][s] = auc(r["curve"]) * 100; S["final"][p][s] = r["curve"][str(BUDGET)] * 100 / 30
                S["trapped"][p][s] = 100 * float(np.mean([e["trapped"] for e in r["final_episodes"]]))
        res = dict(means={k: {p: float(np.mean(list(v[p].values()))) if v[p] else None for p in PRIORS} for k, v in S.items()},
                   n={p: len(S["auc"][p]) for p in PRIORS})
        for k in ("auc", "final"):
            for b in ("human_legal_uniform", "cold_ppo"):
                if len(S[k]["human_bc"]) > 2 and len(S[k][b]) > 2: res[f"{k}/human_bc-{b}"] = contrast(S[k], "human_bc", b, 10.)
        out["learners"][l] = res
        for b in ("human_legal_uniform", "cold_ppo"):
            if f"auc/human_bc-{b}" in res: fam[f"{l}/{b}"] = res[f"auc/human_bc-{b}"]
    for k, h in zip(fam, holm([v["t_p"] for v in fam.values()])): fam[k]["holm_t_p"] = h
    out["primary_auc"] = fam; (OUT / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    for l, r in out["learners"].items(): print(l, r["n"], json.dumps(r["means"]))
    print(json.dumps({k: [round(v["mean"], 1), [round(x, 1) for x in v["ci95"]], round(v["holm_t_p"], 4)] for k, v in fam.items()}))


def check():
    torch.manual_seed(0); net = torch.nn.Module(); net.body = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Tanh(), torch.nn.Linear(3, 3), torch.nn.Tanh())
    net.pi = torch.nn.Linear(3, 4); net.v = torch.nn.Linear(3, 1)
    L = LocalTD(net, .1, False, 0); x = torch.randn(2, 5); hs = [x]
    for l in L.L: hs.append(torch.tanh(l(hs[-1])))
    p = torch.softmax(net.pi(hs[-1]), -1); a = torch.tensor([1, 2])
    inc = L.grads(a, p, hs)
    xr = x.clone(); w = net.pi.weight.detach().clone().requires_grad_(True)       # readout increment == grad of log pi (exact for the head)
    lp = torch.log_softmax(hs[-1].detach() @ w.T + net.pi.bias.detach(), -1)[torch.arange(2), a].sum(); lp.backward()
    assert torch.allclose(inc[0].sum(0), w.grad, atol=1e-6)
    Lf = LocalTD(net, .1, True, 0); incf = Lf.grads(a, p, hs); assert len(incf) == 8 and incf[4].shape == (2, 3, 5) and incf[6].shape == (2, 3, 3)
    w0 = net.pi.weight.clone(); Lf.step(a, p, net.v(hs[-1]).squeeze(-1), hs, np.array([1., 0.]), np.array([True, False]), torch.zeros(2)); assert not torch.equal(w0, net.pi.weight)
    print("check ok")


if __name__ == "__main__":
    torch.set_num_threads(1); c = sys.argv[1]
    {"check": check, "prereg": prereg, "pick": pick, "analyse": analyse}.get(c, lambda: worker(c, int(sys.argv[2]), int(sys.argv[3])))()
