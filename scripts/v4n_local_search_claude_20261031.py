"""Plasticity-architecture search (NAS) + Optuna TPE for biologically plausible local learners on the moved-goal task
(camera-ready, author 2026-10-31). DEV ONLY: exposed continual rooms, seeds 1000-1002, 3 priors (9 runs per trial); the
selected configurations are then tested once on the 30-seed fresh-room protocol of v4n_moved_goal_learning (test3).
Search space
  NAS:    plastic layers {head, last, all}; hidden-credit route {dfa: fixed random feedback (Nokland 2016),
          ed: modulo error routing of Error Diffusion (Yamada et al. 2026, 'Diffusing Blame', Sakana AI), unit j <- action j mod 4}
  Optuna: Adam rate [1e-5, 1e-2] log; trace lambda [.5, .99]; entropy [0, .1]; critic/actor rate ratio [.3, 3] log; feedback scale [.1, 3] log
Every update stays local: TD error (third factor) x per-synapse eligibility trace, per-synapse Adam scaling.
The ED variant borrows ED's credit ROUTING only; the pretrained network is not ED's dual-stream Dale architecture.
  prereg | search W N TRIALS | best | test W N | analyse
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_moved_goal_learning_claude_20261031 as M

OUT = M.ROOT / "data/audit/v4n_local_search_20261031"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"
FAMILIES = ("dfa", "ed")


class SearchTD:
    """Configurable three-factor learner (same math as M.LocalTD v2, plus plastic-layer and feedback choices)."""
    def __init__(self, net, cfg, seed):
        self.net, self.c = net, cfg; dev = net.pi.weight.device
        self.L = [m for m in net.body if isinstance(m, torch.nn.Linear)]
        self.hid = {"head": [], "last": [len(self.L) - 1], "all": list(range(len(self.L)))}[cfg["plastic"]]
        g = torch.Generator().manual_seed(seed); A = net.pi.out_features
        if cfg["feedback"] == "ed":
            self.B = [(torch.arange(l.out_features)[:, None] % A == torch.arange(A)[None]).float().to(dev) * cfg["fb_scale"] for l in self.L]
        else:
            self.B = [torch.randn(l.out_features, A, generator=g).to(dev) / np.sqrt(A) * cfg["fb_scale"] for l in self.L]
        self.params = [net.pi.weight, net.pi.bias, net.v.weight, net.v.bias] + [q for i in self.hid for q in (self.L[i].weight, self.L[i].bias)]
        self.m = [torch.zeros_like(q) for q in self.params]; self.s2 = [torch.zeros_like(q) for q in self.params]; self.t = 0; self.e = None

    act = M.LocalTD.act

    @torch.no_grad()
    def incs(self, g, hs):
        out = [g[:, :, None] * hs[-1][:, None, :], g]
        for i in self.hid:
            d = (g @ self.B[i].T) * (1 - hs[i + 1] ** 2); out += [d[:, :, None] * hs[i][:, None, :], d]
        return out

    @torch.no_grad()
    def step(self, a, p, v, hs, r, done, v_next):
        c = self.c; dev = v.device; bc = lambda t, x: x.view(-1, *[1] * (t.dim() - 1))
        g = torch.nn.functional.one_hot(a.to(dev), p.shape[-1]).float() - p
        inc = self.incs(g, hs); inc = inc[:2] + [hs[-1][:, None, :].clone(), torch.ones(len(v), 1, device=dev)] + inc[2:]
        keep = torch.as_tensor(1. - done.astype(np.float32), device=dev)
        if self.e is None: self.e, self.kp = [torch.zeros_like(t) for t in inc], torch.ones_like(keep)
        self.e = [M.GAMMA * c["lam"] * e * bc(e, self.kp) + t for e, t in zip(self.e, inc)]; self.kp = keep
        delta = torch.as_tensor(r, device=dev) + M.GAMMA * v_next * keep - v
        logp = torch.log(p.clamp_min(1e-8)); H = -(p * logp).sum(-1, keepdim=True); ent = self.incs(-p * (logp + H), hs)
        ent = ent[:2] + [None, None] + ent[2:]
        dirs = []
        for k, e in enumerate(self.e):
            d = (bc(e, delta) * e).mean(0) * (c["critic_ratio"] if k in (2, 3) else 1.)
            if ent[k] is not None: d = d + c["ent"] * ent[k].mean(0)
            dirs.append(d)
        self.t += 1
        for q, d, m, s2 in zip(self.params, dirs, self.m, self.s2):
            d = d.view_as(q); m.mul_(.9).add_(.1 * d); s2.mul_(.999).add_(.001 * d * d)
            q += c["rate"] * (m / (1 - .9 ** self.t)) / ((s2 / (1 - .999 ** self.t)).sqrt() + 1e-5)


def run_cfg(args):
    """One learning run with a SearchTD config (reuses M.run_one's environment/probe code by swapping the learner)."""
    cfg, prior, seed, manifest, out = args
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1"); torch.set_num_threads(1)
    M.LocalTD = lambda net, rate, fa, s, cfg=cfg: SearchTD(net, cfg, s)        # ponytail: process-local swap, learner name 'td_head'
    return M.run_one("td_head", prior, seed, cfg["rate"], Path(manifest), Path(out))


def objective(trial):
    cfg = dict(plastic=trial.suggest_categorical("plastic", ["head", "last", "all"]), feedback=trial.suggest_categorical("feedback", list(FAMILIES)),
               rate=trial.suggest_float("rate", 1e-5, 1e-2, log=True), lam=trial.suggest_float("lam", .5, .99),
               ent=trial.suggest_float("ent", 0., .1), critic_ratio=trial.suggest_float("critic_ratio", .3, 3., log=True),
               fb_scale=trial.suggest_float("fb_scale", .1, 3., log=True))
    jobs = [(cfg, p, s, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}" / p / f"seed{s}.json")) for p in M.PRIORS for s in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = [dict(r, curve={str(k): v for k, v in r["curve"].items()}) for r in ex.map(run_cfg, jobs)]
    xs = np.array(M.CURVE, float); auc = np.mean([np.trapezoid([r["curve"][str(int(x))] for x in xs], xs) / (30 * M.BUDGET) for r in recs])
    trial.set_user_attr("final", float(np.mean([r["curve"][str(M.BUDGET)] for r in recs]))); trial.set_user_attr("auc", float(auc))
    return float(np.mean([r["curve"][str(M.BUDGET)] for r in recs])) + float(auc)      # final success /30 + AUC tie-break


def search(w, n, trials):
    import optuna
    st = optuna.create_study(study_name="v4n_local", storage=DB, direction="maximize", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=20261031 + w, n_startup_trials=8))
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_local", storage=DB); done = [t for t in st.trials if t.value is not None]
    pick = {}
    for fam in FAMILIES:
        tt = [t for t in done if t.params["feedback"] == fam and t.params["plastic"] != "head"]
        if tt: pick[f"{fam}"] = max(tt, key=lambda t: t.value).params
    hh = [t for t in done if t.params["plastic"] == "head"]
    if hh: pick["head"] = max(hh, key=lambda t: t.value).params
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done), configs=pick,
        scores={k: max(t.value for t in done if t.params == v) for k, v in pick.items()}), indent=1) + "\n"); print(json.dumps(pick, indent=1))


def test(w, n):
    pick = json.loads((OUT / "BEST.json").read_text())["configs"]
    jobs = [(name, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for name, cfg in sorted(pick.items())][w::n]
    for name, cfg, p, s in jobs:
        run_cfg((cfg, p, s, str(M.FR.MAN), str(OUT / "test3" / name / p / f"seed{s}.json")))
    print(json.dumps(dict(worker=w, done=len(jobs))), flush=True)


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), outcomes_seen="no test outcome; v1 dev calibration seen",
        search_space=__doc__.split("Search space")[1].split("Every update")[0], objective="mean final dev success (/30) over 3 priors x 3 dev seeds + AUC tie-break",
        selection="best trial per hidden-credit family (dfa, ed; plastic != head) and best head-only trial", test="30 seeds x 3 priors, fresh rooms, as v4n_moved_goal_learning PROTOCOL",
        primary="per selected learner: human_bc - human_legal_uniform and human_bc - cold_ppo learning-curve AUC; Holm over all selected learners",
        caveat="search uses the dev rooms only; learners start from backprop-trained priors; ED = routing only, not dual-stream Dale"), indent=1) + "\n")
    print(PROTOCOL.read_text()[:400])


def check():
    net = torch.nn.Module(); net.body = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8), torch.nn.Tanh())
    net.pi = torch.nn.Linear(8, 4); net.v = torch.nn.Linear(8, 1)
    for plastic in ("head", "last", "all"):
        for fb in FAMILIES:
            L = SearchTD(net, dict(plastic=plastic, feedback=fb, rate=1e-3, lam=.9, ent=.05, critic_ratio=1., fb_scale=1.), 0)
            a, p, v, hs = L.act(np.random.rand(2, 5).astype(np.float32), torch.Generator().manual_seed(0))
            w0 = [q.clone() for q in L.params]; L.step(a, p, v, hs, np.array([1., 0.]), np.array([True, False]), torch.zeros(2))
            assert all(not torch.equal(x, y) for x, y in zip(w0, L.params)), (plastic, fb)
    E = SearchTD(net, dict(plastic="all", feedback="ed", rate=1e-3, lam=.9, ent=0., critic_ratio=1., fb_scale=1.), 0)
    assert torch.equal(E.B[0][5], torch.tensor([0., 1., 0., 0.]))        # unit 5 <- action 5 mod 4 = 1
    print("check ok")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "check": check()
    elif c == "prereg": prereg()
    elif c == "search": search(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif c == "best": best()
    elif c == "test": test(int(sys.argv[2]), int(sys.argv[3]))
