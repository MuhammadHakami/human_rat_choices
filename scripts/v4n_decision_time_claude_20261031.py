"""Decision time and a learning control (camera-ready; author-approved 2026-10-31). Post hoc on saved weights + one control.
Motivation: Schmidbauer & Nieder (Science 2026): crow NCL activity ramps to a bound; choices fast at primate-level accuracy.
Zhou et al. (arXiv 2605.13315): permuted-decoding / no-culture ablations show learning depends on the feedback contingency.

S1 accumulate-to-bound (frozen source checkpoints, familiar goal, fresh rooms pairs 0-1 A, 30 seeds, priors M.PRIORS):
   the signed-IF network of v4n_kpi runs step by step; after step t the actor reads the time-averaged spikes
   (p_t = softmax(pi(acc/t))). A decision stops at the first t with max p_t >= theta (theta in THETAS) or at TMAX;
   the action is sampled from p_t (registered sampled evaluation). Recorded: success, mean steps to decision, hidden
   synaptic ops per decision, and agreement of the step-1 argmax and of the stop-time argmax with the dense network's.
   P1  mean steps to decision (averaged over THETAS), human_bc vs human_legal_uniform, paired over 30 seeds.
   P2  success human_bc - human_legal_uniform averaged over THETAS (paired); per-theta values descriptive.
S2 shuffled-reward control (TD-readout, registered rate, 30 seeds x 3 priors, test rooms as test2): at every step the
   16 parallel environments' rewards are permuted among environments, so reward statistics are kept but the
   action-reward contingency is broken. P3 final success (32,768 frames) registered TD-readout minus shuffled, paired.
   python scripts/v4n_decision_time_claude_20261031.py check | prereg | bound W N | shuffled W N | analyse
"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_kpi_claude_20261031 as K
import v4n_moved_goal_learning_claude_20261031 as M

OUT = M.ROOT / "data/audit/v4n_decision_time_20261031"; PROTOCOL = OUT / "PROTOCOL.json"
THETAS = (.5, .6, .7, .8, .9); TMAX = 32; PAIRS = (0, 1)
STAT = dict(steps=0., spikes=0., decisions=0, first_agree=0, stop_agree=0)   # module-level: survives the probe's deepcopy


class BoundedBody(torch.nn.Module):
    """Signed IF body that stops each row once its actor read-out crosses theta; returns acc/t at the stop step."""
    def __init__(self, body, pi, theta, tmax=TMAX):
        super().__init__(); self.L = [m for m in body if isinstance(m, torch.nn.Linear)]; self.dense = body; self.pi = pi; self.theta = theta; self.tmax = tmax

    @torch.no_grad()
    def forward(self, x):
        lead = x.shape[:-1]; x = x.reshape(-1, x.shape[-1]); I = self.L[0](x); V1 = torch.zeros_like(I); V2 = torch.zeros(len(x), self.L[1].out_features, device=x.device); acc = torch.zeros_like(V2)
        out = torch.zeros_like(V2); live = torch.ones(len(x), dtype=torch.bool, device=x.device); steps = torch.zeros(len(x), device=x.device)
        dense_arg = self.pi(self.dense(x)).argmax(-1); first = None
        for t in range(1, self.tmax + 1):
            V1 += I * live[:, None]; s1 = torch.sign(V1) * (V1.abs() >= 1) * live[:, None]; V1 -= s1
            V2 += self.L[1](s1) * live[:, None]; s2 = torch.sign(V2) * (V2.abs() >= 1) * live[:, None]; V2 -= s2
            acc += s2; STAT["spikes"] += float(s1.abs().sum() * self.L[1].out_features + s2.abs().sum() * 5)
            p = torch.softmax(self.pi(acc / t), -1)
            if first is None: first = p.argmax(-1)
            stop = live & ((p.max(-1).values >= self.theta) | (t == self.tmax))
            out[stop] = acc[stop] / t; steps[stop] = t; live &= ~stop
            if not live.any(): break
        STAT["steps"] += float(steps.sum()); STAT["decisions"] += len(x)
        STAT["first_agree"] += int((first == dense_arg).sum()); STAT["stop_agree"] += int((self.pi(out).argmax(-1) == dense_arg).sum())
        return out.reshape(*lead, -1)


def bound_one(prior, seed, theta, pair):
    m = M.build(prior, seed); dense = m.net.body; m.net.body = BoundedBody(dense, m.net.pi, theta)
    for k in STAT: STAT[k] = 0
    g = K.probe(m, {c: K.FAM for c in M.goal_map(M.FR.MAN)}, pair, "A", seed); m.net.body = dense; n = max(STAT["decisions"], 1)
    return dict(goals=g, steps=STAT["steps"] / n, synops_per_decision=STAT["spikes"] / n, first_step_agree=STAT["first_agree"] / n,
                stop_agree=STAT["stop_agree"] / n, decisions=STAT["decisions"])


def bound(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and PROTOCOL.exists()
    for p, s, th, q in [(p, s, th, q) for s in M.FR.S30 for p in M.PRIORS for th in THETAS for q in PAIRS][w::n]:
        f = OUT / "bound" / p / f"theta{th:g}" / f"seed{s}_pair{q}.json"
        if f.exists(): continue
        f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(dict(prior=p, seed=s, theta=th, pair=q, **bound_one(p, s, th, q))) + "\n")
    print(json.dumps(dict(stage="bound", worker=w, done=True)), flush=True)


def shuffled_env(env, seed):
    """Same environment; rewards permuted among the parallel environments at every step."""
    gen = np.random.default_rng([seed, 20261031]); step = env.step
    def _step(a):
        obs, r, done = step(a); return obs, (r[torch.as_tensor(gen.permutation(len(r)))] if torch.is_tensor(r) else r[gen.permutation(len(r))]), done
    env.step = _step; return env


def shuffled(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and PROTOCOL.exists()
    rate = json.loads((M.OUT / "PICK2.json").read_text())["rates"]["td_head"]; real = M.env_for
    for p, s in [(p, s) for s in M.FR.S30 for p in M.PRIORS][w::n]:
        first = [True]
        def once(*a, **k):                                                    # only the training env is shuffled; probes untouched
            e = real(*a, **k)
            if first[0]: first[0] = False; return shuffled_env(e, a[4])
            return e
        M.env_for = once
        try: M.run_one("td_head", p, s, rate, M.FR.MAN, OUT / "shuffled" / "td_head" / p / f"seed{s}.json")
        finally: M.env_for = real
    print(json.dumps(dict(stage="shuffled", worker=w, done=True)), flush=True)


def _paired(a, b):
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=float(d.mean()), ci=[float(d.mean() - h), float(d.mean() + h)], p=float(stats.ttest_1samp(d, 0).pvalue), n=len(d))


def analyse():
    def per_seed(p, key, th=None):
        ths = THETAS if th is None else (th,)
        return [np.mean([json.loads((OUT / "bound" / p / f"theta{t:g}" / f"seed{s}_pair{q}.json").read_text())[key] for t in ths for q in PAIRS]) for s in M.FR.S30]
    H, L = "human_bc", "human_legal_uniform"; res = dict(P1=_paired(per_seed(H, "steps"), per_seed(L, "steps")),
        P2=_paired(np.array(per_seed(H, "goals")) * 100 / 30, np.array(per_seed(L, "goals")) * 100 / 30), per_theta={})
    for th in THETAS:
        res["per_theta"][f"{th:g}"] = {p: {k: float(np.mean(per_seed(p, k, th))) for k in ("goals", "steps", "synops_per_decision", "first_step_agree", "stop_agree")} for p in M.PRIORS}
        res["per_theta"][f"{th:g}"]["success_H_minus_L_pp"] = _paired(np.array(per_seed(H, "goals", th)) * 100 / 30, np.array(per_seed(L, "goals", th)) * 100 / 30)
    rate = json.loads((M.OUT / "PICK2.json").read_text())["rates"]["td_head"]; fin = str(M.BUDGET)
    res["P3"] = {}
    for p in M.PRIORS:
        reg = [json.loads((M.OUT / "test2" / "td_head" / f"rate{rate:g}" / p / f"seed{s}.json").read_text())["curve"][fin] * 100 / 30 for s in M.FR.S30]
        sh = [json.loads((OUT / "shuffled" / "td_head" / p / f"seed{s}.json").read_text())["curve"][fin] * 100 / 30 for s in M.FR.S30]
        res["P3"][p] = dict(registered_minus_shuffled_pp=_paired(reg, sh), registered=float(np.mean(reg)), shuffled=float(np.mean(sh)))
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res, indent=1))


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), outcomes_seen=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), design=__doc__, thetas=THETAS, tmax=TMAX, pairs=PAIRS,
        predictions=dict(P1="human_bc reaches the bound in fewer steps than human_legal_uniform (its choices are less ambiguous)",
                         P2="human_bc success exceeds human_legal_uniform under bounded decisions", P3="registered TD-readout beats shuffled reward"),
        family="P1-P3 Holm", reading="each prediction reported as supported or not; per-theta results descriptive"), indent=1) + "\n")
    print("registered")


def check():
    lin = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 4), torch.nn.Tanh()); pi = torch.nn.Linear(4, 4)
    with torch.no_grad():
        lin[0].weight.copy_(torch.eye(4, 3) * .5); lin[0].bias.zero_(); lin[2].weight.copy_(torch.eye(4)); lin[2].bias.zero_()
        pi.weight.copy_(torch.eye(4) * 20); pi.bias.zero_()
    x = torch.tensor([[1.8, 0., 0.], [.05, 0., 0.]])                             # strong vs weak evidence
    for k in STAT: STAT[k] = 0
    BoundedBody(lin, pi, .9)(x); steps_two = STAT["steps"]
    for k in STAT: STAT[k] = 0
    BoundedBody(lin, pi, .9)(x[:1]); assert STAT["steps"] <= 2, STAT                # strong evidence decides at once
    assert steps_two - STAT["steps"] >= 10, (steps_two, STAT)                        # weak evidence takes much longer
    for k in STAT: STAT[k] = 0
    BoundedBody(lin, pi, 1.01)(x); assert STAT["steps"] == 2 * TMAX                  # unreachable bound -> TMAX
    r = torch.arange(16.); e = type("E", (), {"step": lambda self, a: (None, r.clone(), None)})(); se = shuffled_env(e, 1)
    _, rs, _ = se.step(None); assert sorted(rs.tolist()) == r.tolist() and not torch.equal(rs, r)
    print("check ok", steps_two, STAT)


if __name__ == "__main__":
    c = sys.argv[1]; torch.set_num_threads(1)
    {"check": check, "prereg": prereg, "analyse": analyse, "bound": lambda: bound(int(sys.argv[2]), int(sys.argv[3])),
     "shuffled": lambda: shuffled(int(sys.argv[2]), int(sys.argv[3]))}[c]()
