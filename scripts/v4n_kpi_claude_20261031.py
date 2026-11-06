"""Four claims, measured on the accepted task (camera-ready; author 2026-10-31): continual learning, forgetting, OOD
generalisation of new learning, and spiking efficiency. Post hoc from saved weights; no learning here.
For every moved-goal learning run (v4n_moved_goal_learning test2: ppo / td_head / td_fa, and v4n_local_search test3)
with prior P, seed s, trained room R_A = (pair s%8, A) and moved goal g:
  continual   : success on g in R_A after learning (already in the learning record)
  forgetting  : familiar goal (3,3) in R_A, before (source checkpoint) vs after learning
  OOD         : g in the held-out paired room R_B = (pair s%8, B), never trained, before vs after
  OOD-familiar: (3,3) in R_B after learning (does the old skill survive elsewhere?)
Spiking: each source checkpoint converted to a signed integrate-and-fire network (rate code, T steps, threshold 1,
reset by subtraction; analog input current; non-spiking readout = mean of per-step logits), probed frozen with the
familiar goal in fresh rooms (pairs 0-3, A). Reported: success and synaptic operations per decision (hidden spikes x
fan-out) versus the dense network's multiply-accumulates. Operation counts, not energy.
  python scripts/v4n_kpi_claude_20261031.py kpi W N | snn W N | check
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_moved_goal_learning_claude_20261031 as M

OUT = M.ROOT / "data/audit/v4n_kpi_20261031"; ML = M.OUT; LS = M.ROOT / "data/audit/v4n_local_search_20261031"
FAM = (3, 3); TS = tuple(int(x) for x in os.environ.get("SNN_TS", "8,16,32").split(",")); SNN_PAIRS = (0, 1)
COUNT = dict(spikes=0., decisions=0)                 # module-level: survives the probe's deepcopy of the agent


def runs():
    out = []
    for base, sub in ((ML, "test2"), (LS, "test3")):
        for f in sorted((base / sub).glob("*/*/*/seed*.json")) + sorted((base / sub).glob("*/*/seed*.json")):
            if f.name.endswith(".paths.json.gz") or not f.with_suffix(".final.pt").exists(): continue
            r = json.loads(f.read_text()); name = f.parts[-3] if sub == "test3" else r["learner"]
            out.append((name, r["prior"], r["seed"], f))
    return sorted(set(out), key=lambda x: (x[0], x[1], x[2]))


def probe(m, gm, pair, stage, seed):
    r = M.probe_now(m, M.FR.MAN, gm, pair, stage, seed); return r["goals"]


def kpi_one(prior, seed, final=None):
    m = M.build(prior, seed)
    if final is not None: m.net.load_state_dict({k: v.to(next(m.net.parameters()).device) for k, v in torch.load(final).items()})
    pair = seed % 8; mg = M.goal_map(M.FR.MAN); fam = {c: FAM for c in mg}
    return dict(familiar_trained_room=probe(m, fam, pair, "A", seed), moved_heldout_room=probe(m, mg, pair, "B", seed),
                familiar_heldout_room=probe(m, fam, pair, "B", seed))


def kpi(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    R = runs(); base = sorted({(p, s) for _, p, s, _ in R})
    jobs = [("before", p, s, None) for p, s in base] + [(l, p, s, f) for l, p, s, f in R]
    for l, p, s, f in jobs[w::n]:
        out = OUT / "kpi" / l / p / f"seed{s}.json"
        if out.exists(): continue
        rec = dict(learner=l, prior=p, seed=s, source=str(f) if f else "source checkpoint", **kpi_one(p, s, f.with_suffix(".final.pt") if f else None))
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec) + "\n")
    print(json.dumps(dict(worker=w, done=len(jobs[w::n]))), flush=True)


class SpikingBody(torch.nn.Module):
    """Signed IF conversion of the frozen tanh body; returns the time-averaged spikes of the last hidden layer."""
    def __init__(self, body, T):
        super().__init__(); self.L = [m for m in body if isinstance(m, torch.nn.Linear)]; self.T = T; self.spikes = 0; self.decisions = 0

    @torch.no_grad()
    def forward(self, x):
        I = self.L[0](x); V1 = torch.zeros_like(I); V2 = None; acc = 0.
        for _ in range(self.T):
            V1 += I; s1 = torch.sign(V1) * (V1.abs() >= 1); V1 -= s1
            I2 = self.L[1](s1)
            if V2 is None: V2 = torch.zeros_like(I2)
            V2 += I2; s2 = torch.sign(V2) * (V2.abs() >= 1); V2 -= s2
            acc = acc + s2; COUNT["spikes"] += float(s1.abs().sum() * self.L[1].out_features + s2.abs().sum() * 5)   # fan-out of each spike
        COUNT["decisions"] += len(x); return acc / self.T


def snn_one(prior, seed, T, pair):
    m = M.build(prior, seed); body = SpikingBody(m.net.body, T); dense = m.net.body; COUNT.update(spikes=0., decisions=0)
    m.net.body = body                                                           # unroll() calls body(x); heads() reads its output
    g = probe(m, {c: FAM for c in M.goal_map(M.FR.MAN)}, pair, "A", seed)
    m.net.body = dense; return g, COUNT["spikes"] / max(COUNT["decisions"], 1)


def snn(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    jobs = [(p, s, T, q) for s in M.FR.S30 for p in M.PRIORS for T in TS for q in SNN_PAIRS][w::n]
    for p, s, T, q in jobs:
        out = OUT / "snn" / p / f"T{T}" / f"seed{s}_pair{q}.json"
        if out.exists(): continue
        g, ops = snn_one(p, s, T, q); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(prior=p, seed=s, T=T, pair=q, goals=g, synops_per_decision=ops, dense_macs_hidden=64 * 64 + 64 * 5,
                                       dense_macs_total=252 * 64 + 64 * 64 + 64 * 5)) + "\n")
    print(json.dumps(dict(worker=w, done=len(jobs))), flush=True)


def check():
    lin = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 4), torch.nn.Tanh())
    with torch.no_grad():
        lin[0].weight.copy_(torch.eye(4, 3) * .5); lin[0].bias.zero_(); lin[2].weight.copy_(torch.eye(4)); lin[2].bias.zero_()
    b = SpikingBody(lin, 64); y = b(torch.tensor([[.8, -.4, 0.]]))                 # rates ~ clip(.4), clip(-.2), 0
    assert abs(float(y[0, 0]) - .4) < .05 and abs(float(y[0, 1]) + .2) < .05 and float(y[0, 2]) == 0, y
    print("check ok", y.tolist())


if __name__ == "__main__":
    c = sys.argv[1]; torch.set_num_threads(1)
    {"check": check, "kpi": lambda: kpi(int(sys.argv[2]), int(sys.argv[3])), "snn": lambda: snn(int(sys.argv[2]), int(sys.argv[3]))}[c]()
