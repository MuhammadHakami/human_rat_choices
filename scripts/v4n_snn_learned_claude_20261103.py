"""Spiking efficiency of the LEARNED networks (camera-ready, author 2026-11-03). GPU 1. Descriptive.
The KPI spiking panel converted the frozen priors. Here the networks after moved-goal learning are converted the same way
(signed integrate-and-fire body, v4n_kpi.SpikingBody, T in {8, 16, 32}) and probed on the learned task (moved goal, trained
room, 30 trials). Synaptic operations per decision count every spike's fan-out; with an expansion layer, each last-layer
spike also reaches all K expansion units and the active units reach the 5 outputs (the expansion code is computed from the
time-averaged spikes). Dense reference: the network's own final success and dense hidden MACs (+ 64 K for the expansion).
Learners: ppo, td_head, td_fa (registered), s2_dfa, s2_ed, s2_head_strat (search2); seeds 1000-1009, 3 priors.
  python scripts/v4n_snn_learned_claude_20261103.py run W N | summary
"""
import json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_kpi_claude_20261031 as K
import v4n_local_search2_claude_20261102 as S2
M = K.M

OUT = M.ROOT / "data/audit/v4n_snn_learned_20261103"; TS = (8, 16, 32); SEEDS = tuple(range(1000, 1010))
DIRS = {"ppo": M.OUT / "test2/ppo/registered", "td_head": M.OUT / "test2/td_head/rate0.003", "td_fa": M.OUT / "test2/td_fa/rate0.001",
        "s2_dfa": S2.OUT / "test/s2_dfa", "s2_ed": S2.OUT / "test/s2_ed", "s2_head_strat": S2.OUT / "test/s2_head_strat"}


class SpikingBodyX(K.SpikingBody):
    def __init__(self, body, T, extra): super().__init__(body, T); self.extra = extra

    @torch.no_grad()
    def forward(self, x):
        before = K.COUNT["spikes"]; out = super().forward(x)
        if self.extra:                                                       # last-layer spikes also fan out to the K expansion units
            K.COUNT["spikes"] += float(out.abs().sum() * self.T * self.extra[0]) + self.extra[1] * len(x)
        return out


def one(l, p, s, T):
    f = OUT / l / p / f"seed{s}_T{T}.json"
    if f.exists(): return
    src = DIRS[l] / p / f"seed{s}.json"; r = json.loads(src.read_text()); m = M.build(p, s); cfg = r.get("cfg")
    if cfg and cfg.get("expand"): S2.expand(m.net, cfg["expand"], cfg["kwta"], s)
    m.net.load_state_dict({k: v.to(next(m.net.parameters()).device) for k, v in torch.load(src.with_suffix(".final.pt")).items()})
    extra = (cfg["expand"], max(1, int(round(cfg["kwta"] * cfg["expand"]))) * 5) if cfg and cfg.get("expand") else None
    dense = m.net.body; m.net.body = SpikingBodyX(dense, T, extra); K.COUNT.update(spikes=0., decisions=0)
    g = K.probe(m, M.goal_map(M.FR.MAN), s % 8, "A", s); m.net.body = dense
    macs = 252 * 64 + 64 * 64 + 64 * 5 + ((64 * cfg["expand"] + extra[1]) if extra else 0)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(learner=l, prior=p, seed=s, T=T, goals=g, dense_goals=r["curve"][str(M.BUDGET)], synops_per_decision=K.COUNT["spikes"] / max(K.COUNT["decisions"], 1),
                                 dense_macs=macs)) + "\n")


def run(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"; torch.set_num_threads(1)
    for job in [(l, p, s, T) for T in TS for s in SEEDS for p in M.PRIORS for l in DIRS][w::n]: one(*job)
    print(json.dumps(dict(worker=w, done=True)), flush=True)


def summary():
    res = {}
    for l in DIRS:
        for T in TS:
            R = [json.loads(f.read_text()) for f in (OUT / l).glob(f"*/seed*_T{T}.json")]
            if R: res[f"{l}/T{T}"] = dict(n=len(R), spiking=float(np.mean([r["goals"] for r in R]) * 100 / 30), dense=float(np.mean([r["dense_goals"] for r in R]) * 100 / 30),
                                         synops=float(np.mean([r["synops_per_decision"] for r in R])), dense_macs=float(np.mean([r["dense_macs"] for r in R])))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items(): print(k, {q: round(x, 1) for q, x in v.items()})


if __name__ == "__main__":
    {"run": lambda: run(int(sys.argv[2]), int(sys.argv[3])), "summary": summary}[sys.argv[1]]()
