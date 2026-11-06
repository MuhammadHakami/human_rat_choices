"""Why do local (backprop-free) learners forget as much as PPO here? (author 2026-11-03). Descriptive, no new runs.
  A  trade-off: per run, old-goal success (familiar goal, trained room) vs new-goal success (moved goal, trained room);
     OLS old ~ new + local + before_old (+ prior FE): at matched learning, do local rules retain more?
  B  update footprint: |w_final - w_prior| per synapse in the shared body + heads; fraction of synapses moved > 10% of
     the layer's prior RMS, Gini of |dw|, relative change per layer. Learners with an expansion layer: body/heads only.
  python scripts/v4n_forgetting_diagnosis_claude_20261103.py trade | footprint
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_moved_goal_learning_claude_20261031 as M

A = M.ROOT / "data/audit"; K = A / "v4n_kpi_20261031/kpi"; OUT = A / "v4n_forgetting_diagnosis_20261103"
RUNS = {"ppo": (A / "v4n_moved_goal_learning_20261031/test2/ppo/registered", False), "td_head": (A / "v4n_moved_goal_learning_20261031/test2/td_head/rate0.003", True),
        "td_fa": (A / "v4n_moved_goal_learning_20261031/test2/td_fa/rate0.001", True), "head": (A / "v4n_local_search_20261031/test3/head", True),
        "dfa": (A / "v4n_local_search_20261031/test3/dfa", True), "ed": (A / "v4n_local_search_20261031/test3/ed", True),
        "s2_dfa": (A / "v4n_local_search2_20261102/test/s2_dfa", True), "s2_ed": (A / "v4n_local_search2_20261102/test/s2_ed", True),
        "s2_head_strat": (A / "v4n_local_search2_20261102/test/s2_head_strat", True)}


def rows():
    out = []
    for l, (d, local) in RUNS.items():
        for f in sorted(d.glob("*/seed*.json")):
            r = json.loads(f.read_text()); p, s = r["prior"], r["seed"]; kf = K / l / p / f"seed{s}.json"; kb = K / "before" / p / f"seed{s}.json"
            if not kf.exists(): continue
            out.append(dict(learner=l, local=local, prior=p, seed=s, new=r["curve"][str(M.BUDGET)] * 100 / 30,
                            old=json.loads(kf.read_text())["familiar_trained_room"] * 100 / 30, before=json.loads(kb.read_text())["familiar_trained_room"] * 100 / 30, f=str(f)))
    return out


def trade():
    R = rows(); OUT.mkdir(parents=True, exist_ok=True)
    pri = sorted({r["prior"] for r in R}); X = np.array([[1, r["new"], float(r["local"]), r["before"]] + [float(r["prior"] == p) for p in pri[1:]] for r in R]); y = np.array([r["old"] for r in R])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    seeds = np.array([r["seed"] for r in R]); XtXi = np.linalg.inv(X.T @ X)                        # cluster-robust by seed
    meat = sum(np.outer(X[seeds == s].T @ e[seeds == s], X[seeds == s].T @ e[seeds == s]) for s in np.unique(seeds)); se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    bins = [(0, 25), (25, 50), (50, 75), (75, 101)]; tab = {}
    for lo, hi in bins:
        for loc in (False, True):
            v = [r["old"] for r in R if lo <= r["new"] < hi and r["local"] == loc]; tab[f"{lo}-{hi}/{'local' if loc else 'ppo'}"] = [round(float(np.mean(v)), 1) if v else None, len(v)]
    res = dict(n=len(R), coef=dict(zip(["const", "new", "local", "before_old"] + pri[1:], b.tolist())), se=dict(zip(["const", "new", "local", "before_old"] + pri[1:], se.tolist())), binned_old=tab,
               per_learner={l: [round(float(np.mean([r[k] for r in R if r["learner"] == l])), 1) for k in ("new", "old", "before")] for l in RUNS})
    (OUT / "trade.json").write_text(json.dumps(dict(res, rows=R), indent=1) + "\n"); print(json.dumps(res, indent=1))


def footprint():
    R = rows(); res = {}
    for l in RUNS:
        stats = []
        for r in [x for x in R if x["learner"] == l][:30]:
            m = M.build(r["prior"], r["seed"]); w0 = {k: v.detach().cpu() for k, v in m.net.state_dict().items()}
            w1 = torch.load(Path(r["f"]).with_suffix(".final.pt"), map_location="cpu")
            w1 = {k.replace(".base.", "."): v for k, v in w1.items()}
            row = {}
            for k in w0:
                if k not in w1 or w0[k].dim() < 2: continue
                d = (w1[k] - w0[k]).abs().flatten(); rms = w0[k].pow(2).mean().sqrt()
                ds = torch.sort(d).values; n = len(ds); gini = float((2 * torch.arange(1, n + 1) - n - 1).float() @ ds / (n * ds.sum() + 1e-12))
                row[k] = dict(moved=float((d > .1 * rms).float().mean()), rel=float(d.norm() / (w0[k].norm() + 1e-12)), gini=gini)
            stats.append(row)
        res[l] = {k: {q: round(float(np.mean([s[k][q] for s in stats])), 3) for q in ("moved", "rel", "gini")} for k in stats[0]}
    (OUT / "footprint.json").write_text(json.dumps(res, indent=1) + "\n")
    for l, v in res.items(): print(l, {k.split(".")[0] + k.split(".")[1][:1] if "." in k else k: (v[k]["moved"], v[k]["rel"], v[k]["gini"]) for k in v})


if __name__ == "__main__":
    {"trade": trade, "footprint": footprint}[sys.argv[1]]()
