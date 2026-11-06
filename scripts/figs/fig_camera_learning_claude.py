"""Camera-ready learning curves for the moved goal: PPO, TD-readout, TD-DFA x three priors (30 seeds, mean +- 95% CI).
  python scripts/figs/fig_camera_learning_claude.py -> paper/wacv2027/figs/fig_camera_learning.pdf
"""
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "data/audit/v4n_moved_goal_learning_20261031/test2"
LEARN = [("ppo", "PPO (backprop)"), ("td_head", "TD-readout (local)"), ("td_fa", "TD-DFA (local, all layers)")]
PRI = [("cold_ppo", "scratch", "#8c8c8c"), ("human_legal_uniform", "human legal labels", "#9cc3e0"), ("human_bc", "human choices", "#1b6ca8")]


def main():
    plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.7), sharey=True)
    for a, (l, name) in zip(ax, LEARN):
        for p, lab, col in PRI:
            rs = [json.loads(f.read_text()) for f in sorted(D.glob(f"{l}/*/{p}/seed*.json"))]
            if not rs: continue
            xs = sorted(int(k) for k in rs[0]["curve"]); Y = np.array([[r["curve"][str(x)] * 100 / 30 for x in xs] for r in rs])
            m = Y.mean(0); h = stats.t.ppf(.975, len(Y) - 1) * Y.std(0, ddof=1) / np.sqrt(len(Y))
            a.plot(np.array(xs) / 1000, m, "-o", ms=2.2, color=col, lw=1, label=lab); a.fill_between(np.array(xs) / 1000, m - h, m + h, color=col, alpha=.18, lw=0)
        a.set_title(f"{name}  (n={len(rs)})", fontsize=6.5); a.set_xlabel("interactions on the new goal (thousands)")
    ax[0].set_ylabel("success on moved goal (%)"); ax[0].legend(frameon=False, fontsize=5.8, loc="upper left")
    for ext in ("pdf", "png"): fig.savefig(ROOT / f"paper/wacv2027/figs/fig_camera_learning.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
