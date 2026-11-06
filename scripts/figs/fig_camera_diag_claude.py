"""Camera-ready diagnostics figure (moved goals, loops, evaluation policy, pose belief, held-out imitation).
Reads data/audit/v4n_camera_ready_20261031/{analysis,bc_heldout}.json.
  python scripts/figs/fig_camera_diag_claude.py -> paper/wacv2027/figs/fig_camera_diag.pdf
"""
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "data/audit/v4n_camera_ready_20261031"
A = json.loads((D / "analysis.json").read_text()); P = A["per_seed"]; BC = json.loads((D / "bc_heldout.json").read_text())
ARMS = [("human_bc", "H choice", "#1b6ca8"), ("human_legal_uniform", "H legal", "#9cc3e0"), ("rat_bc", "R choice", "#c0392b"),
        ("rat_legal_uniform", "R legal", "#e6a49c"), ("cold_ppo", "none", "#8c8c8c")]
PP = 100 / 30


def ci(v):
    v = np.asarray(v, float); h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return v.mean(), h


def vals(fam, arm, m, scale=1.):
    return [x[m] * scale for x in P[fam][arm].values()]


def bars(ax, groups, labels, width=.8):
    """groups: list (per x-group) of list (per arm) of seed-value lists."""
    k = len(ARMS); w = width / k
    for g, row in enumerate(groups):
        for i, v in enumerate(row):
            m, h = ci(v); x = g + (i - (k - 1) / 2) * w
            ax.bar(x, m, w * .92, color=ARMS[i][2], ec="none"); ax.plot([x, x], [m - h, m + h], color="0.15", lw=.6)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels)


def main():
    plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 5, figsize=(7.0, 1.85), gridspec_kw=dict(width_ratios=[.75, 1.3, 1.3, 1, 1.2], wspace=.55))
    f = "registered_sample_source"
    a = ax[0]; bars(a, [[vals(f, x, "trapped", 100) for x, _, _ in ARMS]], ["familiar goal"])
    a.set_ylabel("trapped trials (%)"); a.set_ylim(0, 100); a.set_title("a  loops", loc="left", weight="bold")
    a = ax[1]; bars(a, [[vals(f, x, m, 100) for x, _, _ in ARMS] for m in ("revisit_rate", "contact_rate")], ["revisits", "wall bumps"])
    a.set_ylabel("% of moves"); a.set_ylim(0, 100); a.set_title("b  how they move", loc="left", weight="bold")
    a = ax[2]
    if "sample_moved" in P:
        bars(a, [[vals("sample_moved", x, m, 100) for x, _, _ in ARMS] for m in ("visited_source_goal", "visited_moved_goal")], ["visit old\ngoal (3,3)", "visit new\ngoal"])
    a.set_ylabel("% of trials"); a.set_ylim(0, 100); a.set_title("c  moved goal", loc="left", weight="bold")
    a = ax[3]; conds = ("generality_clean", "generality_retextured", "generality_relit")
    for j, c in enumerate(conds):
        v = [np.mean([P[c][x][s]["pose_top1"] for x in P[c]]) * 100 for s in P[c]["human_bc"]]; m, h = ci(v)
        a.bar(j, m, .6, color="0.55"); a.plot([j, j], [m - h, m + h], color="0.15", lw=.6)
    a.set_xticks(range(3)); a.set_xticklabels(["clean", "tex.", "dim"], fontsize=5.6); a.set_ylim(0, 100); a.set_ylabel("belief top-1 (%)")
    a.set_title("d  pose belief", loc="left", weight="bold")
    a = ax[4]
    for j, (sp, col) in enumerate((("rat", "#c0392b"), ("human", "#1b6ca8"))):
        rows = BC[f"{sp}/trial"]["rows"]
        for i, (k, alpha) in enumerate((("real", 1.), ("legal_uniform", .4), ("expert", .15))):
            m, h = ci([r[k]["acc"] * 100 for r in rows]); x = j + (i - 1) * .27
            a.bar(x, m, .25, color=col, alpha=alpha, ec=col, lw=.4); a.plot([x, x], [m - h, m + h], color="0.15", lw=.6)
    a.set_xticks([0, 1]); a.set_xticklabels(["rat", "human"]); a.set_ylim(0, 100); a.set_ylabel("held-out accuracy (%)")
    a.set_title("e  imitation", loc="left", weight="bold")
    a.text(1.42, 96, "solid: recorded\npale: legal\nfaint: shortest path", fontsize=5, va="top", ha="right")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in ARMS], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.42, 1.1), fontsize=6)
    for ext in ("pdf", "png"): fig.savefig(ROOT / f"paper/wacv2027/figs/fig_camera_diag.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
