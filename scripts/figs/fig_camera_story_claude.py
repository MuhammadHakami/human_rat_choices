"""Camera-ready storyboard (page 1): demonstrators -> inherited behaviour -> transfer -> learning.
a  real humans and rats: revisit rate of their own recorded moves (per subject; de Cothi et al. data)
b  frozen agents at the same states: revisit rate in 16 fresh rooms (familiar goal; registered records)
c  success of the same agents, familiar goal vs moved goals (frozen)
d  learning a moved goal: normalised learning-curve AUC per learner (PPO, local rules), human choices vs legal vs scratch
  python scripts/figs/fig_camera_story_claude.py -> paper/wacv2027/figs/fig_camera_story.pdf
"""
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CR = json.loads((ROOT / "data/audit/v4n_camera_ready_20261031/analysis.json").read_text())["per_seed"]
HUM, RAT, GREY = "#1b6ca8", "#c0392b", "#8c8c8c"
AG = [("human_bc", "H\nchoice", HUM, 1.), ("human_legal_uniform", "H\nlegal", HUM, .35), ("rat_bc", "R\nchoice", RAT, 1.),
      ("rat_legal_uniform", "R\nlegal", RAT, .35), ("cold_ppo", "none", GREY, 1.)]


def ci(v):
    v = np.asarray(v, float); return v.mean(), (stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)


def native():
    T = json.loads((ROOT / "data/unified/tartarus_trajectories.json").read_text())["trials"]; out = {}
    for t in T:
        c = [tuple(x) for x in t["cells"]]; seen = {c[0]}; rv = 0
        for x in c[1:]: rv += x in seen; seen.add(x)
        out.setdefault(t["species"], {}).setdefault(t["subject"], []).append(rv / max(len(c) - 1, 1))
    return {sp: [100 * np.mean(v) for v in d.values()] for sp, d in out.items()}


def bars(ax, fam, metric, scale, jitter=True):
    for i, (a, lab, col, al) in enumerate(AG):
        if fam not in CR or a not in CR[fam]: continue
        v = [x[metric] * scale for x in CR[fam][a].values()]; m, h = ci(v)
        ax.bar(i, m, .7, color=col, alpha=al, ec=col, lw=.6); ax.plot([i, i], [m - h, m + h], color="0.1", lw=.7)
        if jitter: ax.scatter(i + np.random.default_rng(i).uniform(-.22, .22, len(v)), v, s=1.2, color="0.25", zorder=3, lw=0)
    ax.set_xticks(range(len(AG))); ax.set_xticklabels([x[1] for x in AG], fontsize=5.4)


def main():
    plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.0), gridspec_kw=dict(width_ratios=[.6, 1.2, 1.2, 1.2], wspace=.42))
    a = ax[0]; N = native()
    for i, (sp, col) in enumerate((("human", HUM), ("rat", RAT))):
        v = N[sp]; m, h = ci(v); a.bar(i, m, .65, color=col); a.plot([i, i], [m - h, m + h], color="0.1", lw=.7)
        a.scatter(i + np.random.default_rng(9 + i).uniform(-.2, .2, len(v)), v, s=3, color="0.2", zorder=3, lw=0)
    a.set_xticks([0, 1]); a.set_xticklabels([f"humans\n(n={len(N['human'])})", f"rats\n(n={len(N['rat'])})"], fontsize=5.6)
    a.set_ylabel("moves revisiting a cell (%)"); a.set_title("a  demonstrators", loc="left", weight="bold", fontsize=6.6)
    a = ax[1]; bars(a, "registered_sample_source", "revisit_rate", 100); a.set_ylim(40, 80); a.set_ylabel("moves revisiting a cell (%)")
    a.set_title("b  agents, same states", loc="left", weight="bold", fontsize=6.6)
    a = ax[2]
    for k, (fam, mk, lab) in enumerate((("registered_sample_source", "o", "familiar goal"), ("sample_moved", "D", "moved goals"))):
        if fam not in CR: continue
        for i, (ag, _, col, al) in enumerate(AG):
            if ag not in CR[fam]: continue
            m, h = ci([x["success"] * 100 / 30 for x in CR[fam][ag].values()]); x = i + (k - .5) * .32
            a.errorbar(x, m, yerr=h, fmt=mk, ms=3.2, color=col, mfc=col if k == 0 else "white", alpha=max(al, .5), capsize=1.2, lw=.8,
                       label=lab if i == 0 else None)
    a.axhline(13.3, color="0.4", ls=":", lw=.6); a.set_xticks(range(len(AG))); a.set_xticklabels([x[1] for x in AG], fontsize=5.4)
    a.set_ylabel("success, frozen (%)"); a.set_ylim(0, 60); a.legend(frameon=False, fontsize=5.4, loc="upper right"); a.set_title("c  unseen rooms", loc="left", weight="bold", fontsize=6.6)
    a = ax[3]; L = ROOT / "data/audit/v4n_moved_goal_learning_20261031/test2"
    names = [("ppo", "PPO"), ("td_head", "TD-\nreadout"), ("td_fa", "TD-\nDFA")]; xs = np.array([0, 8192, 16384 // 2 * 2, 32768], float)
    for i, (l, nm) in enumerate(names):
        for j, (p, col, al, lab) in enumerate((("human_bc", HUM, 1., "H choice"), ("human_legal_uniform", HUM, .35, "H legal"), ("cold_ppo", GREY, 1., "none"))):
            v = []
            for f in sorted((L / l).glob(f"*/{p}/seed*.json")):
                c = json.loads(f.read_text())["curve"]; ks = sorted(int(k) for k in c)
                v.append(100 * np.trapezoid([c[str(k)] for k in ks], ks) / (30 * ks[-1]))
            if v:
                m, h = ci(v); x = i + (j - 1) * .26; a.bar(x, m, .24, color=col, alpha=al, label=lab if i == 0 else None); a.plot([x, x], [m - h, m + h], color="0.1", lw=.7)
    a.set_xticks(range(len(names))); a.set_xticklabels([n for _, n in names], fontsize=5.6); a.set_ylabel("learning-curve AUC (%)")
    a.legend(frameon=False, fontsize=5.2, loc="upper right")
    a.set_title("d  learning a new goal", loc="left", weight="bold", fontsize=6.6)
    fig.text(.42, -.06, "prior: H/R = human/rat; choice = recorded choices; legal = random legal moves at the same states; none = no prior", ha="center", fontsize=5.6, color="0.3")
    for ext in ("pdf", "png"): fig.savefig(ROOT / f"paper/wacv2027/figs/fig_camera_story.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
