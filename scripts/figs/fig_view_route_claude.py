"""Main-text Fig.: a view holds the move, not the route. a: next-move prediction from one view (Rosetta matched, action-word-free
captions); b: closed loop driven by the image->text->action map vs the uniform policy; c: hidden-layer CKA to predicted cortex.
Dots: policy arms (a, b) or network seeds (c). -> paper/wacv2027/figs/fig_view_route"""
import collections, glob, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "data/audit/v4n_rosetta_matched_20261031"
ARMS = ("human_bc", "human_legal_uniform", "rat_bc", "cold_ppo")
C = dict(human_bc="#1b6ca8", human_legal_uniform="#9cc3e0", rat_bc="#c0392b", cold_ppo="#8c8c8c")
LAB = dict(human_bc="human", human_legal_uniform="legal", rat_bc="rat", cold_ppo="scratch")


def panel_a(ax):
    S = json.loads((D / "strata/strata.json").read_text())["arms"]; B = json.loads((D / "lexfree/analysis.json").read_text())
    per = collections.defaultdict(list)
    for k, v in B["B"].items(): per[k.split("/")[0]].append(v)
    pooled = {"open ways, junction": ("junction", "legal_uniform"), "text, junction": ("junction", "text"),
              "text, goal in sight": ("junction, goal near", "text"), "+ goal bearing": ("junction, goal near", "text+goal bearing")}
    rows = [("majority", lambda a: np.mean([x["majority"] for x in per[a]])),
            ("text, all views", lambda a: np.mean([x["text"] for x in per[a]])),
            ("open ways, junction", lambda a: S[a]["junction"]["legal_uniform"]),
            ("text, junction", lambda a: S[a]["junction"]["text"]),
            ("text, goal in sight", lambda a: S[a].get("junction, goal near", {}).get("text")),
            ("+ goal bearing", lambda a: S[a].get("junction, goal near", {}).get("text+goal bearing")),
            ("image→text→act", lambda a: B["C"][a]["composed"])]
    for i, (name, f) in enumerate(rows):
        v = {a: f(a) for a in ARMS if f(a) is not None}
        m = 100 * (S["pooled"][pooled[name][0]][pooled[name][1]] if name in pooled else np.mean(list(v.values())))   # strata rows: pooled over rows
        ax.barh(i, m, color="#cfd8e3" if name in ("majority", "open ways, junction") else "#4f7fb0", height=.7)
        for a, x in v.items(): ax.plot(100 * x, i, "o", ms=2, color=C[a], mec="k", mew=.2)
        ax.text(104, i, f"{m:.0f}", va="center", ha="right", fontsize=5, weight="bold")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r for r, _ in rows], fontsize=5.2); ax.invert_yaxis(); ax.set_xlim(0, 106)
    ax.set_xlabel("next move predicted (%)"); ax.set_title("a  one view → the move", loc="left", weight="bold", fontsize=6.2)


def panel_b(ax):
    for i, a in enumerate(ARMS):
        R = [json.loads(Path(f).read_text()) for f in glob.glob(str(D / f"lexfree/closed/{a}/seed*/cfg*.json"))]
        per = [100 * r["goals"] / r["trials"] for r in R]; ax.bar(i, np.mean(per), color=C[a], width=.7)
        ax.errorbar(i, np.mean(per), 1.96 * np.std(per, ddof=1) / np.sqrt(len(per)), color="k", lw=.6, capsize=1.5)
    ax.axhline(13.3, ls="--", lw=.6, color="k")
    ax.annotate("uniform policy", xy=(1.5, 13.3), xytext=(1.5, 34), ha="center", fontsize=4.8, arrowprops=dict(arrowstyle="-", lw=.4, color="k"))   # label clear of the bars
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([LAB[a] for a in ARMS], fontsize=5, rotation=30); ax.set_ylim(0, 50)
    ax.set_ylabel("goal reached (%)"); ax.set_title("b  views → the route", loc="left", weight="bold", fontsize=6.2)


def panel_c(ax):
    G = json.loads((D / "brains/geometry.json").read_text()); A = collections.defaultdict(list)
    for k, v in G.items():
        kind, pair, net, _ = json.loads(k)
        if kind == "P" and net and pair == "cortexfull-hid": A[net.split("/")[0]].append(v["geometry"]["cka"])
    for i, a in enumerate(ARMS):
        ax.bar(i, np.mean(A[a]), color=C[a], width=.7); ax.plot([i] * len(A[a]), A[a], "k.", ms=1.5)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([LAB[a] for a in ARMS], fontsize=5, rotation=30); ax.set_ylim(0, .7)
    ax.set_ylabel("CKA to cortex"); ax.set_title("c  brain-like vision", loc="left", weight="bold", fontsize=6.2)


def main():
    plt.rcParams.update({"font.size": 6.3, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(3.4, 2.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=.75, wspace=.55)
    panel_a(fig.add_subplot(gs[0, :])); panel_b(fig.add_subplot(gs[1, 0])); panel_c(fig.add_subplot(gs[1, 1]))
    for ax in fig.axes: ax.spines[["top", "right"]].set_visible(False)
    for ext in ("pdf", "png"): fig.savefig(ROOT / f"paper/wacv2027/figs/fig_view_route.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
