"""Camera-ready four-claims figure on the accepted task: forgetting, OOD transfer of new learning, old skill elsewhere,
and spiking efficiency (IF conversion). Reads data/audit/v4n_kpi_20261031 and the learning records.
  python scripts/figs/fig_camera_kpi_claude.py -> paper/wacv2027/figs/fig_camera_kpi.pdf
"""
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]; K = ROOT / "data/audit/v4n_kpi_20261031"; JR = ROOT / "data/audit/v4n_joint_reference_20261101/joint"   # diamonds: trained on both goals at once
PRI = [("cold_ppo", "scratch", "#8c8c8c"), ("human_legal_uniform", "human legal", "#9cc3e0"), ("human_bc", "human choices", "#1b6ca8")]
NICE = {"before": "before", "ppo": "PPO", "td_head": "TD-readout", "s2_ed": "local+replay", "cs_dfa": "+consolidation",
        "G2": "+inferred gate", "bl3_none": "brain circuit", "bl2_inferred": "brain circuit+gate"}
A = ROOT / "data/audit"
OWN = {"G2": A / "v4n_context_gating_20261103/runs/G2", "bl3_none": A / "v4n_brainlike3_20261105/test/bl3_none",
       "bl2_inferred": A / "v4n_brainlike2_20261105/test/bl2_inferred"}   # stage 3 ungated / stage 2 gated (stage 3 gated = stage 2); stages 1-2 in the supplement   # records hold all four probes


def load(learner, prior, key):
    """{seed: % success}; key 'new' = moved goal in the trained room (learning-curve end, or the probe for OWN learners)."""
    if learner in OWN:
        R = [json.loads(f.read_text()) for f in sorted((OWN[learner] / prior).glob("seed*.json")) if f.stat().st_size]
        return {r["seed"]: r["moved_trained_room" if key == "new" else key] * 100 / 30 for r in R}
    out = {}
    for f in sorted((K / "kpi" / learner / prior).glob("seed*.json")):
        r = json.loads(f.read_text())
        if key != "new": out[r["seed"]] = r[key] * 100 / 30
        elif r.get("source"): c = json.loads(Path(r["source"]).read_text())["curve"]; out[r["seed"]] = c[str(max(map(int, c)))] * 100 / 30
    return out


def ci(v):
    v = np.asarray(v, float)
    return (v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (v.mean() if len(v) else np.nan, 0)


def panel(ax, key, title, learners):
    w = .8 / len(PRI)
    for i, l in enumerate(learners):
        for j, (p, lab, col) in enumerate(PRI):
            m, h = ci(list(load(l, p, key).values())); x = i + (j - 1) * w
            ax.bar(x, m, w * .92, color=col, alpha=.55 if l == "before" else 1); ax.plot([x, x], [m - h, m + h], color="0.15", lw=.6)
            jf = sorted((JR / l / p).glob("seed*.familiar.json")) if key == "familiar_trained_room" else []
            if jf: ax.plot(x, np.mean([json.loads(f.read_text())["familiar_trained_room"] for f in jf]) * 100 / 30, "D", ms=2.6, mfc="white", mec="k", mew=.6, zorder=5)
    ax.set_xticks(range(len(learners))); ax.set_xticklabels([NICE.get(l, l) for l in learners], fontsize=5.2, rotation=45, ha="right"); ax.set_title(title, loc="left", weight="bold", fontsize=6.2)
    ax.set_ylim(0, 100); ax.set_ylabel("success (%)")


def main():
    plt.rcParams.update({"font.size": 6.3, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    learners = ["before"] + [l for l in NICE if l != "before" and ((K / "kpi" / l).exists() or l in OWN)]
    import sys; supp = "--supp" in sys.argv                                  # supplement: old goal in the unseen room
    if supp:
        fig, ax = plt.subplots(figsize=(3.3, 2.1)); panel(ax, "familiar_heldout_room", "old goal, unseen room", learners)
        for ext in ("pdf", "png"): fig.savefig(ROOT / f"paper/wacv2027/figs/fig_kpi_old_unseen.{ext}", dpi=300, bbox_inches="tight")
        return
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.25), gridspec_kw=dict(width_ratios=[1.35, 1.35, 1.0, 1.0], wspace=.38))
    panel(ax[0], "familiar_trained_room", "a  old goal, trained room", learners)
    panel(ax[1], "moved_heldout_room", "b  new goal, unseen room", learners); ax[1].set_ylabel("")
    t = ax[2]                                                                 # stability-plasticity plane, learner means over priors
    MK = {"ppo": ("o", "k"), "td_head": ("o", "#9cc3e0"), "s2_ed": ("o", "#1b6ca8"), "cs_dfa": ("s", "#6aa84f"), "G2": ("*", "#c0392b"), "bl3_none": ("P", "#8e44ad"), "bl2_inferred": ("X", "#8e44ad")}
    for l, (mk, col) in MK.items():
        nw = [v for p, _, _ in PRI for v in load(l, p, "new").values()]; od = [v for p, _, _ in PRI for v in load(l, p, "familiar_trained_room").values()]
        if nw: t.plot(np.mean(nw), np.mean(od), mk, color=col, ms=6 if mk == "*" else 4.5, mec="white", mew=.4, zorder=4, label=NICE[l])
    J = A / "v4n_joint_reference_20261101/analysis.json"
    if J.exists():
        for l in ("ppo", "td_head"):
            for v in json.loads(J.read_text())[l]["per_prior"].values(): t.plot(v["joint"][1], v["joint"][0], "D", mfc="white", mec="k", ms=2.8, mew=.5)
    t.set_xlim(0, 105); t.set_ylim(0, 100); t.set_xlabel("new goal, trained room (%)"); t.set_ylabel("old goal kept (%)")
    t.plot([], [], "D", mfc="white", mec="k", ms=2.8, mew=.5, label="both goals at once")
    t.legend(frameon=False, fontsize=4.4, loc="upper center", bbox_to_anchor=(.5, -.24), ncol=2, handletextpad=.2, columnspacing=.8, borderaxespad=0, labelspacing=.25)   # below c: clear of its points and of d
    t.set_title("c  keep vs learn", loc="left", weight="bold", fontsize=6.2)
    a = ax[3]
    for p, lab, col in PRI:
        xs, ys, hs = [], [], []
        for T in (1, 2, 4, 8, 16, 32):
            R = [json.loads(f.read_text()) for f in sorted((K / "snn" / p / f"T{T}").glob("*.json"))]
            if not R: continue
            by = {}
            for r in R: by.setdefault(r["seed"], []).append(r["goals"] * 100 / 30)
            m, h = ci([np.mean(v) for v in by.values()]); xs.append(np.mean([r["synops_per_decision"] for r in R])); ys.append(m); hs.append(h)
        if xs: a.errorbar(np.array(xs) / 1000, ys, yerr=hs, color=col, marker="o", ms=2.5, lw=.9, capsize=1.5, label=lab)
    DT = ROOT / "data/audit/v4n_decision_time_20261031/analysis.json"          # accumulate-to-bound (open markers)
    if DT.exists():
        pt = json.loads(DT.read_text())["per_theta"]
        for p, lab, col in PRI:
            th = sorted(pt, key=float); a.plot([pt[t][p]["synops_per_decision"] / 1000 for t in th], [pt[t][p]["goals"] * 100 / 30 for t in th],
                                               ls=":", lw=.8, color=col, marker="o", ms=2.8, mfc="white", mec=col, mew=.7)
    a.axvline(4.416, color="0.3", ls="--", lw=.6); a.text(4.6, 3, "dense hidden\nMACs", fontsize=5)
    a.set_xscale("log"); a.set_xlabel("synaptic ops / decision (k)"); a.set_ylabel("success (%)"); a.set_ylim(0, 100)
    a.set_xlim(.8, 4000); a.set_ylim(0, 104); a.set_title("d  spiking", loc="left", weight="bold", fontsize=6.2)
    SL = A / "v4n_snn_learned_20261103/summary.json"                     # learned networks, T=16 (crosses)
    if SL.exists():
        for l, v in json.loads(SL.read_text()).items():
            if l.endswith("/T16") and l.split("/")[0] in MK: mk, col = MK[l.split("/")[0]]; a.plot(v["synops"] / 1000, v["spiking"], "x", color=col, ms=3.5, mew=.9)
    if 0: a.set_title("d  spiking: fixed T (filled), to bound (open)", loc="left", weight="bold", fontsize=6.2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for _, l, c in PRI], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.45, 1.1))
    for ext in ("pdf", "png"): fig.savefig(ROOT / f"paper/wacv2027/figs/fig_camera_kpi.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
