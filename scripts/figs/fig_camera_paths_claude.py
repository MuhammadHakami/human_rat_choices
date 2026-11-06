"""Camera-ready qualitative figure: where frozen agents go in one fresh room (all 30 seeds x 30 trials per arm),
with the familiar goal (registered records) and a moved goal (camera-ready records). Cell shade = visits per trial.
  python scripts/figs/fig_camera_paths_claude.py [PAIR STAGE TARGET]   (defaults chosen before viewing: pair 0 room A, goal target (6,6))
"""
import gzip, json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; FRD = ROOT / "data/audit/wacv_fresh_rooms_confirm_claude"; CR = ROOT / "data/audit/v4n_camera_ready_20261031"
ARMS = [("cold_ppo", "no prior"), ("human_legal_uniform", "H legal"), ("human_bc", "H choice")]
S30 = range(1000, 1030)


def occupancy(files):
    occ = np.zeros((10, 10)); n = 0; succ = 0
    for f in files:
        for e in json.loads(gzip.decompress(f.read_bytes()))["episodes"]:
            if "spawn" not in e: continue
            for r, c in e["path"]: occ[r, c] += 1
            n += 1; succ += e["success"]
    return occ / max(n, 1), succ / max(n, 1)


def main(pair=0, stage="A", target="66"):
    man = json.loads((FRD / "fresh_geometry_manifest.json").read_text()); cfg = 100 + 2 * pair + int(stage == "B")
    walk = np.asarray(next(r for r in man["rows"] if r["configuration"] == cfg)["walkable"], bool)
    mg = tuple(json.loads((CR / "PROTOCOL.json").read_text())["moved_goals"][str(tuple(map(int, target)))][str(cfg)])
    rows = [("familiar goal (3,3)", (3, 3), lambda a: [FRD / "cohort/main" / a / f"seed{s}" / f"pair{pair}_{stage}.full.json.gz" for s in S30]),
            (f"moved goal {mg}", mg, lambda a: [CR / "cohort/sample_moved" / f"goal{target}" / a / f"seed{s}" / f"pair{pair}_{stage}.full.json.gz" for s in S30])]
    plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(2, 3, figsize=(3.3, 2.4), gridspec_kw=dict(hspace=.25, wspace=.08))
    for i, (lab, goal, files) in enumerate(rows):
        for j, (a, name) in enumerate(ARMS):
            occ, sr = occupancy(files(a)); x = ax[i, j]
            img = np.where(walk, np.log1p(occ), np.nan); x.imshow(img, cmap="magma_r", vmin=0, vmax=np.log1p(4))
            x.imshow(np.where(walk, np.nan, 1.), cmap="Greys", vmin=0, vmax=1.4)
            x.plot(goal[1], goal[0], marker="*", ms=7, mfc="#2ecc71", mec="k", mew=.4)
            if goal != (3, 3): x.plot(3, 3, marker="o", ms=4, mfc="none", mec="#2ecc71", mew=.8)
            for s in man["starts"]: x.plot(s[1], s[0], marker="s", ms=1.8, color="#1b6ca8")
            x.set_xticks([]); x.set_yticks([]); x.set_title(f"{name}: {100 * sr:.0f}%" if i == 0 else f"{100 * sr:.0f}%", fontsize=6)
            if j == 0: x.set_ylabel(lab, fontsize=6)
    for ext in ("pdf", "png"): fig.savefig(ROOT / f"paper/wacv2027/figs/fig_camera_paths.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(*(sys.argv[1:2] and [int(sys.argv[1]), sys.argv[2], sys.argv[3]]))
