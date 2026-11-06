"""Camera-ready navigation showcase: what the agent saw, where it believed it was, and where it went.
Selection rules (seed 1000, fresh room pair 0 / A): row 1 = the human-choice agent's first SOLVED trial of >=10 moves; row 2 = its
first FAILED trial (shows the typical failure: a loop); rows 3-5 = the same start and texture as row 1 for the legal-label
agent, the human-choice agent with a moved goal (frozen), and after learning that goal with TD-DFA (local rule).
Views are re-rendered with the registered renderer; the belief is replayed with the registered filter and checked
against the recorded pose audit (familiar-goal rows).
  python scripts/figs/fig_camera_showcase_claude.py -> paper/wacv2027/figs/fig_camera_showcase.pdf
"""
import gzip, json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import fresh_rooms_confirm_claude as FR
from arc_fly.unimaze import NB
W = FR.worlds; fresh = W.fresh
CR = ROOT / "data/audit/v4n_camera_ready_20261031"; ML = ROOT / "data/audit/v4n_moved_goal_learning_20261031"
SEED, PAIR, STAGE, NFR = 1000, 0, "A", 6
CFG = 100 + 2 * PAIR


def episodes(f):
    if str(f).endswith(".paths.json.gz"):
        c = json.loads(gzip.decompress(Path(f).read_bytes())); return c[max(c, key=int)]
    return [e for e in json.loads(gzip.decompress(Path(f).read_bytes()))["episodes"] if "spawn" in e]


def replay(cache, atlas, ep):
    """(frames, beliefs, headings) along the path, with the registered filter semantics."""
    filt = fresh.LogBelief(atlas, .03); path = [tuple(p) for p in ep["path"]]; acts = ep["actions"]
    heading = NB[fresh.source.inward_heading(tuple(ep["spawn"]))]; out = []
    for t, cell in enumerate(path):
        if t: heading = NB[acts[t - 1]]
        x = cache.frame(CFG, cell, heading, ep["rotation_index"])
        contact = bool(t and path[t] == path[t - 1])
        b = filt.observe(x[:64], None if t == 0 else int(acts[t - 1]), contact)
        img = fresh.render(cache.view, cache.worlds[CFG], cell, heading, cache.rotations[ep["rotation_index"]])[0]
        out.append((img, b.reshape(10, 10), heading))
    return out


def rows():
    hf = FR.D / f"cohort/main/human_bc/seed{SEED}/pair{PAIR}_{STAGE}.full.json.gz"; H = episodes(hf)
    ok = next(e for e in H if e["success"] and e["decisions"] >= 10); bad = next(e for e in H if not e["success"])
    key = lambda e: (tuple(e["spawn"]), e["rotation_index"]); same = lambda f: next((e for e in episodes(f) if key(e) == key(ok)), None)
    proto = json.loads((CR / "PROTOCOL.json").read_text()); mg = tuple(proto["moved_goals"]["(6, 6)"][str(CFG)])
    R = [("human-choice prior: a solved trial", ok, (3, 3), "#1b6ca8"), ("human-choice prior: a failed trial (loop)", bad, (3, 3), "#1b6ca8"),
         ("legal-label prior, same start", same(FR.D / f"cohort/main/human_legal_uniform/seed{SEED}/pair{PAIR}_{STAGE}.full.json.gz"), (3, 3), "#5f8fb4")]
    f = CR / f"cohort/sample_moved/goal66/human_bc/seed{SEED}/pair{PAIR}_{STAGE}.full.json.gz"
    if f.exists(): R.append(("human-choice prior, moved goal (frozen)", same(f), mg, "#0b3d63"))
    g = next(iter(sorted(ML.glob(f"test2/td_fa/*/human_bc/seed{SEED}.paths.json.gz"))), None)
    if g is not None and json.loads(Path(str(g).replace(".paths.json.gz", ".json")).read_text())["pair"] == PAIR:
        R.append(("after learning it with TD-DFA (local rule)", same(g), mg, "#2e8b57"))
    return [r for r in R if r[1] is not None]


def main():
    _, _, source = fresh.load(); cache = W.RoomCache(source, json.loads(FR.MAN.read_text()))
    atlas = source.raw.features[0, :, :, :, 0, :64].reshape(100, 4, 64); walk = cache.worlds[CFG]
    R = rows(); plt.rcParams.update({"font.size": 6.2, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.0, 1.6 * len(R)))
    gs = fig.add_gridspec(3 * len(R), NFR + 1, width_ratios=[1.3] + [1.0] * NFR, height_ratios=[.2, 1, .62] * len(R), wspace=.06, hspace=.3)   # row 0 of each panel: its title
    for r, (lab, ep, goal, col) in enumerate(R):
        seq = replay(cache, atlas, ep); path = np.array(ep["path"]); T = len(path) - 1
        succ = tuple(path[-1]) == goal
        hd = fig.add_subplot(gs[3 * r, :]); hd.axis("off")
        hd.text(0, .1, f"{'abcde'[r]}  {lab}", transform=hd.transAxes, fontsize=6.4, color=col, weight="bold", ha="left", va="bottom")
        m = fig.add_subplot(gs[3 * r + 1:3 * r + 3, 0]); m.imshow(np.where(walk, .93, .3), cmap="gray", vmin=0, vmax=1)
        m.scatter(path[:, 1], path[:, 0], c=np.arange(len(path)), cmap="viridis", s=4, zorder=3)
        m.plot(path[:, 1], path[:, 0], "-", color=col, lw=.8, alpha=.7); m.plot(path[0, 1], path[0, 0], "s", color="k", ms=3)
        m.plot(goal[1], goal[0], "*", ms=8, mfc="gold", mec="k", mew=.4)
        if goal != (3, 3): m.plot(3, 3, "o", ms=4, mfc="none", mec="gold", mew=.8)
        m.set_xticks([]); m.set_yticks([])
        m.set_title(f"{'reached' if succ else 'not reached'} ({T} moves)", fontsize=5.4, pad=2)
        for k, t in enumerate(np.linspace(0, T, NFR).round().astype(int)):
            img, b, h = seq[t]
            a = fig.add_subplot(gs[3 * r + 1, 1 + k]); a.imshow(img, cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"t={t}", fontsize=5.2, pad=1.5)
            q = fig.add_subplot(gs[3 * r + 2, 1 + k]); q.imshow(np.where(walk, .93, .3), cmap="gray", vmin=0, vmax=1)
            q.imshow(np.ma.masked_less(b, .02), cmap="Reds", vmin=0, vmax=.6, alpha=1.)
            q.plot(path[t, 1], path[t, 0], "s", mfc="none", mec="#00b3b3", ms=4.2, mew=.7); q.set_xticks([]); q.set_yticks([])
            if k == 0: q.set_ylabel("belief", fontsize=5)
    for ext in ("pdf", "png"): fig.savefig(ROOT / f"paper/wacv2027/figs/fig_camera_showcase.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
