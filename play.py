"""Play with the agents and the animals.

  python play.py agents  [--seed 1000] [--room 0] [--unseen] [--goal familiar|moved] [--trial 0] [--models ...] [--show]
      Same start, same unseen room: agents cloned from HUMAN choices, RAT choices, random LEGAL moves at the same
      states, and a scratch PPO agent, each acting from its own first-person view. Writes outputs/agents_*.gif.
  python play.py animals [--config 1] [--start 1] [--show]
      The recorded people (virtual maze) and rats (physical maze) in the same configuration (de Cothi et al. 2022).
      Writes outputs/animals_*.gif and .png. Needs `bash setup.sh` (downloads the raw data). Default: the configuration
      and start with the most trials completed by both species (16 / 9).
  python play.py gallery
      Regenerates the GIFs shown in the README.
Models: seed 1000 of every prior ships with the repository; `bash setup.sh --models` adds all 30 seeds.
"""
import argparse, json, sys
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
OUT = ROOT / "outputs"
MODELS = {"human_bc": ("human choices", "#1b6ca8"), "rat_bc": ("rat choices", "#c0392b"),
          "human_legal_uniform": ("legal moves at human states", "#7fa6c6"), "rat_legal_uniform": ("legal moves at rat states", "#d9928a"),
          "rat_sham": ("rat states, shuffled choices", "#a0522d"), "cold_ppo": ("no demonstrator (PPO)", "#8c8c8c")}
SPECIES = {"human": "#1b6ca8", "rat": "#c0392b"}


def maze_axes(ax, walk, goal, title, col):
    ax.imshow(np.where(walk, .95, .28), cmap="gray", vmin=0, vmax=1); ax.plot(goal[1], goal[0], "*", ms=13, mfc="gold", mec="k", mew=.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, color=col, fontsize=9, weight="bold")


def save(anim, fig, name, show):
    OUT.mkdir(exist_ok=True)
    if show: import matplotlib.pyplot as plt; plt.show(); return
    from matplotlib.animation import PillowWriter
    anim.save(OUT / name, writer=PillowWriter(fps=4), dpi=90); print("wrote", OUT / name)


def cpu_fallback():
    """The registered code pins cuda:0 (and stays byte-identical); for playing only, remap it to the CPU."""
    import torch
    if torch.cuda.is_available(): return
    fix = lambda a: tuple("cpu" if isinstance(x, str) and x.startswith("cuda") else x for x in a)
    for cls in (torch.nn.Module, torch.Tensor):
        orig = cls.to; cls.to = lambda self, *a, _o=orig, **k: _o(self, *fix(a), **{**k, **({"device": "cpu"} if str(k.get("device", "")).startswith("cuda") else {})})
    torch.cuda.get_device_name = lambda *a: "cpu"


def agents(seed=1000, room=0, unseen=False, goal="familiar", trial=-1, models=("human_bc", "rat_bc", "human_legal_uniform", "cold_ppo"), show=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import fresh_rooms_confirm_claude as FR
    import v4n_moved_goal_learning_claude_20261031 as M
    from arc_fly.unimaze import NB
    cpu_fallback(); W = FR.worlds; fresh = W.fresh; stage = "B" if unseen else "A"; cfg = 100 + 2 * room + int(unseen)
    gm = M.goal_map(FR.MAN) if goal == "moved" else {c: FR.GOAL for c in M.goal_map(FR.MAN)}; target = tuple(gm[cfg])
    runs = {}
    for k in models:
        ck = ROOT / f"data/wacv/seeds30/cohort/{k}/seed{seed}/checkpoint_source.pt"
        if not ck.exists(): sys.exit(f"missing {ck.relative_to(ROOT)}: run `bash setup.sh --models` for seeds other than 1000")
        r = M.probe_now(M.build(k, seed), FR.MAN, gm, room, stage, seed); runs[k] = r
        print(f"{MODELS[k][0]:32s} reached the goal in {sum(e['success'] for e in r['episodes'])}/30 trials")
    if trial < 0:                                           # README rule: the first-listed agent's first solved trial of >= 10 moves
        ok = [i for i, (e, q) in enumerate(zip(runs[models[0]]["episodes"], runs[models[0]]["paths"])) if e["success"] and len(q["path"]) > 10]
        trial = ok[0] if ok else 0
    ref = runs[models[0]]["paths"][trial]; key = (tuple(ref["spawn"]), ref["rotation_index"])
    eps = {k: next(p for p in r["paths"] if (tuple(p["spawn"]), p["rotation_index"]) == key) for k, r in runs.items()}
    _, _, source = fresh.load(); cache = W.RoomCache(source, json.loads(FR.MAN.read_text())); walk = cache.worlds[cfg]; rot = cache.rotations[ref["rotation_index"]]
    def views(ep):
        h = NB[fresh.source.inward_heading(tuple(ep["spawn"]))]; out = []
        for t, cell in enumerate(ep["path"]):
            if t: h = NB[ep["actions"][t - 1]]
            out.append(fresh.render(cache.view, walk, tuple(cell), h, rot)[0])
        return out
    V = {k: views(e) for k, e in eps.items()}; T = max(len(e["path"]) for e in eps.values())
    fig, ax = plt.subplots(2, len(models), figsize=(2.6 * len(models), 4.6), gridspec_kw=dict(height_ratios=[.62, 1], hspace=.08))
    ax = np.atleast_2d(ax).reshape(2, -1); arts = []
    for j, k in enumerate(models):
        lab, col = MODELS[k]; ax[0, j].set_xticks([]); ax[0, j].set_yticks([]); ax[0, j].set_title(lab, color=col, fontsize=9, weight="bold")
        im = ax[0, j].imshow(V[k][0], cmap="gray", vmin=0, vmax=1)
        maze_axes(ax[1, j], walk, target, "", col); s0 = eps[k]["path"][0]; ax[1, j].plot(s0[1], s0[0], "s", ms=8, color="k")
        ln, = ax[1, j].plot([], [], "-", color=col, lw=2.2, alpha=.85)
        dot, = ax[1, j].plot([], [], "o", ms=9, color=col, mec="k"); txt = ax[1, j].set_xlabel("", fontsize=9)
        arts.append((k, im, ln, dot, txt))
    fig.suptitle(f"same start in fresh room {room}{stage} (never seen in training), {goal} goal, seed {seed}, trial {trial}\ntop: the agent's first-person view   bottom: its path", fontsize=9)
    def frame(t):
        for k, im, ln, dot, txt in arts:
            p = np.asarray(eps[k]["path"]); u = min(t, len(p) - 1); im.set_data(V[k][u])
            ln.set_data(p[:u + 1, 1], p[:u + 1, 0]); dot.set_data([p[u, 1]], [p[u, 0]])
            done = u == len(p) - 1; ok = tuple(p[-1]) == target
            txt.set_text(f"move {u}" + ((" - reached" if ok else " - time out") if done else "")); txt.set_color("green" if done and ok else ("crimson" if done else "black"))
        return []
    save(FuncAnimation(fig, frame, frames=T + 4), fig, f"agents_seed{seed}_room{room}{stage}_{goal}_trial{trial}.gif", show)


def animals(config=16, start=9, show=False):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from arc_fly.unified_env import load_tasks
    f = ROOT / "data/unified/tartarus_trajectories.json"
    if not f.exists(): sys.exit("run `bash setup.sh` first: it downloads the recorded trajectories from Zenodo")
    task = {t.id: t for t in load_tasks(ROOT / "data/unified/tartarus/tasks.json")}[f"tartarus/{config}/{start}"]
    T = [t for t in json.loads(f.read_text())["trials"] if t["task_id"] == task.id]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 4.0)); pick = {}
    for a, sp in zip(ax, SPECIES):
        tr = [t for t in T if t["species"] == sp]; col = SPECIES[sp]
        rev = np.mean([sum(c in map(tuple, t["cells"][:i]) for i, c in enumerate(map(tuple, t["cells"])) if i) / max(len(t["cells"]) - 1, 1) for t in tr]) * 100
        maze_axes(a, task.walkable, task.goal, f"{sp}s: {sum(t['success'] for t in tr)}/{len(tr)} reached, {rev:.0f}% revisits", col)
        occ = np.zeros(task.walkable.shape)
        for t in tr:
            for c in {tuple(x) for x in t["cells"]}: occ[c] += 1 / len(tr)
        a.imshow(np.ma.masked_equal(occ, 0), cmap="Blues" if sp == "human" else "Reds", vmin=0, vmax=1.4, alpha=.8)
        a.plot(task.spawn[1], task.spawn[0], "s", ms=8, color="k"); pick[sp] = next((t for t in tr if t["success"]), tr[0])
    fig.suptitle(f"recorded trials, configuration {config}, start {start} (de Cothi et al. 2022)\nshade: share of trials that entered a cell   line: one successful trial", fontsize=9)
    OUT.mkdir(exist_ok=True); fig.savefig(OUT / f"animals_config{config}_start{start}.png", dpi=110, bbox_inches="tight"); print("wrote", OUT / f"animals_config{config}_start{start}.png")
    dots = {sp: ax[i].plot([], [], "o", ms=10, color=SPECIES[sp], mec="k")[0] for i, sp in enumerate(SPECIES)}
    lines = {sp: ax[i].plot([], [], "-", lw=2.4, color=SPECIES[sp])[0] for i, sp in enumerate(SPECIES)}
    n = max(len(p["cells"]) for p in pick.values())
    def frame(t):
        for sp, p in pick.items():
            c = np.asarray(p["cells"]); u = min(t, len(c) - 1); lines[sp].set_data(c[:u + 1, 1], c[:u + 1, 0]); dots[sp].set_data([c[u, 1]], [c[u, 0]])
        return []
    save(FuncAnimation(fig, frame, frames=n + 4), fig, f"animals_config{config}_start{start}.gif", show)


def gallery():
    """README assets (fixed choices): seed 1000, fresh room 0A, first solved trial; familiar and moved goal; animals 16/9."""
    global OUT; OUT = ROOT / "assets"; agents(); agents(goal="moved"); animals()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); s = p.add_subparsers(dest="cmd", required=True)
    a = s.add_parser("agents"); a.add_argument("--seed", type=int, default=1000); a.add_argument("--room", type=int, default=0, choices=range(8))
    a.add_argument("--unseen", action="store_true", help="second room of the pair (never used for learning)"); a.add_argument("--goal", default="familiar", choices=("familiar", "moved"))
    a.add_argument("--trial", type=int, default=-1, choices=range(-1, 30), help="-1: first solved trial (>=10 moves) of the first model"); a.add_argument("--models", nargs="+", default=["human_bc", "rat_bc", "human_legal_uniform", "cold_ppo"], choices=list(MODELS))
    a.add_argument("--show", action="store_true")
    b = s.add_parser("animals"); b.add_argument("--config", type=int, default=16, choices=range(1, 26)); b.add_argument("--start", type=int, default=9, choices=range(1, 11)); b.add_argument("--show", action="store_true")
    s.add_parser("gallery"); q = p.parse_args()
    if not getattr(q, "show", False): matplotlib.use("Agg")
    {"agents": lambda: agents(q.seed, q.room, q.unseen, q.goal, q.trial, tuple(q.models), q.show),
     "animals": lambda: animals(q.config, q.start, q.show), "gallery": gallery}[q.cmd]()
