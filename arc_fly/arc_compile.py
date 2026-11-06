# Copyright 2026 The ArcFly / Connectome-Drosophila authors. Apache-2.0.
"""Compile any ARC-AGI-3 frame into a 3-D fly arena.

ARC-AGI-3 hands the agent a 64x64x16-colour bird's-eye grid, which is already the right input
for building a walkable arena. The pipeline needs no per-game annotation:

  1. AGENT      = the small, moving, constant-area colour (a HUD bar has varying area, so
                  constant area -- not 2-D motion -- is the discriminator);
  2. PLAY AREA  = the bounding box of agent positions, which also crops away UI chrome
                  (on ls20 the chrome is 79% of the frame);
  3. FLOOR      = what a visited cell shows when the agent is elsewhere;
  4. WALKABLE   = cells that are ever floor-or-agent; everything else is wall;
  5. GOALS      = the remaining interactive colours, ranked by rarity, become the odour sources.

Honest coverage: 20/25 public environments are navigable in this sense. Four (bp35, ft09,
sb26, tr87) are 1-D click/selection puzzles with no spatial agent -- ft09 is ported separately
by `arc_fly.stencil`, which reproduces its real lights-out mechanic -- and lf52 has no
detectable agent.
"""
import numpy as np

MIN_PLAY = 5          # a play area smaller than this is a selector, not a maze


def agent_colour(stack, max_frac=0.05, min_move=1.0, max_area_cv=0.6):
    best, best_move = None, -1.0
    for c in np.unique(stack):
        cents, areas = [], []
        for t in range(len(stack)):
            m = stack[t] == c
            if m.any():
                ys, xs = np.nonzero(m)
                cents.append((ys.mean(), xs.mean())); areas.append(m.sum())
        if len(cents) < 5:
            continue
        areas = np.asarray(areas, float)
        if areas.mean() / stack[0].size > max_frac:
            continue
        move = float(np.std(np.asarray(cents), axis=0).mean())
        if move < min_move or areas.std() / max(areas.mean(), 1e-9) > max_area_cv:
            continue
        if move > best_move:
            best, best_move = int(c), move
    return best


def compile_frames(stack):
    """(T,64,64) frames -> dict with occupancy, spawn, goal candidates, or None if not navigable."""
    a = agent_colour(stack)
    if a is None:
        return None
    ever = np.any(stack == a, axis=0)
    ys, xs = np.nonzero(ever)
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if (r1 - r0 + 1) < MIN_PLAY or (c1 - c0 + 1) < MIN_PLAY:
        return None                                   # 1-D selector, not a maze
    vals = np.concatenate([stack[t][ever & (stack[t] != a)] for t in range(len(stack))])
    if vals.size == 0:
        return None
    floor = int(np.bincount(vals - vals.min()).argmax() + vals.min())
    sub = stack[:, r0:r1 + 1, c0:c1 + 1]
    walkable = np.any((sub == floor) | (sub == a), axis=0).astype(int)
    # goal candidates: interactive colours inside the play area that are neither floor nor agent
    first = sub[0]
    cand = {}
    for c in np.unique(first):
        if int(c) in (floor, a):
            continue
        m = first == c
        if 0 < m.sum() <= 0.15 * m.size:
            yy, xx = np.nonzero(m)
            cand[int(c)] = (int(np.median(yy)), int(np.median(xx)), int(m.sum()))
    spawn = tuple(np.argwhere(stack[0][r0:r1 + 1, c0:c1 + 1] == a)[0]) if (stack[0][r0:r1+1, c0:c1+1] == a).any() \
        else tuple(np.argwhere(walkable)[0])
    return dict(agent=a, floor=floor, bbox=(r0, r1, c0, c1), occ=walkable,
                spawn=(int(spawn[0]), int(spawn[1])), goals=cand)


def reachable_from(occ, start):
    """Connected component of walkable cells containing `start` (4-connectivity).

    `walkable` is the union of cells the agent was ever seen on, so under random play it can be
    disconnected. A goal outside the spawn's component is not a task, it is a bug."""
    from collections import deque
    occ = np.asarray(occ); H, W = occ.shape
    seen = np.zeros((H, W), bool)
    start = tuple(int(x) for x in start)
    if not occ[start]:
        return seen
    seen[start] = True; q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and occ[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True; q.append((rr, cc))
    return seen


def nearest_walkable(occ, cell):
    """Snap a cell to the closest walkable one (goal markers often sit ON a wall tile)."""
    occ = np.asarray(occ)
    w = np.argwhere(occ == 1)
    if len(w) == 0:
        raise ValueError("no walkable cells")
    d = np.abs(w - np.asarray(cell)).sum(1)
    return tuple(int(x) for x in w[d.argmin()])


def farthest_pair(occ, start):
    """Two walkable cells far from `start` and from each other (BFS-hop distance)."""
    from collections import deque
    occ = np.asarray(occ); H, W = occ.shape

    def bfs(src):
        d = -np.ones((H, W), int); d[src] = 0; q = deque([src])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and occ[rr, cc] and d[rr, cc] < 0:
                    d[rr, cc] = d[r, c] + 1; q.append((rr, cc))
        return d

    ds = bfs(tuple(start))
    a = tuple(int(x) for x in np.unravel_index(np.argmax(ds), ds.shape))
    da = bfs(a)
    both = np.where((ds >= 0) & (da >= 0), ds + da, -1)
    b = tuple(int(x) for x in np.unravel_index(np.argmax(both), both.shape))
    return a, b


def to_world(comp, cell_mm=2.5):
    """Compiled dict -> kwargs for ArcAgi3World.from_occupancy.

    Goal placement is only faithful when the game marks its goals as distinct floor colours.
    Several games (ls20 among them) encode the goal in HUD widgets outside the play area, so
    pixels alone cannot recover it. There we keep the REAL topology and synthesise a maximally
    separated goal pair, flagging it -- the embodiment experiments are about navigation cost on
    real maze structure, not about reproducing each game's symbol mechanic.
    """
    occ = np.asarray(comp["occ"])
    start = nearest_walkable(occ, comp["spawn"])
    # keep only the component the fly can actually reach, so every compiled maze is solvable
    comp_mask = reachable_from(occ, start).astype(int)
    goals = sorted(comp["goals"].items(), key=lambda kv: kv[1][2])      # rarest first
    if len(goals) >= 2:
        b1 = nearest_walkable(comp_mask, goals[0][1][:2])
        b2 = nearest_walkable(comp_mask, goals[1][1][:2])
        if b1 != b2:
            return dict(occ=comp_mask, start=start, button=b1, food=b2, goals_faithful=True)
    a, b = farthest_pair(comp_mask, start)
    return dict(occ=comp_mask, start=start, button=a, food=b, goals_faithful=False)

CUT = 0.20          # frame-to-frame change above this is a LEVEL REDRAW, not agent motion


def segments(stack, cut=CUT, min_len=30):
    """Split a frame stack at level transitions.

    Occupancy must be computed WITHIN a level. Pooling frames across a level change takes the
    union of different layouts, so a wall in level 1 that is floor in level 2 dissolves -- which
    is why heavier exploration made wall fractions go DOWN (ar25 8% -> 0%) instead of up.
    """
    if len(stack) < 2:
        return [(0, len(stack))]
    d = np.array([(stack[t] != stack[t - 1]).mean() for t in range(1, len(stack))])
    cuts = [0] + [t + 1 for t, v in enumerate(d) if v > cut] + [len(stack)]
    segs = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= min_len]
    return segs or [(0, len(stack))]


def compile_levels(stack, cut=CUT, min_len=30):
    """One compiled arena PER LEVEL, rather than one blurred arena per environment."""
    out = []
    for a, b in segments(stack, cut, min_len):
        c = compile_frames(stack[a:b])
        if c is not None:
            c["seg"] = (int(a), int(b))
            out.append(c)
    return out


def detour_stats(occ, spawn):
    """max/mean of shortest-path / straight-line over reachable cells.

    Taking the ratio only at the BFS-farthest cell is useless: in an open room that cell is a
    corner on a monotone path, so the ratio is 1.00 by construction for every environment.
    """
    from collections import deque
    occ = np.asarray(occ); H, W = occ.shape
    spawn = tuple(int(x) for x in spawn)
    if not occ[spawn]:
        return None
    d = -np.ones((H, W), int); d[spawn] = 0; q = deque([spawn])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and occ[rr, cc] and d[rr, cc] < 0:
                d[rr, cc] = d[r, c] + 1; q.append((rr, cc))
    rs, cs = np.nonzero(d > 0)
    if len(rs) == 0:
        return None                       # spawn is walled in: no reachable cell but itself
    manh = np.abs(rs - spawn[0]) + np.abs(cs - spawn[1])
    ratio = d[rs, cs] / np.maximum(manh, 1)
    return dict(reach=int((d >= 0).sum()), max_detour=float(ratio.max()),
                mean_detour=float(ratio.mean()), dist=d)
