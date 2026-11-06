"""Unified mazes: one canonical grid for every rung of the ladder.

An ARC-AGI-3 level, one of the monkey's 108 MC_Maze barrier layouts, a procedural maze, or a
microfluidic channel raster all become the same object: a boolean occupancy grid (True = free)
with a physical cell size, a spawn, a goal, and a record of where it came from. Exporters turn it
back into whatever a substrate's simulator wants. cell_mm is body-relative on purpose: a fly's
6 mm corridor and a cell's 40 um channel are the same maze at the same dimensionless width.
"""
from collections import deque
from dataclasses import dataclass, field
import json, os
import numpy as np

NB = ((1, 0), (-1, 0), (0, 1), (0, -1))
CAPSTONE = os.environ.get("CAPSTONE", "")
MCMAZE_NPZ = f"{CAPSTONE}/notebooks/save/mc_maze_configs.npz"


def bfs(occ, start):
    """Cell distances from start through free cells; -1 = unreachable."""
    H, W = occ.shape
    d = np.full(occ.shape, -1, np.int32)
    if not occ[start]:
        return d
    d[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in NB:
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and occ[n] and d[n] < 0:
                d[n] = d[r, c] + 1
                q.append(n)
    return d


@dataclass(frozen=True)
class Maze:
    occ: np.ndarray                  # bool HxW, True = free
    cell_mm: float
    spawn: tuple
    goal: tuple
    source: dict = field(default_factory=dict)
    decoy: tuple = None

    # ---- the one accounting --------------------------------------------------------------
    @property
    def dist(self):
        return bfs(self.occ, self.spawn)

    @property
    def opt(self):
        return int(self.dist[self.goal])

    @property
    def detour(self):
        """max over reachable cells of shortest-path / L1 distance; 1.0 = never forced to turn."""
        d = self.dist
        rr, cc = np.nonzero(d > 0)
        l1 = np.abs(rr - self.spawn[0]) + np.abs(cc - self.spawn[1])
        return float(np.max(d[rr, cc] / np.maximum(l1, 1))) if len(rr) else 1.0

    def spl(self, path_cells, success):
        """Success-weighted path length, the cross-rung score (Anderson et al. 2018)."""
        return float(success) * self.opt / max(self.opt, float(path_cells), 1e-9)

    # ---- exports -------------------------------------------------------------------------
    def ascii(self, agent=None):
        rows = []
        for r in range(self.occ.shape[0]):
            row = ""
            for c in range(self.occ.shape[1]):
                p = (r, c)
                row += ("A" if p == agent else "S" if p == self.spawn else "G" if p == self.goal
                        else "D" if p == self.decoy else "." if self.occ[p] else "#")
            rows.append(row)
        return "\n".join(rows)

    def to_png(self, path):
        from PIL import Image
        Image.fromarray(np.where(self.occ, 255, 0).astype(np.uint8)).save(path)
        return path

    def to_graph(self):
        """4-neighbour graph over free cells: (nodes, edges, laplacian). The Tero/Physarum null
        and any flow model live on this."""
        nodes = [tuple(x) for x in np.argwhere(self.occ)]
        ix = {n: i for i, n in enumerate(nodes)}
        edges = [(ix[n], ix[(n[0] + dr, n[1] + dc)]) for n in nodes for dr, dc in NB[::2]
                 if (n[0] + dr, n[1] + dc) in ix]
        L = np.zeros((len(nodes), len(nodes)), np.float32)
        for i, j in edges:
            L[i, i] += 1; L[j, j] += 1; L[i, j] -= 1; L[j, i] -= 1
        return nodes, edges, L

    def to_rects(self, exact=True):
        """Wall rectangles (cx, cy, half_w, half_h) in mm. MC_Maze's own layout if we have it;
        otherwise an EXACT cover of the wall cells (row runs merged down identical columns), so
        a rectangle never covers a free cell; exact=False gives the lossy component bounding
        boxes. Coordinates: x = column, y = row, origin at the grid's top-left corner."""
        if "barriers_m" in self.source:
            return np.asarray(self.source["barriers_m"]) * 1000.0
        if exact:
            wall = ~self.occ
            H, W = wall.shape
            runs = {}                                   # (c0, c1) -> [r_start, r_end]
            out = []
            for r in range(H + 1):
                row = set()
                if r < H:
                    c = 0
                    while c < W:
                        if wall[r, c]:
                            c0 = c
                            while c < W and wall[r, c]: c += 1
                            row.add((c0, c))
                        else:
                            c += 1
                for k in list(runs):
                    if k not in row:
                        r0, r1 = runs.pop(k); c0, c1 = k
                        out.append(((c0 + c1) / 2 * self.cell_mm, (r0 + r1) / 2 * self.cell_mm,
                                    (c1 - c0) / 2 * self.cell_mm, (r1 - r0) / 2 * self.cell_mm))
                for k in row:
                    if k in runs: runs[k][1] = r + 1
                    else: runs[k] = [r, r + 1]
            return np.array(out, np.float32).reshape(-1, 4)
        from scipy import ndimage
        lab, n = ndimage.label(~self.occ)
        out = []
        for k in range(1, n + 1):
            rr, cc = np.nonzero(lab == k)
            cy, cx = (rr.min() + rr.max() + 1) / 2, (cc.min() + cc.max() + 1) / 2
            out.append((cx * self.cell_mm, cy * self.cell_mm,
                        (cc.max() - cc.min() + 1) / 2 * self.cell_mm,
                        (rr.max() - rr.min() + 1) / 2 * self.cell_mm))
        return np.array(out, np.float32).reshape(-1, 4)


# ---- imports -------------------------------------------------------------------------------
def _clear(occ, p, radius=1):
    """Open a small disc around p so a spawn or goal is never inside a wall."""
    occ = occ.copy()
    H, W = occ.shape
    for r in range(max(0, p[0] - radius), min(H, p[0] + radius + 1)):
        for c in range(max(0, p[1] - radius), min(W, p[1] + radius + 1)):
            occ[r, c] = True
    return occ


def from_arc(key, cell_mm=6.0, corpus="data/arc_corpus/arc_all_levels.npz"):
    """An ARC-AGI-3 level, compiled exactly as make_env does it: farthest reachable cell is the
    goal, the median-distance cell the decoy."""
    from arc_fly.arc_compile import compile_frames
    comp = compile_frames(np.load(corpus)[key])
    if comp is None:
        raise ValueError(f"{key}: not compilable")
    occ = np.asarray(comp["occ"], bool)
    spawn = tuple(int(x) for x in comp["spawn"])
    occ = _clear(occ, spawn, 0)
    d = bfs(occ, spawn)
    goal = tuple(int(x) for x in np.unravel_index(int(np.argmax(np.where(d >= 0, d, -1))), d.shape))
    reach = [tuple(int(y) for y in x) for x in np.argwhere(d > 0)]
    decoy = reach[len(reach) // 2] if reach else goal
    return Maze(occ, cell_mm, spawn, goal, dict(substrate="fly", dataset="ARC-AGI-3", id=key,
                                                 goals_faithful=False), decoy)


def from_mcmaze(i, cell_mm=2.0, npz=MCMAZE_NPZ, margin_m=0.02, spawn_m=(0.0, 0.0)):
    """MC_Maze configuration i (0..107): barriers (cx, cy, half_w, half_h) in metres of the
    monkey's hand workspace, rasterised at cell_mm. The monkey starts at the centre and the
    cued target is the goal; the real barriers were virtual (fail-on-contact), which for a maze
    is the same as impassable."""
    z = np.load(npz)
    bars = z["barriers"][i][z["mask"][i].astype(bool)]
    tgt = z["targets"][i]
    lo = np.array([-0.20, -0.16]) - margin_m
    hi = np.array([0.20, 0.16]) + margin_m
    m = cell_mm / 1000.0
    W, H = int(np.ceil((hi[0] - lo[0]) / m)), int(np.ceil((hi[1] - lo[1]) / m))
    occ = np.ones((H, W), bool)
    ys = lo[1] + (np.arange(H) + 0.5) * m
    xs = lo[0] + (np.arange(W) + 0.5) * m
    for cx, cy, hw, hh in bars:
        occ[np.ix_(np.abs(ys - cy) <= hh, np.abs(xs - cx) <= hw)] = False
    cell = lambda xy: (int((xy[1] - lo[1]) / m), int((xy[0] - lo[0]) / m))
    spawn, goal = cell(spawn_m), cell(tgt)
    occ = _clear(_clear(occ, spawn), goal)
    return Maze(occ, cell_mm, spawn, goal, dict(substrate="monkey", dataset="MC_Maze DANDI 000128",
                                                 id=int(i), key=[int(x) for x in z["keys"][i]],
                                                 barriers_m=bars.tolist(), target_m=tgt.tolist(),
                                                 origin_m=lo.tolist()))


def from_png(path, cell_mm, spawn, goal, substrate="cell", thresh=128):
    """A raster maze (e.g. a microfluidic channel layout): bright = free."""
    from PIL import Image
    occ = np.asarray(Image.open(path).convert("L")) >= thresh
    return Maze(occ, cell_mm, tuple(spawn), tuple(goal), dict(substrate=substrate, dataset=path))


def procedural(seed, H=21, W=21, cell_mm=6.0, dead_ends=True):
    """Recursive-backtracker maze on an odd grid; spawn top-left, goal the BFS-farthest cell."""
    rng = np.random.default_rng(seed)
    occ = np.zeros((H, W), bool)
    stack = [(1, 1)]; occ[1, 1] = True
    while stack:
        r, c = stack[-1]
        opts = [(r + 2 * dr, c + 2 * dc, r + dr, c + dc) for dr, dc in NB
                if 0 < r + 2 * dr < H - 1 and 0 < c + 2 * dc < W - 1 and not occ[r + 2 * dr, c + 2 * dc]]
        if opts:
            nr, nc, wr, wc = opts[int(rng.integers(len(opts)))]
            occ[wr, wc] = occ[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()
    if not dead_ends:                                   # knock out a few walls -> loops, no dead ends
        for _ in range(H * W // 20):
            r, c = int(rng.integers(1, H - 1)), int(rng.integers(1, W - 1))
            occ[r, c] = True
    d = bfs(occ, (1, 1))
    goal = tuple(int(x) for x in np.unravel_index(int(np.argmax(d)), d.shape))
    return Maze(occ, cell_mm, (1, 1), goal, dict(substrate="procedural", dataset="backtracker", id=int(seed)))


def _selfcheck():
    empty = from_mcmaze(0)
    assert empty.source["barriers_m"] == [] and abs(empty.detour - 1.0) < 1e-9, "config 0 has no barriers"
    walled = next(from_mcmaze(i) for i in range(108) if len(from_mcmaze(i).source["barriers_m"]) >= 6)
    assert walled.opt > 0 and walled.detour > 1.0, f"barriers must force a detour, got {walled.detour}"
    assert walled.to_rects().shape[1] == 4
    # exact cover: rasterising the rectangles back must reproduce the wall mask exactly
    R = p_ = procedural(5)
    rects = R.to_rects(exact=True); back = np.ones(R.occ.shape, bool)
    for cx, cy, hw, hh in rects / R.cell_mm:
        back[int(round(cy - hh)):int(round(cy + hh)), int(round(cx - hw)):int(round(cx + hw))] = False
    assert np.array_equal(back, R.occ), "exact rectangle cover must reproduce the occupancy"
    p = procedural(3)
    assert p.opt > 10 and p.detour > 1.5
    nodes, edges, L = p.to_graph()
    assert np.allclose(L.sum(1), 0) and len(nodes) == p.occ.sum()
    png = "/tmp/unimaze_check.png"; p.to_png(png)
    back = from_png(png, p.cell_mm, p.spawn, p.goal)
    assert np.array_equal(back.occ, p.occ), "png round trip"
    assert abs(p.spl(p.opt, True) - 1.0) < 1e-9 and p.spl(3 * p.opt, True) - 1 / 3 < 1e-9 and p.spl(1, False) == 0
    if os.path.exists("data/arc_corpus/arc_all_levels.npz"):
        m = from_arc("bp35#7")
        meta = {x["level"]: x for x in json.load(open("data/arc_corpus/arc_all_levels.json"))}
        assert abs(m.detour - meta["bp35#7"]["max_detour"]) < 0.05, (m.detour, meta["bp35#7"]["max_detour"])
    print(f"unimaze ok: MC_Maze {walled.source['id']} {walled.occ.shape} opt={walled.opt} d={walled.detour:.2f} "
          f"rects={len(walled.to_rects())}; procedural opt={p.opt} d={p.detour:.2f}; "
          f"arc bp35#7 d={m.detour:.2f}" if os.path.exists("data/arc_corpus/arc_all_levels.npz") else "unimaze ok")


if __name__ == "__main__":
    _selfcheck()


# ---- mouse rung: the FreelyMovingVR4Mice occlusion task as an occlusion maze --------------
MOUSE_VIS = {0: 0.99, 1: 0.72, 2: 0.35, 3: 0.10, 4: 0.0}   # Benquet et al. 2026: five occlusion levels


def mouse_vr(level=0, target_side="left", cell_cm=0.5, arena_cm=52.0, object_w_cm=6.0):
    """Benquet et al. 2026 (Current Biology): a 52 x 52 cm arena; the mouse starts in an initiation
    zone 21.25 cm from the front screen, two objects sit left/right in front of the screen behind
    occluder walls whose aperture sets how much of the object is visible from the start (99, 72,
    35, 10, 0 %), and reward is a lick port 5 cm behind the screen on the target's side. Here the
    occluders are wall cells with an aperture whose width is the visible fraction of the object
    seen from the start cell; the goal is the target-side port and the decoy the other port. The
    latent variable is which side the target is on, resolvable only by moving to a viewpoint."""
    n = int(round(arena_cm / cell_cm))
    occ = np.ones((n, n), bool)
    front = 1                                              # row of the screen (top); ports behind it
    occl_row = int(round(12.0 / cell_cm))                  # occluder wall row, in front of the objects
    vis = MOUSE_VIS[level]
    obj_w = int(round(object_w_cm / cell_cm))
    start = (int(round(21.25 / cell_cm)) + occl_row, n // 2)
    occ[occl_row, :] = False                                # a full occluder wall ...
    for c0 in (n // 4, 3 * n // 4):                        # ... opened where the START's sightline to each
        obj_r = front + 3                                   # object cell crosses it, for round(vis*width) cells
        cols = sorted(range(c0 - obj_w // 2, c0 + obj_w - obj_w // 2), key=lambda c: abs(c - c0))
        for c in cols[:int(round(vis * obj_w))]:
            t = (occl_row - start[0]) / (obj_r - start[0])
            occ[occl_row, int(round(start[1] + t * (c - start[1])))] = True
    occ[occl_row, n // 2 - 1:n // 2 + 1] = True             # central gap: the mouse can always go and look
    start = (int(round(21.25 / cell_cm)) + occl_row, n // 2)
    port = {"left": (front, n // 4), "right": (front, 3 * n // 4)}
    goal, decoy = port[target_side], port["right" if target_side == "left" else "left"]
    occ = _clear(_clear(occ, start), goal); occ = _clear(occ, decoy)
    return Maze(occ, cell_cm * 10.0, start, goal, dict(substrate="mouse", dataset="FreelyMovingVR4Mice",
                                                       id=f"occl{level}-{target_side}", visible=vis,
                                                       zenodo="10.5281/zenodo.19091270"), decoy)


def visible_fraction(maze, cell, target, width=6):
    """Fraction of a `width`-cell object centred at `target` that has an unblocked straight line to
    `cell` (Bresenham through free cells): the mouse rung's sensory channel."""
    H, W = maze.occ.shape
    seen = 0
    for dc in range(-(width // 2), width - width // 2):
        t = (target[0], target[1] + dc)
        if not (0 <= t[1] < W): continue
        r0, c0 = cell; r1, c1 = t
        n = max(abs(r1 - r0), abs(c1 - c0), 1); ok = True
        for k in range(1, n):
            r, c = round(r0 + (r1 - r0) * k / n), round(c0 + (c1 - c0) * k / n)
            if not maze.occ[r, c]: ok = False; break
        seen += ok
    return seen / width
