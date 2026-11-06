"""One 3-D world per frozen task, built at a species' body scale and sensed by GPU ray casting.

Every species walks the SAME frozen layout (arc_fly.unified_env.Task); only lengths change. The scale rule
puts every species in corridors of the same width in its own body widths, so the geometry each species meets
is identical in body units. Senses are rays cast on GPU 1 with MuJoCo-Warp: photoreceptors, whiskers,
antennae and membrane contacts are all rays; odour is a wall-respecting field read at the sensor positions.

Design choices that are NOT measurements are listed in WORLD_CONTRACT and travel with every output.
"""
from dataclasses import dataclass
import os

import numpy as np

from arc_fly.unimaze import bfs

WORLD_CONTRACT = {
    "version": 1,
    "layout": "frozen Task.walkable grid; walls are an exact rectangle cover of non-walkable cells",
    "scale": "cell_m chosen so the maze passage width (narrowest gap on the widest spawn-goal route) equals CORRIDOR_BODY_WIDTHS body widths",
    "arena": "the passage width is capped at max(grid side) / 10, so every arena spans at least ten passages",
    "wall_height": "2 x the species' eye height (no species sees over a wall)",
    "goal_beacon": "cylinder at the goal cell, height 0.8 x wall height, radius 0.25 x corridor width; visible only in line of sight",
    "distal_cues": "the four perimeter walls carry four distinct colours (allocentric orientation cue)",
    "floor": "two-tone checker on grid cells (optic-flow texture)",
    "odour": "exp(-geodesic_distance / (4 corridor widths)) through walkable cells, read bilinearly at each sensor",
    "light": "fixed overhead-oblique directional light plus ambient; no shadows",
    "vision_groups": "geom groups 0 (walls, floor, beacon, panels) and 1 (visual-only occluders)",
    "touch_groups": "geom groups 0 and 2 (collision geometry), never the floor",
    "measured": False,
}
CORRIDOR_BODY_WIDTHS = 2.5
COLORS = dict(floor=(0.45, 0.45, 0.42), wall=(0.55, 0.60, 0.50), beacon=(1.0, 0.90, 0.20),
              north=(0.80, 0.25, 0.20), east=(0.20, 0.35, 0.85), south=(0.90, 0.80, 0.70),
              west=(0.25, 0.65, 0.30), sky=(0.55, 0.70, 0.90), gap=(0.04, 0.04, 0.04))
LIGHT = np.array([0.3, 0.2, 1.0]) / np.linalg.norm([0.3, 0.2, 1.0])
KIND = dict(floor=0, wall=1, beacon=2, panel=3, occluder=4, gap=5)


def passage_cells(task):
    """Passage width of the maze in cells: the largest clearance w such that spawn and goal stay connected
    through cells whose free width (2 x distance to the nearest wall, minus one) is at least w. It is the
    narrowest gap on the widest route, so a body sized to it can always pass; open arenas get large values."""
    from scipy.ndimage import distance_transform_edt
    width = 2 * distance_transform_edt(np.pad(task.walkable, 1, constant_values=False))[1:-1, 1:-1] - 1
    lo_ok = 1.0
    for w in sorted(set(np.round(width[task.walkable], 6))):
        if w <= lo_ok: continue
        ok = task.walkable & (width >= w)
        if not (ok[task.spawn] and ok[task.goal]) or bfs(ok, task.spawn)[task.goal] < 0:
            break
        lo_ok = w
    return float(max(lo_ok, 1.0))


ARENA_PASSAGES = 10      # an arena spans at least this many passage widths (barrier-free layouts would otherwise collapse)


def cell_size_for(task, body_width_m):
    passage = min(passage_cells(task), max(task.walkable.shape) / ARENA_PASSAGES)
    return CORRIDOR_BODY_WIDTHS * body_width_m / max(passage, 1.0)


def _rects(mask):
    """Exact rectangle cover of True cells: (r0, r1, c0, c1) half-open; row runs merged down identical columns."""
    H, W = mask.shape
    out, open_runs = [], {}
    for r in range(H + 1):
        runs = set()
        if r < H:
            c = 0
            while c < W:
                if mask[r, c]:
                    c0 = c
                    while c < W and mask[r, c]: c += 1
                    runs.add((c0, c))
                else:
                    c += 1
        for key in list(open_runs):
            if key not in runs:
                out.append((open_runs.pop(key), r) + key)
        for key in runs:
            open_runs.setdefault(key, r)
    return out


@dataclass
class World3D:
    task: object
    cell_m: float
    wall_h: float
    eye_h: float
    corridor_m: float
    model: object                    # mujoco.MjModel
    geom_kind: np.ndarray
    geom_rgb: np.ndarray
    odour: np.ndarray                # (H, W) field at grid cells, 0 in walls

    # grid <-> world: cell (r, c) centre sits at x = (c + 0.5) * cell, y = -(r + 0.5) * cell
    def xy_of_rc(self, rc):
        rc = np.asarray(rc, float)
        return np.stack([(rc[..., 1] + 0.5) * self.cell_m, -(rc[..., 0] + 0.5) * self.cell_m], -1)

    def rc_of_xy(self, xy):
        xy = np.asarray(xy, float)
        return np.stack([-xy[..., 1] / self.cell_m - 0.5, xy[..., 0] / self.cell_m - 0.5], -1)

    def odour_at(self, xy):
        """Bilinear read of the odour field at world positions (any leading shape)."""
        rc = self.rc_of_xy(xy)
        H, W = self.odour.shape
        r = np.clip(rc[..., 0], 0, H - 1); c = np.clip(rc[..., 1], 0, W - 1)
        r0 = np.floor(r).astype(int); c0 = np.floor(c).astype(int)
        r1 = np.minimum(r0 + 1, H - 1); c1 = np.minimum(c0 + 1, W - 1)
        fr, fc = r - r0, c - c0
        f = self.odour
        return ((1 - fr) * (1 - fc) * f[r0, c0] + (1 - fr) * fc * f[r0, c1]
                + fr * (1 - fc) * f[r1, c0] + fr * fc * f[r1, c1])


def build(task, body_width_m, eye_h_m, beacon=True, odour=True, holes=False, beacon_rc=None):
    """beacon/odour False reproduce hidden-goal experiments (e.g. Tartarus): no visible goal, no odour field.
    holes=True renders non-walkable cells as dark floor gaps that do not block sight (Tartarus removes floor tiles;
    de Cothi et al. 2022), instead of walls of twice eye height. beacon_rc moves the visible beacon off the goal (cue
    conflict); the odour field always comes from the goal."""
    import mujoco
    cell = cell_size_for(task, body_width_m)
    corridor = CORRIDOR_BODY_WIDTHS * body_width_m
    wall_h = 2.0 * eye_h_m
    H, W = task.walkable.shape
    geoms, kinds, rgbs = [], [], []

    def box(name, x, y, z, hx, hy, hz, rgb, kind, group=0, collide=True):
        con = 'contype="1" conaffinity="1"' if collide else 'contype="0" conaffinity="0"'
        geoms.append(f'<geom name="{name}" type="box" pos="{x:.6g} {y:.6g} {z:.6g}" size="{hx:.6g} {hy:.6g} {hz:.6g}" '
                     f'rgba="{rgb[0]} {rgb[1]} {rgb[2]} 1" group="{group}" {con}/>')
        kinds.append(KIND[kind]); rgbs.append(rgb)

    geoms.append(f'<geom name="floor" type="plane" pos="{W * cell / 2:.6g} {-H * cell / 2:.6g} 0" '
                 f'size="{W * cell:.6g} {H * cell:.6g} 1" rgba="{" ".join(map(str, COLORS["floor"]))} 1" group="0"/>')
    kinds.append(KIND["floor"]); rgbs.append(COLORS["floor"])
    solid = np.zeros_like(task.walkable) if holes else ~task.walkable
    occluder_only = (task.visual_occlusion & task.walkable) if not holes else np.zeros_like(task.walkable)
    invisible_solid = solid & ~task.visual_occlusion             # blocks movement, not sight
    if holes:                                                     # a gap: a dark plate flush with the floor
        for i, (r0, r1, c0, c1) in enumerate(_rects(~task.walkable)):
            box(f"h{i}", (c0 + c1) / 2 * cell, -(r0 + r1) / 2 * cell, 1e-3 * cell,
                (c1 - c0) / 2 * cell, (r1 - r0) / 2 * cell, 5e-4 * cell, COLORS["gap"], "gap", collide=False)
    for i, (r0, r1, c0, c1) in enumerate(_rects(solid & ~invisible_solid)):
        box(f"w{i}", (c0 + c1) / 2 * cell, -(r0 + r1) / 2 * cell, wall_h / 2,
            (c1 - c0) / 2 * cell, (r1 - r0) / 2 * cell, wall_h / 2, COLORS["wall"], "wall")
    for i, (r0, r1, c0, c1) in enumerate(_rects(occluder_only)):
        box(f"o{i}", (c0 + c1) / 2 * cell, -(r0 + r1) / 2 * cell, wall_h / 2,
            (c1 - c0) / 2 * cell, (r1 - r0) / 2 * cell, wall_h / 2, COLORS["wall"], "occluder", group=1, collide=False)
    for i, (r0, r1, c0, c1) in enumerate(_rects(invisible_solid)):
        box(f"g{i}", (c0 + c1) / 2 * cell, -(r0 + r1) / 2 * cell, wall_h / 2,
            (c1 - c0) / 2 * cell, (r1 - r0) / 2 * cell, wall_h / 2, COLORS["wall"], "wall", group=2)
    t = cell / 2                                                     # perimeter panels, half a cell thick
    box("north", W * cell / 2, t, wall_h / 2, W * cell / 2 + 2 * t, t, wall_h / 2, COLORS["north"], "panel")
    box("south", W * cell / 2, -H * cell - t, wall_h / 2, W * cell / 2 + 2 * t, t, wall_h / 2, COLORS["south"], "panel")
    box("west", -t, -H * cell / 2, wall_h / 2, t, H * cell / 2, wall_h / 2, COLORS["west"], "panel")
    box("east", W * cell + t, -H * cell / 2, wall_h / 2, t, H * cell / 2, wall_h / 2, COLORS["east"], "panel")
    br_, bc_ = task.goal if beacon_rc is None else beacon_rc
    gx, gy = (bc_ + 0.5) * cell, -(br_ + 0.5) * cell
    bh, br = 0.8 * wall_h, 0.25 * corridor
    if beacon:
        geoms.append(f'<geom name="beacon" type="cylinder" pos="{gx:.6g} {gy:.6g} {bh / 2:.6g}" size="{br:.6g} {bh / 2:.6g}" '
                     f'rgba="{" ".join(map(str, COLORS["beacon"]))} 1" group="0" contype="0" conaffinity="0"/>')
        kinds.append(KIND["beacon"]); rgbs.append(COLORS["beacon"])
    xml = ('<mujoco model="world3d"><option timestep="0.001"/><worldbody>'
           '<light directional="true" dir="-0.3 -0.2 -1"/>' + "".join(geoms) + '</worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    d = bfs(task.walkable, task.goal).astype(float)
    lam_cells = 4 * corridor / cell
    odour = np.where(d >= 0, np.exp(-d / lam_cells), 0.0) if odour else np.zeros_like(d)
    return World3D(task, cell, wall_h, eye_h_m, corridor, model, np.array(kinds), np.array(rgbs, float), odour)


class RayCaster:
    """Batched ray casting on the GPU visible to this process (run with CUDA_VISIBLE_DEVICES=1)."""

    def __init__(self, world):
        import mujoco, mujoco_warp as mjw, warp as wp
        self.mjw, self.wp, self.world = mjw, wp, world
        data = mujoco.MjData(world.model); mujoco.mj_forward(world.model, data)
        self.m = mjw.put_model(world.model)
        self.d = mjw.put_data(world.model, data, nworld=1)

    def cast(self, origins, dirs, groups=(1, 1, 0, 0, 0, 0)):
        """origins, dirs: (N, 3). Returns dist (N,) [-1 = miss], geom id (N,), normal (N, 3)."""
        wp, mjw = self.wp, self.mjw
        o = np.ascontiguousarray(origins, np.float32).reshape(1, -1, 3)
        v = np.asarray(dirs, np.float64)
        v = np.ascontiguousarray(v / np.linalg.norm(v, axis=-1, keepdims=True), np.float32).reshape(1, -1, 3)
        n = o.shape[1]
        P, V = wp.array(o, dtype=wp.vec3f), wp.array(v, dtype=wp.vec3f)
        dist = wp.zeros((1, n), dtype=wp.float32); gid = wp.zeros((1, n), dtype=wp.int32)
        nrm = wp.zeros((1, n), dtype=wp.vec3f)
        flags = wp.types.vector(6, wp.float32)(*[float(g) for g in groups])   # geom groups 0-5 included in the cast
        mjw.rays(self.m, self.d, P, V, flags, True, wp.full(n, -1, dtype=wp.int32), dist, gid, nrm)
        return dist.numpy()[0], gid.numpy()[0], nrm.numpy()[0]

    def shade(self, origins, dirs, dist, gid, nrm):
        """Linear RGB radiance per ray: geom colour x (ambient + Lambert), floor checker, sky for misses."""
        w = self.world
        v = np.asarray(dirs, float); v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        hit = (dist >= 0) & (gid >= 0)
        rgb = np.tile(np.array(COLORS["sky"]), (len(dist), 1))
        g = np.where(hit, gid, 0)
        n = np.asarray(nrm, float)
        n = np.where((np.sum(n * v, -1, keepdims=True) > 0), -n, n)   # normal faces the viewer
        lam = 0.45 + 0.55 * np.clip(n @ LIGHT, 0, 1)
        base = w.geom_rgb[g] * lam[:, None]
        floor = hit & (w.geom_kind[g] == KIND["floor"])
        if floor.any():
            p = np.asarray(origins, float)[floor] + dist[floor, None] * v[floor]
            parity = (np.floor(p[:, 0] / w.cell_m) + np.floor(-p[:, 1] / w.cell_m)) % 2
            base[floor] *= np.where(parity > 0, 0.82, 1.0)[:, None]
        rgb[hit] = base[hit]
        return rgb


def _selfcheck():
    """Geometry: downward rays at every cell centre must hit a wall exactly where the grid is not walkable
    (and the floor where it is), on GPU; the odour field must be 1 at the goal and fall along the path."""
    from arc_fly.unified_env import load_tasks
    tasks = load_tasks("data/unified/tasks.json")
    for task in (next(t for t in tasks if t.id.startswith("mcmaze")), next(t for t in tasks if t.id.startswith("arc"))):
        w = build(task, body_width_m=1e-3, eye_h_m=0.8e-3)
        rc = np.argwhere(np.ones(task.walkable.shape, bool))
        xy = w.xy_of_rc(rc)
        o = np.c_[xy, np.full(len(xy), 3 * w.wall_h)]
        caster = RayCaster(w)
        dist, gid, _ = caster.cast(o, np.tile([0, 0, -1.0], (len(o), 1)), groups=(1, 0, 1, 0, 0, 0))
        kind = w.geom_kind[gid]
        hit_wall = (kind == KIND["wall"]).reshape(task.walkable.shape)
        beacon = (kind == KIND["beacon"]).reshape(task.walkable.shape)
        mismatch = int(((hit_wall != ~task.walkable) & ~beacon).sum())
        assert mismatch == 0, f"{task.id}: {mismatch} cells disagree with the frozen grid"
        assert beacon[task.goal], "the beacon must stand on the goal cell"
        assert np.isclose(w.odour[task.goal], 1.0) and w.odour[task.spawn] < 1.0
        assert np.isclose(w.odour_at(w.xy_of_rc(np.array(task.goal))), 1.0)
        print(f"{task.id}: grid {task.walkable.shape}, cell {w.cell_m * 1e3:.3f} mm, corridor {w.corridor_m * 1e3:.2f} mm, "
              f"{w.model.ngeom} geoms, {len(o)} downward rays agree with the grid; odour at spawn {w.odour[task.spawn]:.3g}")
    task = load_tasks("data/unified/tartarus/tasks.json")[0]                         # floor-gap maze (holes=True)
    w = build(task, body_width_m=0.06, eye_h_m=0.04, beacon=False, odour=False, holes=True); caster = RayCaster(w)
    rc = np.argwhere(np.ones(task.walkable.shape, bool)); xy = w.xy_of_rc(rc)
    _, gid, _ = caster.cast(np.c_[xy, np.full(len(xy), 3 * w.wall_h)], np.tile([0, 0, -1.0], (len(xy), 1)))
    gap = (w.geom_kind[gid] == KIND["gap"]).reshape(task.walkable.shape)
    assert (gap == ~task.walkable).all(), "downward rays must hit a gap exactly on the removed tiles"
    th = np.linspace(-np.pi, np.pi, 72, endpoint=False); o = np.tile([*w.xy_of_rc(np.array(task.spawn)), w.eye_h], (72, 1))
    _, gid, _ = caster.cast(o, np.c_[np.cos(th), np.sin(th), np.zeros(72)])
    assert not np.isin(w.geom_kind[gid], [KIND["wall"], KIND["occluder"]]).any(), "gaps must not block sight"
    print(f"{task.id}: {(~task.walkable).sum()} gaps seen from above; eye-level rays reach the distal panels")
    print("world3d self-check OK")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    _selfcheck()
