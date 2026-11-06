"""One frozen navigation task and sensor/scoring contract for every body adapter.

Native experimental observations live alongside these derived navigation tasks.
Rescaling a task changes lengths, never its grid, endpoints, split or information.
This module does not claim that the common sensors reproduce animal physiology.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from arc_fly.unimaze import Maze, NB, bfs


CONTRACT = {
    "version": 1,
    "actions": ["row+1", "row-1", "column+1", "column-1", "wait"],
    "observation": ["cue_row+1", "cue_row-1", "cue_column+1", "cue_column-1", "previous_contact"],
    "cue": "exp(-geodesic_distance_cells/10); walls and unreachable cells = 0",
    "cue_noise_sd": 0.05,
    "cue_noise": "independent Gaussian per decision and channel; seeded per episode",
    "goal_event": "first entry to the goal grid cell",
    "collision": "blocked move consumes a decision and sets previous_contact",
    "budget": "6 * optimal_path_cells + 50 decisions",
    "score": "success * optimal_path_cells / max(optimal_path_cells, travelled_cells)",
    "zero_length_success_spl": 1.0,
    "privileged_information": "geodesic cue is constructed using the full layout; no pose, map or goal coordinates in observation",
    "native_equivalence": False,
}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Task:
    id: str
    walkable: np.ndarray
    visual_occlusion: np.ndarray
    spawn: tuple
    goal: tuple
    native_cell_m: float
    source: dict
    split: str

    def __post_init__(self):
        for field in ("walkable", "visual_occlusion"):
            value = np.asarray(getattr(self, field))
            if value.ndim != 2 or value.dtype != np.bool_:
                raise ValueError(f"{field} must be a boolean 2D grid")
            value = value.copy()
            value.flags.writeable = False
            object.__setattr__(self, field, value)
        if self.walkable.shape != self.visual_occlusion.shape:
            raise ValueError("Collision and visual layers must share a frame")
        for name in ("spawn", "goal"):
            cell = tuple(getattr(self, name))
            if (len(cell) != 2 or any(not isinstance(v, (int, np.integer)) for v in cell)
                    or any(v < 0 or v >= n for v, n in zip(cell, self.walkable.shape))
                    or not self.walkable[cell]):
                raise ValueError(f"{name} is outside walkable geometry; endpoints are never silently cleared")
            object.__setattr__(self, name, tuple(map(int, cell)))
        if not np.isfinite(self.native_cell_m) or self.native_cell_m <= 0:
            raise ValueError("Native cell size must be positive metres")
        if self.split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        if bfs(self.walkable, self.spawn)[self.goal] < 0:
            raise ValueError("Unreachable goal")

    @property
    def geometry_sha256(self):
        return digest_json(dict(shape=self.walkable.shape,
            walkable=np.packbits(self.walkable).tobytes().hex(),
            visual_occlusion=np.packbits(self.visual_occlusion).tobytes().hex(),
            spawn=self.spawn, goal=self.goal))

    @property
    def task_sha256(self):
        return digest_json(dict(id=self.id, geometry=self.geometry_sha256,
            native_cell_m=self.native_cell_m, source=self.source, split=self.split,
            contract=CONTRACT))

    @property
    def opt(self):
        return int(bfs(self.walkable, self.spawn)[self.goal])

    def maze(self, cell_mm=None):
        """Body scaling without re-rasterizing or exposing native rectangle shortcuts."""
        cell_mm = self.native_cell_m * 1000 if cell_mm is None else float(cell_mm)
        if not np.isfinite(cell_mm) or cell_mm <= 0:
            raise ValueError("Body cell size must be positive")
        return Maze(self.walkable, cell_mm, self.spawn, self.goal,
            dict(id=self.id, dataset=self.source["dataset"], task_sha256=self.task_sha256,
                 geometry_sha256=self.geometry_sha256, split=self.split,
                 source_record=self.source, adaptation="uniform scaling of frozen common grid"))

    def fly_level(self, cell_mm=6.0):
        m = self.maze(cell_mm)
        return dict(name=self.id, occ=m.occ, spawn=m.spawn, goal=m.goal, decoy=m.goal,
                    opt=m.opt, detour=m.detour, stratum="unified", cell_mm=m.cell_mm,
                    task_sha256=self.task_sha256, geometry_sha256=self.geometry_sha256,
                    note="Geometry adapter only; legacy fly sensing/scoring does not implement CONTRACT")


def cursor_config(tasks, cell_mm=None):
    """Exact common-grid rectangles, starts and goals in one centred metre frame."""
    if not tasks:
        raise ValueError("No tasks")
    rows = []
    for task in tasks:
        maze = task.maze(cell_mm)
        cell = maze.cell_mm / 1000
        height, width = maze.occ.shape
        origin = -np.array([width, height]) * cell / 2
        rects = maze.to_rects(exact=True).astype(float) / 1000
        rects[:, :2] += origin
        xy = lambda rc: origin + (np.array(rc[::-1]) + 0.5) * cell
        rows.append((rects, xy(maze.spawn), xy(maze.goal), origin, cell))
    max_rects = max(len(row[0]) for row in rows)
    h = max(t.walkable.shape[0] for t in tasks)
    w = max(t.walkable.shape[1] for t in tasks)
    bars = np.zeros((len(tasks), max_rects, 4))
    mask = np.zeros((len(tasks), max_rects), dtype=bool)
    occupancy = np.zeros((len(tasks), h, w), dtype=bool)
    visual = np.zeros_like(occupancy)
    for i, (task, row) in enumerate(zip(tasks, rows)):
        bars[i, :len(row[0])] = row[0]
        mask[i, :len(row[0])] = True
        hh, ww = task.walkable.shape
        occupancy[i, :hh, :ww] = task.walkable
        visual[i, :hh, :ww] = task.visual_occlusion
    return dict(ids=np.array([t.id for t in tasks]), barriers=bars, mask=mask,
        starts=np.array([r[1] for r in rows]), targets=np.array([r[2] for r in rows]),
        n_barriers=mask.sum(1), occ=occupancy, visual_occlusion=visual,
        occ_origin_m=np.array([r[3] for r in rows]), occ_cell_m=np.array([r[4] for r in rows]),
        occ_shape=np.array([t.walkable.shape for t in tasks]),
        task_sha256=np.array([t.task_sha256 for t in tasks]),
        geometry_sha256=np.array([t.geometry_sha256 for t in tasks]),
        split=np.array([t.split for t in tasks]), contract_json=np.array(json.dumps(CONTRACT)))


def save_tasks(tasks, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    arrays, rows = {}, []
    for i, task in enumerate(tasks):
        arrays[f"walkable_{i}"] = task.walkable
        arrays[f"visual_{i}"] = task.visual_occlusion
        rows.append(dict(id=task.id, spawn=task.spawn, goal=task.goal,
            native_cell_m=task.native_cell_m, source=task.source, split=task.split,
            geometry_sha256=task.geometry_sha256, task_sha256=task.task_sha256,
            optimal_path_cells=task.opt))
    np.savez_compressed(directory / "tasks.npz", **arrays)
    payload = dict(schema_version=1, contract=CONTRACT, tasks=rows,
                   arrays_sha256=file_hash(directory / "tasks.npz"))
    payload["manifest_sha256"] = digest_json(payload)
    (directory / "tasks.json").write_text(json.dumps(payload, indent=2) + "\n")


def load_tasks(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    signature = payload.pop("manifest_sha256")
    if digest_json(payload) != signature or payload["contract"] != CONTRACT:
        raise ValueError("Task manifest or contract checksum mismatch")
    array_path = path.with_suffix(".npz")
    if file_hash(array_path) != payload["arrays_sha256"]:
        raise ValueError("Task array checksum mismatch")
    tasks = []
    with np.load(array_path, allow_pickle=False) as arrays:
        for i, row in enumerate(payload["tasks"]):
            task = Task(row["id"], arrays[f"walkable_{i}"], arrays[f"visual_{i}"],
                        tuple(row["spawn"]), tuple(row["goal"]), row["native_cell_m"],
                        row["source"], row["split"])
            if task.task_sha256 != row["task_sha256"]:
                raise ValueError("Task checksum mismatch")
            tasks.append(task)
    return tasks


class GridEnvironment:
    """Executable common condition; records actual contact and first arrival."""
    def __init__(self, task, seed=0):
        self.task = task
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.distance = bfs(task.walkable, task.goal)
        self.position = task.spawn
        self.path = [task.spawn]
        self.actions = []
        self.contact = False
        self.first_arrival = 0 if task.spawn == task.goal else None
        self.budget = 6 * task.opt + 50
        self._observation = self._sense()

    def _sense(self):
        cue = np.zeros(4)
        for index, offset in enumerate(NB):
            p = tuple(a+b for a, b in zip(self.position, offset))
            if all(0 <= v < n for v, n in zip(p, self.distance.shape)) and self.distance[p] >= 0:
                cue[index] = np.exp(-float(self.distance[p]) / 10)
        cue += self.rng.normal(0, CONTRACT["cue_noise_sd"], 4)
        return np.r_[cue, float(self.contact)]

    def observe(self):
        # Reading the same decision twice must not grant extra independent samples.
        return self._observation.copy()

    @property
    def done(self):
        return self.first_arrival is not None or len(self.actions) >= self.budget

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode already terminated")
        if type(action) not in (int, np.int64, np.int32) or action not in range(5):
            raise ValueError("Action must be an integer from 0 through 4")
        self.contact = False
        if action < 4:
            p = tuple(a+b for a, b in zip(self.position, NB[action]))
            valid = all(0 <= v < n for v, n in zip(p, self.task.walkable.shape))
            if valid and self.task.walkable[p]:
                self.position = p
            else:
                self.contact = True
        self.actions.append(int(action))
        self.path.append(self.position)
        if self.position == self.task.goal:
            self.first_arrival = len(self.actions)
        self._observation = self._sense()
        return self.observe(), self.done

    def result(self):
        success = self.first_arrival is not None
        length = sum(a != b for a, b in zip(self.path, self.path[1:]))
        spl = float(success) * (self.task.opt / max(self.task.opt, length, 1)
                               if self.task.opt else 1)
        return dict(task_id=self.task.id, task_sha256=self.task.task_sha256,
                    seed=self.seed, complete=self.done, success=success,
                    first_arrival_decision=self.first_arrival, decisions=len(self.actions),
                    path_cells=length, spl=spl, path=self.path, actions=self.actions,
                    time_unit="decision; not physical seconds")
