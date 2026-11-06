"""Privilege-free closed-loop contract for the Tartarus maze (de Cothi et al. 2022), SOURCE-ONLY prototype.

arc_fly.unified_env.GridEnvironment senses exp(-geodesic/10), a monotone transform of the optimal value computed
from the full layout: greedy ascent solves every maze, so success there is not evidence of memory or planning.
This contract keeps the same frozen tasks, actions, contact rule, budget and scoring, but the observation carries
only what a navigating animal can plausibly sense locally:
    [row/9, col/9,                      own position (distal room cues; an ASSUMPTION, declared)
     wall_row+1, wall_row-1, wall_col+1, wall_col-1,   adjacent cell blocked or outside (local touch/sight)
     previous_contact,                  the last move was blocked
     reward]                            1 on the decision that first enters the hidden goal, else 0
No goal coordinates, no distance, no map. The goal is fixed across all 250 tasks (inferred in the import), so
finding it again is a memory problem and every configuration change is a contingency change.
UNVERIFIED (root checks the primary protocol): budget/timeout rule of the animals, within-configuration start
order, and the 0.2 m cell size. No learner is fitted here.
  python scripts/tartarus_closedloop_env.py check
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc_fly.unified_env import GridEnvironment, Task, load_tasks
from arc_fly.unimaze import NB, bfs

TASKS = "data/unified/tartarus/tasks.json"
TRAJ = "data/unified/tartarus_trajectories.json"


class LocalGridEnvironment(GridEnvironment):
    """Same dynamics, budget and result() as GridEnvironment; only the sensor differs."""

    def _sense(self):
        r, c = self.position; H, W = self.task.walkable.shape
        walls = [float(not (0 <= r + dr < H and 0 <= c + dc < W and self.task.walkable[r + dr, c + dc])) for dr, dc in NB]
        reward = float(self.first_arrival is not None and self.first_arrival == len(self.actions) and len(self.actions) > 0)
        return np.array([r / (H - 1), c / (W - 1), *walls, float(self.contact), reward])


def animal_orders(path=TRAJ):
    """Per species/subject: the (configuration, trial, task_id) sequence in FILE order (no outcomes used). The file
    order is the importer's (subject, configuration, trial) enumeration, so it is not independent evidence of the
    animals' chronological order; root verifies that against the primary protocol."""
    rows = json.load(open(path))["trials"]; out = {}
    for t in rows:
        out.setdefault((t["species"], t["subject"]), []).append((t["configuration"], t["trial"], t["task_id"]))
    return out


def oracle(env):
    """Positive control with the privileged map + goal (shortest path); a reference, not a competitor."""
    d = bfs(env.task.walkable, env.task.goal)
    while not env.done:
        r, c = env.position
        a = min(range(4), key=lambda i: d[r + NB[i][0], c + NB[i][1]] if 0 <= r + NB[i][0] < d.shape[0] and
                0 <= c + NB[i][1] < d.shape[1] and d[r + NB[i][0], c + NB[i][1]] >= 0 else 1e9)
        env.step(a)
    return env.result()


def random_walk(env, rng):
    while not env.done: env.step(int(rng.integers(4)))
    return env.result()


def check():
    tasks = load_tasks(TASKS); rng = np.random.default_rng(0)
    # 1. dynamics identical to GridEnvironment (path, contact, budget, termination) under the same actions
    for t in tasks:
        a, b = GridEnvironment(t, 0), LocalGridEnvironment(t, 0); acts = rng.integers(5, size=a.budget)
        for x in acts:
            if a.done: break
            a.step(int(x)); b.step(int(x)); assert a.position == b.position and a.contact == b.contact
        assert a.result()["path"] == b.result()["path"] and a.done == b.done
    # 2. goal blindness: moving the goal changes no observation before the goal is entered
    t = tasks[0]; free = [tuple(p) for p in np.argwhere(t.walkable) if tuple(p) not in (t.spawn, t.goal)]
    other = next(p for p in free if bfs(t.walkable, t.spawn)[p] > 0)
    t2 = Task(t.id + "_moved_goal", t.walkable, t.visual_occlusion, t.spawn, other, t.native_cell_m, t.source, t.split)
    for seed in range(20):
        e1, e2 = LocalGridEnvironment(t, seed), LocalGridEnvironment(t2, seed); acts = np.random.default_rng(seed).integers(4, size=200)
        for x in acts:
            if e1.done or e2.done or e1.position in (t.goal, other): break
            assert np.array_equal(e1.observe(), e2.observe()); e1.step(int(x)); e2.step(int(x))
    # 3. wall bits equal the layout; reward fires exactly once, on first goal entry
    e = LocalGridEnvironment(t, 0); o = e.observe(); r, c = t.spawn
    for i, (dr, dc) in enumerate(NB):
        inside = 0 <= r + dr < 10 and 0 <= c + dc < 10
        assert o[2 + i] == float(not (inside and t.walkable[r + dr, c + dc]))
    res = oracle(LocalGridEnvironment(t, 0)); assert res["success"] and res["spl"] == 1.0
    # 4. references over all 250 tasks: privileged oracle and random walk (scripted, no fitting)
    orc = [oracle(LocalGridEnvironment(t, 0))["spl"] for t in tasks]
    rnd = [random_walk(LocalGridEnvironment(t, 0), np.random.default_rng(i)) for i, t in enumerate(tasks)]
    assert min(orc) == 1.0
    orders = animal_orders()
    mono = {k: all(x[:2] <= y[:2] for x, y in zip(v, v[1:])) for k, v in orders.items()}
    print(json.dumps({"ok": True, "tasks": len(tasks), "oracle_spl_min": min(orc),
                      "random_walk_success": float(np.mean([x["success"] for x in rnd])),
                      "random_walk_spl": float(np.mean([x["spl"] for x in rnd])),
                      "subjects": {s: sum(1 for k in orders if k[0] == s) for s in ("rat", "human")},
                      "file_order_is_config_then_trial_all_subjects": all(mono.values())}, indent=1))


if __name__ == "__main__":
    {"check": check}[sys.argv[1]]()
