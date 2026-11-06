"""Tartarus closed-loop contract with the SOURCE agent budget: a fixed 45 decisions per trial.

de Cothi et al. 2022 (Curr Biol, STAR Methods pp. e1-e3; root-verified PDF): animals had 45 s per trial and the
source grid agents 45 transitions at 1 transition/s. scripts/tartarus_closedloop_env.LocalGridEnvironment inherited
GridEnvironment's 6*optimal_path+50 budget, which depends on the hidden map and goal; this prototype fixes it.
This is a CUSTOM grid comparison with the SOURCE AGENTS' 45-transition rule, not physical-time equivalence with
the animals (rats on 0.2 m tiles, humans in scaled VR). Sensor and dynamics are unchanged from
tartarus_closedloop_env (goal-blind local sensing); the original unified/hash-bound files are not modified.
  python scripts/tartarus_closedloop_env45.py check
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tartarus_closedloop_env import LocalGridEnvironment, TASKS, oracle, random_walk
from arc_fly.unified_env import Task, load_tasks
from arc_fly.unimaze import bfs

BUDGET = 45


class Local45Environment(LocalGridEnvironment):
    def __init__(self, task, seed=0):
        super().__init__(task, seed)
        self.budget = BUDGET          # fixed; independent of layout, start and goal


def check():
    tasks = load_tasks(TASKS); rng = np.random.default_rng(0)
    # 1. the budget is 45 for every task, and an always-blocked agent stops after exactly 45 decisions
    for t in tasks:
        e = Local45Environment(t, 0); assert e.budget == BUDGET
        n = 0
        while not e.done: e.step(4); n += 1           # 'wait' never reaches the goal unless spawn == goal
        assert n == (0 if t.spawn == t.goal else BUDGET), (t.id, n)
    # 2. moving the goal (and changing the layout) cannot change the budget
    t = tasks[0]; free = [tuple(p) for p in np.argwhere(t.walkable) if tuple(p) not in (t.spawn, t.goal)]
    for g in free[:: max(1, len(free) // 10)]:
        if bfs(t.walkable, t.spawn)[g] <= 0: continue
        moved = Task(t.id + f"_goal{g}", t.walkable, t.visual_occlusion, t.spawn, g, t.native_cell_m, t.source, t.split)
        assert Local45Environment(moved, 0).budget == BUDGET
    assert len({Local45Environment(x, 0).budget for x in tasks}) == 1
    # 3. references under the fixed budget (scripted, no fitting): shortest paths all fit inside 45?
    opt = np.array([x.opt for x in tasks]); orc = [oracle(Local45Environment(x, 0)) for x in tasks]
    rnd = [random_walk(Local45Environment(x, 0), np.random.default_rng(i)) for i, x in enumerate(tasks)]
    assert all(o["success"] for o, k in zip(orc, opt) if k <= BUDGET)
    print(json.dumps({"ok": True, "budget": BUDGET, "max_optimal_path": int(opt.max()),
                      "tasks_with_optimal_path_over_budget": int((opt > BUDGET).sum()),
                      "oracle_success": float(np.mean([o["success"] for o in orc])),
                      "random_walk_success": float(np.mean([x["success"] for x in rnd])),
                      "random_walk_spl": float(np.mean([x["spl"] for x in rnd]))}, indent=1))


if __name__ == "__main__":
    {"check": check}[sys.argv[1]]()
