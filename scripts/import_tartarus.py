"""Import the Tartarus maze (de Cothi et al. 2022, Curr. Biol.): rats (physical maze) and humans (VR) on the SAME
25 obstacle configurations of a 10 x 10 grid, 10 start positions each, one hidden goal.

Sources: Zenodo 10.5281/zenodo.6734981 (trajectories, licence "other-open") and the authors' repository file
"Standard Functions (add to path)/mazes.mat" (configurations; -1 = blocked, 1..10 = start positions).
Writes data/unified/tartarus_tasks.json (frozen Task records, one per configuration x start) and
data/unified/tartarus_trajectories.json (every recorded trial as grid cells, per species and subject).
The grid-index convention and the goal cell are inferred from the data and validated, not assumed.
"""
import glob, json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, scipy.io as sio
from collections import Counter

from arc_fly.unified_env import Task, save_tasks
from arc_fly.unimaze import bfs

SRC = "data/real_sources/decothi2022"


def to_rc(s, order):
    s = int(s) - 1
    return (s % 10, s // 10) if order == "F" else (s // 10, s % 10)


def main():
    mazes = sio.loadmat(f"{SRC}/mazes.mat", squeeze_me=True)["mazes"]
    data = {"rat": sio.loadmat(glob.glob(f"{SRC}/willdecothi-*/Analyses/Rats/rat.mat")[0], squeeze_me=True),
            "human": sio.loadmat(glob.glob(f"{SRC}/willdecothi-*/Analyses/Humans/humans.mat")[0], squeeze_me=True)}
    trials = {"rat": (data["rat"]["rat"], data["rat"]["rat_succ"]), "human": (data["human"]["humans"], data["human"]["human_succ"])}
    # index convention: the one under which every recorded step is a 4-neighbour move between free squares
    scores = {}
    for order in ("F", "C"):
        bad = total = 0
        for sp, (T, _) in trials.items():
            for (subj, cfg, k), traj in np.ndenumerate(T):
                cells = [to_rc(s, order) for s in np.atleast_1d(traj)] if np.size(traj) else []
                for a, b in zip(cells, cells[1:]):
                    total += 1
                    bad += (abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1) or mazes[cfg][b] == -1
        scores[order] = (bad, total)
    order = min(scores, key=lambda o: scores[o][0])
    print("index convention violations (bad steps / steps):", scores, "->", order)
    ends = Counter(to_rc(np.atleast_1d(traj)[-1], order) for sp, (T, S) in trials.items()
                   for (subj, cfg, k), traj in np.ndenumerate(T) if np.size(traj) and S[subj, cfg, k] == 1)
    goal, n_goal = ends.most_common(1)[0]
    print(f"goal cell {goal}: end of {n_goal} of {sum(ends.values())} successful trials")
    tasks, trajs = [], []
    for cfg in range(25):
        m = np.asarray(mazes[cfg]); walk = m != -1
        for start in range(1, 11):
            rc = tuple(int(v) for v in np.argwhere(m == start)[0])
            split = "test" if cfg >= 20 else ("validation" if cfg >= 17 else "train")
            tasks.append(Task(id=f"tartarus/{cfg + 1}/{start}", walkable=walk, visual_occlusion=~walk, spawn=rc,
                              goal=goal, native_cell_m=0.2, split=split,
                              source=dict(dataset="Tartarus maze (de Cothi et al. 2022)", species=["Rattus norvegicus", "Homo sapiens"],
                                          url="https://doi.org/10.5281/zenodo.6734981", configuration=cfg + 1, start=start,
                                          native_cell_m_status="0.2 m assumed; verify against the paper before any metric claim",
                                          goal_status=f"inferred: modal end cell of successful trials ({n_goal}/{sum(ends.values())})",
                                          start_status="trials are matched to tasks by their recorded first cell (task_id in the trajectory file)",
                                          split_definition="configurations 1-17 train, 18-20 validation, 21-25 test")))
    for sp, (T, S) in trials.items():
        for (subj, cfg, k), traj in np.ndenumerate(T):
            if np.size(traj):
                trajs.append(dict(species=sp, subject=int(subj), configuration=cfg + 1, trial=k + 1,
                                  success=int(S[subj, cfg, k]), cells=[list(to_rc(s, order)) for s in np.atleast_1d(traj)]))
    spawn_task = {(int(t.id.split("/")[1]), t.spawn): t.id for t in tasks}
    for t in trajs:                                  # a trial belongs to the task whose spawn is its actual first cell
        t["task_id"] = spawn_task.get((t["configuration"], tuple(t["cells"][0])))
    matched = sum(t["task_id"] is not None for t in trajs)
    order_ok = sum(t["task_id"] == f"tartarus/{t['configuration']}/{t['trial']}" for t in trajs)
    print(f"{len(tasks)} tasks; {len(trajs)} trials ({Counter(t['species'] for t in trajs)}); first cell is one of the "
          f"configuration's 10 starts on {matched}/{len(trajs)} trials; trial number = start number on {order_ok}")
    reach = sum(bfs(t.walkable, t.spawn)[t.goal] >= 0 for t in tasks)
    print(f"goal reachable on {reach}/{len(tasks)} tasks")
    save_tasks(tasks, "data/unified/tartarus")
    json.dump(dict(source="de Cothi et al. 2022", index_order=order, goal=list(goal), trials=trajs),
              open("data/unified/tartarus_trajectories.json", "w"))
    print("wrote data/unified/tartarus/ and data/unified/tartarus_trajectories.json")


if __name__ == "__main__":
    main()
