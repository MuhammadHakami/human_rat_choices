"""E1 umwelt information: how much decision information each species' senses carry, alone and fused.

For every frozen task, the SAME poses (grid position + heading) are placed in each species' own world (same
layout, scaled to its body) and sensed through its suite (arc_fly.species). A small MLP probe, identical for
every species and modality, is trained on the train-split tasks, early-stopped on the validation split and
scored on the held-out test split. Labels:
  move  -- egocentric bearing of the optimal next move (front / left / back / right), from the BFS field
  goal  -- egocentric bearing of the straight line to the goal (front / left / back / right)
Run (GPU 1):  CUDA_VISIBLE_DEVICES=1 python scripts/umwelt_e1.py [--poses 48] [--seeds 3]
Outputs data/cvpr/umwelt_e1.json (+ features npz, git-ignored).
"""
import argparse, json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.getcwd())
import numpy as np

from arc_fly.unified_env import load_tasks
from arc_fly.unimaze import NB, bfs
from arc_fly.world3d import build, RayCaster, WORLD_CONTRACT
from arc_fly.species import SPECIES, sense

OUT = "data/cvpr"
MODALITIES = ("vision", "olfaction", "touch", "all")
PCA_DIM = 256
LINEAR = False                                   # robustness: a linear readout instead of the MLP


def bearing_class(vec_xy, heading):
    """Egocentric bearing of a world vector -> 0 front, 1 left, 2 back, 3 right."""
    ang = (np.arctan2(vec_xy[..., 1], vec_xy[..., 0]) - heading + np.pi) % (2 * np.pi) - np.pi
    return np.round(ang / (np.pi / 2)).astype(int) % 4


def sample_poses(task, n, rng):
    """Half on the optimal path (heading toward the next optimal cell +- 60 deg), half anywhere walkable."""
    d = bfs(task.walkable, task.goal)
    path, cell = [task.spawn], task.spawn
    while cell != task.goal:
        cell = min(((cell[0] + a, cell[1] + b) for a, b in NB
                    if 0 <= cell[0] + a < d.shape[0] and 0 <= cell[1] + b < d.shape[1] and d[cell[0] + a, cell[1] + b] >= 0),
                   key=lambda k: d[k])
        path.append(cell)
    path = path[:-1]                                                  # the goal cell has no next move
    on = [path[i] for i in rng.choice(len(path), min(n // 2, len(path)), replace=False)]
    free = np.argwhere((d > 0))
    off = [tuple(free[i]) for i in rng.choice(len(free), n - len(on), replace=len(free) < n - len(on))]
    rows = []
    for k, cells in ((0, on), (1, off)):
        for r, c in cells:
            nb = [(a, b) for a, b in NB if 0 <= r + a < d.shape[0] and 0 <= c + b < d.shape[1]
                  and d[r + a, c + b] >= 0 and d[r + a, c + b] < d[r, c]]
            mv = np.mean([[b, -a] for a, b in nb], 0)                 # optimal move(s) as a world vector
            h = (np.arctan2(mv[1], mv[0]) + rng.uniform(-np.pi / 3, np.pi / 3)) if k == 0 else rng.uniform(-np.pi, np.pi)
            rows.append((r + rng.uniform(-0.25, 0.25), c + rng.uniform(-0.25, 0.25), h, mv[0], mv[1], k))
    a = np.array(rows, float)
    g = np.array([task.goal[1] - a[:, 1], -(task.goal[0] - a[:, 0])]).T  # world vector to goal (x = col, y = -row)
    return dict(rc=a[:, :2], heading=a[:, 2], move=bearing_class(a[:, 3:5], a[:, 2]),
                goal=bearing_class(g, a[:, 2]), on_path=a[:, 5] == 0)


def features(tasks, n_poses, seed):
    rng = np.random.default_rng(seed)
    F = {s: {m: [] for m in ("vision", "olfaction", "touch")} for s in SPECIES}
    meta = dict(task=[], split=[], move=[], goal=[], on_path=[])
    t0 = time.time()
    for i, task in enumerate(tasks):
        P = sample_poses(task, n_poses, rng)
        n = len(P["heading"])
        meta["task"] += [task.id] * n; meta["split"] += [task.split] * n
        for k in ("move", "goal", "on_path"): meta[k].append(P[k])
        for name, sp in SPECIES.items():
            w = build(task, sp.body_width_m, sp.eye_h_m)
            s = sense(w, RayCaster(w), sp, w.xy_of_rc(P["rc"]), P["heading"])
            for m in ("vision", "olfaction", "touch"):
                if s[m] is not None: F[name][m].append(s[m].reshape(n, -1))
        if i % 10 == 0:
            print(f"  features {i + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    for k in ("move", "goal", "on_path"): meta[k] = np.concatenate(meta[k])
    for s in F:
        for m in list(F[s]):
            F[s][m] = np.concatenate(F[s][m]) if F[s][m] else None
    return F, meta


def olf(sp, X):
    """Paired chemosensors (left/right adjacent per sample) -> per-sample means, then left-minus-right differences. The
    probe z-scores each feature, so raw pairs hide the bilateral difference (a tiny fraction of the shared level):
    sensing_law.py found snapshot nostril information at chance raw and well above chance re-expressed. Other layouts
    (the cell's 32-receptor ring) pass through."""
    if X is None or len(SPECIES[sp].chemo) != 2: return X
    P = X.reshape(len(X), -1, 2); return np.concatenate([P.mean(2), P[:, :, 0] - P[:, :, 1]], 1).astype(np.float32)


def _fit_reducer(X, tr, dev, q):
    """z-score on the training split, then PCA to q dimensions fitted on the training split only.
    Returns the reduced data and the same transform as a function for new data."""
    import torch
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev); V = pm = ps = None
    if Xt.shape[1] > q:
        trt = torch.tensor(tr, device=dev)
        V = torch.pca_lowrank(Xt[trt], q=q, center=False)[2][:, :q]
        Xt = Xt @ V; pm, ps = Xt[trt].mean(0), Xt[trt].std(0) + 1e-6
        Xt = (Xt - pm) / ps

    def transform(Xn):
        Z = torch.tensor((Xn - mu) / sd, dtype=torch.float32, device=dev)
        return Z if V is None else (Z @ V - pm) / ps
    return Xt, transform


def _reduce(X, tr, dev, q):
    return _fit_reducer(X, tr, dev, q)[0]


def _train(Xt, y, tr, va, dev, epochs, patience):
    import torch
    yt = torch.tensor(y, device=dev)
    net = (torch.nn.Linear(Xt.shape[1], 4) if LINEAR else
           torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                               torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 4))).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    itr = torch.tensor(np.flatnonzero(tr), device=dev)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train()
        for b in itr[torch.randperm(len(itr), device=dev)].split(512):
            opt.zero_grad(); torch.nn.functional.cross_entropy(net(Xt[b]), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(Xt[va]).argmax(1) == yt[va]).float().mean().item()
        if acc > best: best, bad, best_state = acc, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad > patience: break
    net.load_state_dict(best_state); net.eval()
    return net, best


def probe(X, y, split, seed, epochs=400, patience=30, return_logp=False):
    """X is one array or a list of per-modality arrays; each modality is reduced on its own (PCA <= PCA_DIM / n)
    before concatenation, so a high-dimensional sense cannot crowd out a low-dimensional one."""
    import torch
    torch.manual_seed(seed)
    dev = "cuda"
    tr, va, te = (split == "train"), (split == "validation"), (split == "test")
    parts = X if isinstance(X, list) else [X]
    Xt = torch.cat([_reduce(P, tr, dev, max(16, PCA_DIM // len(parts))) for P in parts], 1)
    net, best = _train(Xt, y, tr, va, dev, epochs, patience)
    if return_logp:
        with torch.no_grad():
            lp = torch.log_softmax(net(Xt), 1).cpu().numpy()
        return lp[va], lp[te]
    with torch.no_grad():
        pred = net(Xt[te]).argmax(1).cpu().numpy()
    return pred == y[te], best


def fit_probe(X, y, split, seed, epochs=400, patience=30):
    """The probe above, trained once and returned as a function: logp(X_new) -> (n, 4) log-probabilities, applying the
    training-split reduction to new data (e.g. counterfactual worlds). Returns (logp, validation accuracy)."""
    import torch
    torch.manual_seed(seed)
    dev = "cuda"
    tr, va = (split == "train"), (split == "validation")
    parts = X if isinstance(X, list) else [X]
    red = [_fit_reducer(P, tr, dev, max(16, PCA_DIM // len(parts))) for P in parts]
    net, best = _train(torch.cat([r[0] for r in red], 1), y, tr, va, dev, epochs, patience)

    def logp(Xn):
        pn = Xn if isinstance(Xn, list) else [Xn]
        with torch.no_grad():
            return torch.log_softmax(net(torch.cat([r[1](P) for r, P in zip(red, pn)], 1)), 1).cpu().numpy()
    return logp, best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--poses", type=int, default=96); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    tasks = load_tasks("data/unified/tasks.json")
    if a.limit:                                                      # smoke test: equal share from each split
        tasks = [t for sp in ("train", "validation", "test") for t in [x for x in tasks if x.split == sp][:a.limit // 3]]
    os.makedirs(OUT, exist_ok=True)
    cache = f"{OUT}/umwelt_e1_features.npz"
    if a.reuse and os.path.exists(cache):                              # features are deterministic given seed 0
        z = np.load(cache)
        F = {s: {m: (z[f"{s}_{m}"] if f"{s}_{m}" in z else None) for m in ("vision", "olfaction", "touch")} for s in SPECIES}
        meta = dict(move=z["move"], goal=z["goal"], on_path=z["on_path"], split=list(z["split"]), task=list(z["task"]))
        print(f"reused features from {cache}", flush=True)
    else:
        F, meta = features(tasks, a.poses, seed=0)
    split = np.array(meta["split"])
    if not (a.reuse and os.path.exists(cache)): np.savez_compressed(f"{OUT}/umwelt_e1_features.npz", **{f"{s}_{m}": v for s in F for m, v in F[s].items() if v is not None},
                        move=meta["move"], goal=meta["goal"], on_path=meta["on_path"], split=split, task=np.array(meta["task"]))
    res = dict(contract=WORLD_CONTRACT, poses_per_task=a.poses, n_tasks=len(tasks),
               n_test=int((split == "test").sum()), chance={}, results={})
    rng = np.random.default_rng(0)
    for label in ("move", "goal"):
        y = meta[label]
        res["chance"][label] = float(np.bincount(y[split == "test"], minlength=4).max() / (split == "test").sum())
        for s in SPECIES:
            for m in MODALITIES:
                G = {**F[s], "olfaction": olf(s, F[s]["olfaction"])}
                parts = [G[k] for k in ("vision", "olfaction", "touch") if G[k] is not None] if m == "all" else \
                        ([G[m]] if G[m] is not None else [])
                if not parts: continue
                X = parts if len(parts) > 1 else parts[0]
                correct = [probe(X, y, split, seed)[0] for seed in range(a.seeds)]
                acc = np.array([c.mean() for c in correct])
                boot = [np.mean(c[rng.integers(0, len(c), len(c))]) for c in correct for _ in range(200)]
                res["results"][f"{label}/{s}/{m}"] = dict(acc=float(acc.mean()), seeds=acc.tolist(),
                                                          ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                                          dim=int(sum(p.shape[1] for p in parts)))
                print(f"{label:4s} {s:8s} {m:9s} acc {acc.mean():.3f} (seeds {np.round(acc, 3).tolist()})", flush=True)
                json.dump(res, open(f"{OUT}/umwelt_e1.json", "w"), indent=1)
    print("chance:", res["chance"])


if __name__ == "__main__":
    main()
