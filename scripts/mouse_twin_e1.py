"""M-twin / E6 mouse: how much decision information survives the MICrONS digital twin of mouse visual cortex?

The identical E1 poses (96 per task, asserted) are rendered as the MICrONS monitor would show them to the left eye:
a 55.2 x 31.0 cm flat screen 15 cm from the eye (~123 x 92 deg; the MICrONS stimulus geometry), 256 x 144 px,
grayscale luminance gamma-encoded (1/2.2) as monitor pixel values. ASSUMPTION, stated: the virtual monitor is
centred on the left eye's optical axis (azimuth +60 deg, elevation +20 deg; Oommen & Stahl 2008), not on the
recorded population's receptive fields. Each pose plays as 10 gray frames then 15 frames of the view (30 Hz) through
the foundation-model twin of MICrONS scan 8-5 (Wang et al. 2025, Nature; cajal/fnn, MIT; default eye position and
behaviour); features = mean predicted response over the last 10 frames.
Arms: pixels (the input baseline: the twin cannot add information to its input, but a finite probe on PCA-reduced
pixels need not extract all of it, so pixels are NOT a ceiling for the probe), twin, and a random-core twin (the trained perspective,
modulation and readout over a freshly initialised core; a fully untrained twin outputs a constant, so it is no
floor) to ask whether the learned cortical tuning matters beyond the architecture. Labels and probes are E1's (next optimal move, goal side; held-out tasks, 3 seeds).
Run: CUDA_VISIBLE_DEVICES=1 python scripts/mouse_twin_e1.py [--limit K]
"""
import json, os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.getcwd())
import numpy as np, torch

from arc_fly.unified_env import load_tasks
from arc_fly.world3d import build, RayCaster
from arc_fly.species import SPECIES, _dirs, _yaw
from scripts.umwelt_e1 import sample_poses, probe

TWIN_DIR, SESSION, SCAN = "data/real_sources/microns_fnn", 8, 5
H, W, SCREEN_W_M, SCREEN_H_M, SCREEN_D_M = 144, 256, 0.552, 0.310, 0.15
AXIS_AZ, AXIS_EL = np.radians(60.0), np.radians(20.0)
GRAY, SHOW, AVG, BATCH = 10, 15, 10, 64


def screen_dirs():
    """Body-frame ray direction through each monitor pixel (row-major, row 0 at the top)."""
    n = _dirs(AXIS_AZ, AXIS_EL); r = np.cross(n, [0, 0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, n)
    x = ((np.arange(W) + 0.5) / W - 0.5) * SCREEN_W_M; y = (0.5 - (np.arange(H) + 0.5) / H) * SCREEN_H_M
    d = SCREEN_D_M * n + x[None, :, None] * r + y[:, None, None] * u
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)


def render(task, P, sp, dirs, chunk=1_500_000):
    w = build(task, sp.body_width_m, sp.eye_h_m); c = RayCaster(w)
    xy = w.xy_of_rc(P["rc"]); h = P["heading"]; n = len(h); eye = sp.eyes[0][0]           # eyes[0] = left
    o = np.broadcast_to(np.c_[xy, np.zeros(n)][:, None, :] + _yaw(eye[None], h), (n, len(dirs), 3)).reshape(-1, 3)
    d = _yaw(dirs, h).reshape(-1, 3); lum = np.empty(len(o))
    for a in range(0, len(o), chunk):
        dist, gid, nrm = c.cast(o[a:a + chunk], d[a:a + chunk])
        lum[a:a + chunk] = c.shade(o[a:a + chunk], d[a:a + chunk], dist, gid, nrm) @ [0.2126, 0.7152, 0.0722]
    return np.round(255 * np.clip(lum, 0, 1) ** (1 / 2.2)).astype(np.uint8).reshape(n, H, W)


def load_twin(random_core=False):
    from fnn.microns.build import network
    import pandas as pd
    units = int(pd.read_csv(f"{TWIN_DIR}/scans.csv").set_index(["session", "scan_idx"]).loc[(SESSION, SCAN)].units)
    load = lambda f: torch.load(f"{TWIN_DIR}/{f}", map_location="cpu", weights_only=True)   # tensors only
    params = {**load("params_core.pt"), **load(f"params_{SESSION}_{SCAN}.pt")}
    if random_core:
        torch.manual_seed(0); fresh = network(units).state_dict()
        params = {**params, **{k: v for k, v in fresh.items() if k.startswith("core.")}}
    model = network(units); model.load_state_dict(params)
    return model.to("cuda").eval()


@torch.no_grad()
def twin_features(model, img):
    gray = np.full_like(img, 128)
    clip = [gray[..., None]] * GRAY + [img[..., None]] * SHOW                        # T x [N, H, W, 1]
    r = model.predict(stimuli=clip)                                                  # [T, N, U]
    return r[-AVG:].mean(0).astype(np.float32)


def main(limit=None):
    tasks = load_tasks("data/unified/tasks.json"); sp = SPECIES["mouse"]; dirs = screen_dirs()
    rng = np.random.default_rng(0); imgs, y, g, split = [], [], [], []
    for task in tasks:
        P = sample_poses(task, 96, rng)
        if limit is None or sum(1 for t in split if t == task.split) < limit * 96:
            imgs.append(render(task, P, sp, dirs)); y += list(P["move"]); split += [task.split] * len(P["heading"])
        else:
            y += [-1] * len(P["heading"]); split += ["skip"] * len(P["heading"])
    e1 = np.load("data/cvpr/umwelt_e1_features.npz"); y = np.array(y); split = np.array(split); used = split != "skip"
    assert (e1["move"][used] == y[used]).all(), "poses differ from E1"
    y = {"move": e1["move"][used], "goal": e1["goal"][used]}; split = split[used]; imgs = np.concatenate(imgs)
    print(f"{len(imgs)} renders; mean pixel {imgs.mean():.1f}; sd {imgs.std():.1f}", flush=True)
    feats = {"pixels": imgs.reshape(len(imgs), -1).astype(np.float32)}
    for arm, rnd in (("twin", False), ("random_core_twin", True)):
        model = load_twin(rnd)
        feats[arm] = np.concatenate([twin_features(model, imgs[a:a + BATCH]) for a in range(0, len(imgs), BATCH)])
        print(f"  {arm}: {feats[arm].shape}, mean rate {feats[arm].mean():.3f}", flush=True); del model; torch.cuda.empty_cache()
    np.savez("data/cvpr/mouse_twin_e1_features.npz", imgs=imgs, split=split, **y,
             **{k: v for k, v in feats.items() if k != "pixels"})
    out = {"n_poses": int(len(imgs)), "scan": f"{SESSION}-{SCAN}", "units": int(feats["twin"].shape[1])}
    for label, yy in y.items():
        out[f"chance/{label}"] = float(np.bincount(yy[split == "test"]).max() / (split == "test").sum())
        for name, X in feats.items():
            acc = [probe(X, yy, split, sd)[0].mean() for sd in range(3)]
            out[f"{label}/{name}"] = float(np.mean(acc))
            print(f"{label} {name:16s} {np.mean(acc):.3f} (chance {out[f'chance/{label}']:.3f})", flush=True)
            if limit is None: json.dump(out, open("data/cvpr/mouse_twin_e1.json", "w"), indent=1)




def stats():
    """Task-clustered intervals from the cached features (identical E1 poses; task ids from the E1 cache)."""
    from arc_fly.stats import cluster_summary
    z = np.load("data/cvpr/mouse_twin_e1_features.npz"); e1 = np.load("data/cvpr/umwelt_e1_features.npz")
    assert (z["move"] == e1["move"]).all() and (z["split"] == e1["split"]).all()
    split, te = z["split"], z["split"] == "test"; out = json.load(open("data/cvpr/mouse_twin_e1.json")); out["intervals"] = {}
    feats = {"pixels": z["imgs"].reshape(len(z["imgs"]), -1).astype(np.float32), "twin": z["twin"], "random_core_twin": z["random_core_twin"]}
    for label in ("move", "goal"):
        y = z[label]; corr = {k: np.mean([probe(X, y, split, sd)[0] for sd in range(3)], 0) for k, X in feats.items()}
        r = cluster_summary(e1["task"][te], corr, out[f"chance/{label}"], [("twin", "pixels"), ("twin", "random_core_twin"), ("pixels", "random_core_twin")])
        out["intervals"][label] = r; print(label, {k: (round(v.get("acc", v.get("diff")), 3), [round(x, 3) for x in v["ci95"]]) for k, v in r.items()}, flush=True)
    json.dump(out, open("data/cvpr/mouse_twin_e1.json", "w"), indent=1)


if __name__ == "__main__":
    stats() if "--stats" in sys.argv else main(int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None)
