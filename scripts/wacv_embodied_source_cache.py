"""Source-development visual cache. No learner fit, reward outcome or target evaluation.

Only the registered open arena and configurations 1--7 are rendered. The public
geometry supplies synthetic views; these are not recorded animal retinal data.
Cached pixel features are indexed inside the environment and never expose pose,
layout, goal, task split or curriculum identity to a policy.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from arc_fly.unified_env import Task, load_tasks
from tartarus_visual_env import View, render, HEADINGS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_geometry(protocol):
    tasks = load_tasks(ROOT / protocol["tasks"])
    configs = protocol["source"]["train_configurations"] + protocol["source"]["select_configurations"]
    by = {}
    for t in tasks:
        cfg = t.source["configuration"]
        if cfg in configs:
            by.setdefault(cfg, []).append(t)
    if set(by) != set(configs):
        raise ValueError("Source configuration inventory mismatch")
    out = []
    # Open arena is the published initial training condition, with the imported
    # goal. This is still a digital reconstruction, not an additional observation.
    goal = tasks[0].goal
    out.append(dict(configuration=0, walkable=np.ones((10, 10), bool),
                    goal=goal, native_starts=[tuple(t.spawn) for t in by[configs[0]]],
                    original_task_ids=[]))
    for cfg in configs:
        group = sorted(by[cfg], key=lambda t: t.source["start"])
        if len(group) != 10 or any(t.goal != goal or not np.array_equal(t.walkable, group[0].walkable) for t in group):
            raise ValueError("Source geometry/start mismatch")
        out.append(dict(configuration=cfg, walkable=group[0].walkable.copy(),
                        goal=goal, native_starts=[t.spawn for t in group],
                        original_task_ids=[t.id for t in group]))
    return out


def pool_pixels(image, view, spec):
    # Eight-bit camera quantization is explicit and identical for every learner.
    pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    oh, ow = spec["pooled_height"], spec["pooled_width"]
    if view.height % oh or view.width % ow:
        raise ValueError("Pooling dimensions must divide image dimensions")
    x = pixels.astype(np.float32) / np.float32(255)
    pooled = x.reshape(oh, view.height // oh, ow, view.width // ow).mean((1, 3))
    return pixels, (2 * pooled - 1).reshape(-1).astype(np.float32)


def build(protocol_path, out_dir):
    protocol_path, out_dir = Path(protocol_path), Path(out_dir)
    protocol = json.loads(protocol_path.read_text())
    view = View(**protocol["view"])
    if sha(ROOT / "scripts/tartarus_visual_env.py") != protocol["renderer_sha256"]:
        raise ValueError("Registered renderer changed")
    geometries = source_geometry(protocol)
    rotations = protocol["features"]["texture_rotations"]
    shape = (len(geometries), 10, 10, len(HEADINGS), len(rotations))
    images = np.zeros(shape + (view.height, view.width), np.uint8)
    features = np.full(shape + (protocol["features"]["pooled_height"] *
                               protocol["features"]["pooled_width"],), np.nan, np.float32)
    start = time.perf_counter()
    frames = 0
    for gi, g in enumerate(geometries):
        for r, c in np.argwhere(g["walkable"]):
            for hi, heading in enumerate(HEADINGS):
                for ti, rotation in enumerate(rotations):
                    image, _, _ = render(view, g["walkable"], (int(r), int(c)), heading, rotation)
                    pixels, x = pool_pixels(image, view, protocol["features"])
                    images[gi, r, c, hi, ti] = pixels
                    features[gi, r, c, hi, ti] = x
                    frames += 1
        print(json.dumps({"configuration":g["configuration"], "frames":frames,
                          "elapsed_s":round(time.perf_counter()-start, 2)}), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    array_path = out_dir / "source_visual_cache.npz"
    np.savez_compressed(array_path, images=images, features=features,
                        walkable=np.stack([g["walkable"] for g in geometries]))
    meta = dict(protocol_sha256=sha(protocol_path), renderer_sha256=protocol["renderer_sha256"],
                builder_sha256=sha(__file__), array_sha256=sha(array_path), frames=frames,
                shape=list(features.shape), image_dtype="uint8", feature_dtype="float32",
                view=protocol["view"], feature_contract=protocol["features"],
                configurations=[dict(configuration=g["configuration"], goal=g["goal"],
                                     native_starts=g["native_starts"],
                                     original_task_ids=g["original_task_ids"])
                                for g in geometries],
                elapsed_s=time.perf_counter()-start, target_data_rendered=False,
                raw_sensation=False)
    meta["manifest_sha256"] = signature(meta)
    (out_dir / "source_visual_cache.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps({"complete":True, "frames":frames, "arrays":str(array_path),
                      "array_sha256":meta["array_sha256"]}), flush=True)


class SourceCache:
    def __init__(self, manifest_path, protocol_path):
        self.path = Path(manifest_path)
        self.protocol_path = Path(protocol_path)
        self.protocol = json.loads(self.protocol_path.read_text())
        self.meta = json.loads(self.path.read_text())
        expected = self.meta.pop("manifest_sha256")
        if signature(self.meta) != expected:
            raise ValueError("Cache manifest signature mismatch")
        if sha(self.protocol_path) != self.meta["protocol_sha256"]:
            raise ValueError("Cache protocol mismatch")
        if sha(ROOT / "scripts/tartarus_visual_env.py") != self.meta["renderer_sha256"]:
            raise ValueError("Cache renderer mismatch")
        array = self.path.with_suffix(".npz")
        if sha(array) != self.meta["array_sha256"]:
            raise ValueError("Cache array mismatch")
        with np.load(array, allow_pickle=False) as a:
            self.features, self.images, self.walkable = a["features"], a["images"], a["walkable"]
        self.index = {x["configuration"]:i for i,x in enumerate(self.meta["configurations"])}
        self.goal = tuple(self.meta["configurations"][0]["goal"])
        self.rotations = self.protocol["features"]["texture_rotations"]
        self.obs_dim = self.features.shape[-1] + 2 + 4
        if self.obs_dim != self.protocol["features"]["obs_dim"]:
            raise ValueError("Observation dimension mismatch")

    def frame(self, cfg, position, heading, rotation_index):
        gi = self.index[cfg]
        r, c = position
        x = self.features[gi, r, c, HEADINGS.index(tuple(heading)), rotation_index]
        if not np.isfinite(x).all():
            raise ValueError("Invalid camera pose")
        return x.copy()

    def geometry(self, cfg):
        # Environment-side metadata. A learner receives only observation().
        i = self.index[cfg]
        return self.walkable[i], self.meta["configurations"][i]["native_starts"]


def audit(protocol_path, manifest_path):
    cache = SourceCache(manifest_path, protocol_path)
    view = View(**cache.protocol["view"])
    gs = source_geometry(cache.protocol)
    rng = np.random.default_rng(731)
    comparisons = 0
    max_pool_error = 0.0
    for gi, g in enumerate(gs):
        cells = np.argwhere(g["walkable"])
        for _ in range(12):
            r,c = cells[rng.integers(len(cells))]
            hi = int(rng.integers(4)); ti = int(rng.integers(len(cache.rotations)))
            image, _, _ = render(view,g["walkable"],(int(r),int(c)),HEADINGS[hi],cache.rotations[ti])
            # Independent pooling, avoiding pool_pixels.
            pixels = np.rint(np.clip(image,0,1)*255).astype(np.uint8)
            assert np.array_equal(pixels,cache.images[gi,r,c,hi,ti])
            ref = np.array([[pixels[a*4:(a+1)*4,b*4:(b+1)*4].astype(np.float32).mean()/np.float32(255)*2-1
                            for b in range(16)] for a in range(9)],np.float32).ravel()
            err = float(np.max(np.abs(ref-cache.features[gi,r,c,hi,ti])))
            if err > 3e-7:
                raise AssertionError(err)
            max_pool_error = max(max_pool_error,err); comparisons += 1
    assert set(cache.index) == {0,1,2,3,4,5,6,7}
    assert not cache.meta["target_data_rendered"]
    valid = np.repeat(np.repeat(cache.walkable[...,None,None],4,3),len(cache.rotations),4)
    assert np.isfinite(cache.features[valid]).all()
    assert np.isnan(cache.features[~valid]).all()
    report=dict(ok=True, source_only=True, comparisons=comparisons, obs_dim=cache.obs_dim,
                max_independent_pool_error=max_pool_error, array_sha256=cache.meta["array_sha256"],
                protocol_sha256=cache.meta["protocol_sha256"], configurations=sorted(cache.index),
                frames=cache.meta["frames"], raw_retinal_data=False, fits=0)
    p=Path(manifest_path).parent/"source_visual_cache_audit.json"
    p.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["build","audit"])
    parser.add_argument("--protocol",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    if args.command=="build":build(args.protocol,args.out)
    else:audit(args.protocol,Path(args.out)/"source_visual_cache.json")
