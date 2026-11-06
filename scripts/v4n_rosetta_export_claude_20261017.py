"""Frozen activation export for the unpaired cross-species alignment analysis (Claude, 2026-10-17; Codex owns the
Rosetta/Wasserstein-Procrustes method; this file only exports embeddings, computes no alignment and no outcome).

Fixed evaluation set (seeded, outcome-blind): 500 unique frames of source layouts 1-5 + 500 unique frames of the 16
exposed E2 rooms, drawn from the frozen MICrONS frame set (data/wacv/microns_perception_20261017/frames.npy).
Observation per frame = the pipeline's own first-step observation: pooled camera (144), contact 0, reward 0, no last
action, and the pixel-space observer's one-frame posterior (+ coordinates), exactly as causal_rows builds step 0.
Per checkpoint we save body layer-1 and layer-2 tanh activations (64 + 64) and policy logits (4), with the
checkpoint's own input hooks active (camera-only masks apply). Embedding sets:
  seeds30      rat_bc, rat_legal_uniform, rat_sham, human_bc, human_legal_uniform, cold_ppo x seeds 1000-1029 (full input)
  camera_only  human_bc, human_legal_uniform, cold_ppo x seeds 1200-1209 (input_ablation no_posterior)
  microns      twin / rcore raw predicted responses (9,941 units) to the same frames; pooled pixels as input baseline
Rat/human sets carry recorded-CHOICE provenance (behaviour), not neural recordings; MICrONS is a model of recorded
mouse V1 neurons, not responses recorded to these frames.
  python scripts/v4n_rosetta_export_claude_20261017.py frames | agents | microns
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve(); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1]))
import torch
import input_ablation_claude as S
import wacv_native_prior_confirm_env as CONFIRM
import wacv_native_behavior_prior_20261012 as RUN

ROOT = HERE.parents[1]; MP = ROOT / "data/wacv/microns_perception_20261017"; OUT = ROOT / "data/wacv/rosetta_export_20261017"
ash = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
SEEDS30 = ROOT / "data/wacv/seeds30/cohort"; ABL = ROOT / "data/wacv/input_ablation/cohort"
ARMS30 = ("rat_bc", "rat_legal_uniform", "rat_sham", "human_bc", "human_legal_uniform", "cold_ppo")


def frames():
    _, cache, _ = S.run.load(); raw = cache.raw; valid = np.isfinite(raw.features[..., 0])
    X = np.load(MP / "frames.npy"); pos = {x.tobytes(): i for i, x in enumerate(X)}
    src = sorted({pos[raw.images[tuple(i)].tobytes()] for i in np.argwhere(valid) if raw.index[1] <= i[0] <= raw.index[5]})
    n_src = json.loads((MP / "views.json").read_text())["unique_source_frames"]; rooms = list(range(n_src, len(X)))
    assert not set(src) & set(rooms)
    rng = np.random.default_rng(20261017); idx = np.r_[np.sort(rng.choice(src, 500, replace=False)), np.sort(rng.choice(rooms, 500, replace=False))]
    from tartarus_visual_env import View
    view = View(**raw.protocol["view"]); cam = np.stack([CONFIRM.pool_pixels(X[i] / 255.0, view, raw.protocol["features"])[1] for i in idx])
    atlas = raw.features[0, :, :, :, 0, :64].reshape(100, 4, 64); obs = []
    for c in cam:
        f = RUN.LandmarkBelief(atlas, .03); f.observe(c[:64], None, False); p = f.p.ravel().astype(np.float32)   # = causal_rows step 0
        obs.append(np.concatenate((c, np.zeros(6, np.float32), p, p @ RUN.parent.COORD)))
    obs = np.asarray(obs, np.float32); assert obs.shape == (1000, 252) and np.isfinite(obs).all()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "eval_set.npz", frame_index=idx, obs=obs, domain=np.r_[np.zeros(500, int), np.ones(500, int)])
    (OUT / "eval_set.json").write_text(json.dumps(dict(frames_file_sha256=hashlib.sha256((MP / "frames.npy").read_bytes()).hexdigest(),
        frame_index_sha256=ash(idx), obs_sha256=ash(obs), domains={"0": "source layouts 1-5 (500)", "1": "exposed E2 rooms (500)"}, seed=20261017), indent=1))


@torch.no_grad()
def embed(ckpt, obs):
    m = torch.load(ckpt, map_location="cpu", weights_only=False); b = m.net.body; acts = {}
    hs = [b[1].register_forward_hook(lambda mod, i, o: acts.__setitem__("h1", o)), b[3].register_forward_hook(lambda mod, i, o: acts.__setitem__("h2", o))]
    h = b(torch.as_tensor(obs)); logits = m.net.pi(h)
    for x in hs: x.remove()
    return acts["h1"].numpy(), acts["h2"].numpy(), logits.numpy()


def agents():
    torch.set_num_threads(4); obs = np.load(OUT / "eval_set.npz")["obs"]; out = {}; meta = []
    sets = [("seeds30", a, s, SEEDS30 / a / f"seed{s}") for a in ARMS30 for s in range(1000, 1030)]
    sets += [("camera_only", a, s, ABL / f"{a}__no_posterior" / f"seed{s}") for a in ("human_bc", "human_legal_uniform", "cold_ppo") for s in range(1200, 1210)]
    for group, a, s, d in sets:
        r = json.loads((d / "result.json").read_text()); ck = d / "checkpoint_source.pt"
        assert r["status"] == "complete" and hashlib.sha256(ck.read_bytes()).hexdigest() == r["checkpoint_sha256"]
        h1, h2, lg = embed(ck, obs); k = f"{group}/{a}/seed{s}"
        out[f"{k}/h1"], out[f"{k}/h2"], out[f"{k}/logits"] = h1, h2, lg; meta.append(dict(key=k, checkpoint_sha256=r["checkpoint_sha256"]))
    np.savez_compressed(OUT / "agents.npz", **out)
    (OUT / "agents.json").write_text(json.dumps(dict(eval_set_sha256=hashlib.sha256((OUT / "eval_set.npz").read_bytes()).hexdigest(),
                                                     checkpoints=meta, agents_npz_sha256=hashlib.sha256((OUT / "agents.npz").read_bytes()).hexdigest()), indent=1))
    print(len(meta), "checkpoints exported")


def microns():
    idx = np.load(OUT / "eval_set.npz")["frame_index"]; out = dict(pooled_pixels=np.load(OUT / "eval_set.npz")["obs"][:, :144]); meta = {}
    for perc in ("twin", "rcore"):
        R = np.load(MP / f"raw_{perc}_part0of1.npy", mmap_mode="r"); out[perc] = np.asarray(R[idx], np.float32)
        meta[perc] = dict(raw_file_sha256=hashlib.sha256((MP / f"raw_{perc}_part0of1.npy").read_bytes()).hexdigest(), units=int(R.shape[1]))
    np.savez_compressed(OUT / "microns.npz", **out)
    (OUT / "microns.json").write_text(json.dumps(dict(meta, raw_generation_sha256=hashlib.sha256((MP / "raw_generation.json").read_bytes()).hexdigest(),
        microns_npz_sha256=hashlib.sha256((OUT / "microns.npz").read_bytes()).hexdigest()), indent=1))


if __name__ == "__main__":
    dict(frames=frames, agents=agents, microns=microns)[sys.argv[1]]()
