"""12-case learner follow-up for the label controls (surrogate, legal-uniform) x {rat, human} x seeds 60-62.
Standalone: imports the bound 21-case module read-only (build, select_arm, parent dynamics) and changes ONLY the BC label
source. X comes from the original prior arrays (hash-checked); labels from surrogate_v2/{species}_labels.npz (hash-checked
against its manifest). BC order RNG, budgets, PPO, curriculum and 6-7 selection are the original ones. Needs a root GO.
  python scripts/native_prior_controls_runner_20261012.py dry  rat_surrogate 0 --out OUT --device cuda:0
  python scripts/native_prior_controls_runner_20261012.py fit  rat_surrogate 60 --out OUT --device cuda:0
"""
import argparse, json, os, resource, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wacv_native_behavior_prior_20261012 as run
import native_prior_surrogate_v2_20261012 as sur

ROOT, BASE, parent = run.ROOT, run.BASE, run.parent
DeviceFFAgent = run.DeviceFFAgent  # checkpoint class
LABELS = BASE / "surrogate_v2"
GO = BASE / "controls_GO.json"
ARMS = ("rat_surrogate", "human_surrogate", "rat_legal_uniform", "human_legal_uniform")


def hashes():
    h = dict(source=run.hashes(), surrogate_protocol=parent.sha(sur.PROTOCOL), surrogate_script=parent.sha(sur.__file__),
             runner=parent.sha(__file__))
    if (LABELS / "manifest.json").exists(): h["labels_manifest"] = parent.sha(LABELS / "manifest.json")
    return h


def labels(arm, dry):
    species, key = arm.split("_", 1)
    meta = next(r for r in json.loads((BASE / "data_manifest.json").read_text())["datasets"] if r["species"] == species)
    with np.load(ROOT / meta["path"]) as a:
        assert parent.sha(ROOT / meta["path"]) == meta["sha256"]
        X, ident = a["X"], a["identity"]
    if dry:   # technical only: legal-uniform at a TEST seed, never a registered label seed
        Fe = sur.features(species, run.load()[1], json.loads((ROOT / "data/unified/tartarus_trajectories.json").read_text()))
        return X, sur.sample(np.zeros(2), Fe, 11)
    m = json.loads((LABELS / "manifest.json").read_text())[species]
    f = ROOT / m["path"]; assert parent.sha(f) == m["sha256"]
    with np.load(f) as b:
        assert np.array_equal(b["identity"], ident), "labels must be at exactly the prior rows"
        return X, b[key]


def pretrain(net, arm, seed, P, dry=False):
    """run.pretrain with only the label source changed (same RNG stream, batches, optimiser, clip, critic freeze)."""
    X, y = labels(arm, dry); dev = next(net.net.parameters()).device
    x = torch.as_tensor(X.reshape(-1, 252), device=dev); y = torch.as_tensor(np.repeat(y, 3), device=dev)
    rng = np.random.default_rng([20261012, 601, seed])
    opt = torch.optim.Adam(net.net.parameters(), lr=P["bc"]["learning_rate"], eps=1e-5)
    critic = [p.detach().clone() for p in net.net.v.parameters()]
    presentations = updates = 0; stats = []
    for _ in range(1 if dry else P["bc"]["epochs"]):
        order = rng.permutation(len(x))
        if dry: order = order[:512]
        s = 0.
        for start in range(0, len(order), P["bc"]["batch_size"]):
            ii = torch.as_tensor(order[start:start + P["bc"]["batch_size"]], device=dev)
            loss = F.cross_entropy(net.net.pi(net.net.body(x[ii])), y[ii])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.net.parameters(), P["bc"]["gradient_clip"]); opt.step()
            presentations += len(ii); updates += 1; s += float(loss.detach()) * len(ii)
        stats.append(s / len(order))
    assert all(torch.equal(a, b) for a, b in zip(critic, net.net.v.parameters()))
    net.opt = torch.optim.Adam(net.net.parameters(), lr=.001, eps=1e-5); assert not net.opt.state
    return dict(unique_native_moves=6000, labeled_presentations=presentations, updates=updates, cross_entropy_per_epoch=stats,
                critic_weights_unchanged=True, sensory_frames_measured=False, label_arm=arm)


def fit(arm, seed, out, dry, device):
    P, cache, L = run.load(); C = json.loads(sur.PROTOCOL.read_text())
    assert arm in ARMS and (dry or seed in C["learner_followup"]["seeds"])
    if not dry:
        g = json.loads(GO.read_text())
        assert g["status"] == "GO" and g["hashes"] == hashes() and Path(out).resolve() == Path(g["output"]).resolve()
    d = Path(out) / arm / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    with (d / "started.json").open("x") as f: json.dump(dict(hashes=hashes(), arm=arm, seed=seed, dry=dry), f)
    r = dict(arm=arm, seed=seed, dry=dry, hashes=hashes(), status="running"); t0 = time.perf_counter(); env = None
    try:
        net = run.build(seed, device, L)
        r["bc"] = pretrain(net, arm, seed, P, dry)
        torch.save(net, d / "checkpoint_prior.pt"); r["prior_sha256"] = parent.sha(d / "checkpoint_prior.pt")
        frames = 1024 if dry else P["rl"]["frames_each"]
        env = parent.LongBatch(cache, n=8, seed=seed, frames=frames)
        probs = parent.base.run_ppo_train(net, env)
        assert env.frames == net.count["frames"] == frames
        eps = parent.base.episodes(env, probs, net._values_log)
        torch.save(net, d / "checkpoint_source.pt"); r["checkpoint_sha256"] = parent.sha(d / "checkpoint_source.pt")
        r["train"] = dict(frames=frames, resources=parent.resources(net, [env]),
                          successes=sum(e["success"] for e in env.completed), episodes=len(env.completed))
        sels = {f"{a}_{rep}": run.select_arm(net, cache, seed, a, rep, dry) for a in ("frozen", "adaptive") for rep in range(3)}
        r["selection"] = {k: {q: v for q, v in s.items() if q != "episodes"} for k, s in sels.items()}
        r["episodes_sha256"] = parent.gz_write(d / "episodes.json.gz", dict(train=eps, selection={k: s["episodes"] for k, s in sels.items()}))
        for a in ("frozen", "adaptive"): r[f"{a}_successes"] = sum(sels[f"{a}_{rep}"]["successes"] for rep in range(3))
        r["status"] = "complete"
    except FloatingPointError as exc:
        r.update(status="failed", error=repr(exc), triage="numerical_nonfinite", training_frames_completed=env.frames if env else 0)
    except Exception as exc:
        r.update(status="invalid", error=repr(exc), traceback=traceback.format_exc(), triage="implementation_or_io_invalidates_cohort")
        (d / "result.json").write_text(json.dumps(r, indent=2) + "\n"); raise
    r.update(wall_s=time.perf_counter() - t0, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, device=device,
             gpu_name=torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu")
    (d / "result.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({k: r.get(k) for k in ("arm", "seed", "dry", "status", "wall_s")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["dry", "fit"]); ap.add_argument("arm"); ap.add_argument("seed", type=int)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if a.device.startswith("cuda"):
        assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" and os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1")
        free, _ = torch.cuda.mem_get_info(); assert free > 3 * 2 ** 30, "Preserve desktop memory headroom"
    fit(a.arm, a.seed, a.out, a.command == "dry", a.device)
