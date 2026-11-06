"""30-seed paired replication of the native-prior study (Claude, GPU 1; author 2026-10-12: "use GPU 1 only and continue
don't wait for codex"). Same build/BC/PPO/selection code as the audited cohorts, imported read-only:
  rat_bc, rat_sham, cold_ppo  -> wacv_native_behavior_prior_20261012.pretrain
  rat_legal_uniform, human_legal_uniform -> native_prior_controls_runner_20261012.pretrain (registered label file)
Only the seeds (fresh, disjoint from 60-62) and the output directory differ. Pre-registration is written by `prereg`
before any fit and its hash is bound into every result.
  python scripts/prior_seeds30_claude.py prereg
  python scripts/prior_seeds30_claude.py fit ARM SEED       (or `dry ARM 0`)
  python scripts/prior_seeds30_claude.py queue WORKER NWORKERS
"""
import argparse, json, os, resource, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wacv_native_behavior_prior_20261012 as run
import native_prior_controls_runner_20261012 as ctl
import prior_seed_power_claude as power

ROOT, parent = run.ROOT, run.parent
DeviceFFAgent = run.DeviceFFAgent
OUT = ROOT / "data/wacv/seeds30"
PREREG = OUT / "preregistration.json"
ARMS = ("rat_bc", "rat_legal_uniform", "cold_ppo", "rat_sham", "human_bc", "human_legal_uniform")
SEEDS = tuple(range(1000, 1030))


def hashes():
    return dict(source=run.hashes(), controls_runner=parent.sha(ctl.__file__), analysis=parent.sha(power.__file__),
                runner=parent.sha(__file__))


def prereg():
    assert not PREREG.exists(), "never overwrite the pre-registration"
    OUT.mkdir(parents=True, exist_ok=True)
    PREREG.write_text(json.dumps(dict(
        written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), hashes=hashes(), arms=ARMS, seeds=SEEDS,
        order="seed-major: all arms of seed s before seed s+1 (pairs complete together if stopped early)",
        outcome="adaptive_successes /60 per seed (3 reps x 20 trials on layouts 6-7), as in the audited cohorts",
        primary=dict(contrast="rat_bc - rat_legal_uniform", tests="paired t, exact/1e5 sign-flip, TOST +-10/60",
                     reading="equivalent if TOST p<.05; rat-choice advantage if t and sign-flip p<.05 and mean>0; else inconclusive"),
        secondary=["rat_bc - cold_ppo", "rat_bc - rat_sham", "rat_legal_uniform - cold_ppo", "human_bc - human_legal_uniform",
                   "solved seeds (>=40/60) per arm, Fisher exact vs rat_bc"],
        multiplicity="Holm over the 4 secondary paired contrasts; primary unadjusted",
        device="GPU 1 (RTX 3090) for every arm; audited 3-seed cohort ran on the 4090, so no cross-cohort pooling",
        exclusions="none; failed/nonfinite fits are reported as failures and their seed pairs listed"), indent=1) + "\n")
    print(PREREG.read_text())


def pretrain(net, arm, seed, P, dry):
    if "legal_uniform" in arm: return ctl.pretrain(net, arm, seed, P, dry)
    return run.pretrain(net, arm, seed, P, dry)


def fit(arm, seed, dry, device="cuda:0"):
    P, cache, L = run.load(); assert arm in ARMS and (dry or seed in SEEDS)
    pre = parent.sha(PREREG) if not dry else None
    if not dry: assert json.loads(PREREG.read_text())["hashes"] == hashes(), "code changed after pre-registration"
    d = (OUT / ("dry" if dry else "cohort")) / arm / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    if (d / "result.json").exists() and json.loads((d / "result.json").read_text())["status"] == "complete" and not dry: return
    r = dict(arm=arm, seed=seed, dry=dry, hashes=hashes(), preregistration_sha256=pre, status="running"); t0 = time.perf_counter(); env = None
    try:
        net = run.build(seed, device, L); r["bc"] = pretrain(net, arm, seed, P, dry)
        frames = 1024 if dry else P["rl"]["frames_each"]
        env = parent.LongBatch(cache, n=8, seed=seed, frames=frames)
        parent.base.run_ppo_train(net, env); assert env.frames == net.count["frames"] == frames
        torch.save(net, d / "checkpoint_source.pt"); r["checkpoint_sha256"] = parent.sha(d / "checkpoint_source.pt")
        r["train"] = dict(frames=frames, successes=sum(e["success"] for e in env.completed), episodes=len(env.completed))
        sels = {f"{a}_{rep}": run.select_arm(net, cache, seed, a, rep, dry) for a in ("frozen", "adaptive") for rep in range(3)}
        r["selection"] = {k: {q: v for q, v in s.items() if q != "episodes"} for k, s in sels.items()}
        r["episodes_sha256"] = parent.gz_write(d / "episodes.json.gz", dict(selection={k: s["episodes"] for k, s in sels.items()}))
        for a in ("frozen", "adaptive"): r[f"{a}_successes"] = sum(sels[f"{a}_{rep}"]["successes"] for rep in range(3))
        r["status"] = "complete"
    except FloatingPointError as exc:
        r.update(status="failed", error=repr(exc), triage="numerical_nonfinite", training_frames_completed=env.frames if env else 0)
    except Exception as exc:
        r.update(status="invalid", error=repr(exc), traceback=traceback.format_exc())
        (d / "result.json").write_text(json.dumps(r, indent=2) + "\n"); raise
    r.update(wall_s=time.perf_counter() - t0, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, device=device,
             gpu_name=torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu")
    (d / "result.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({k: r.get(k) for k in ("arm", "seed", "status", "wall_s")}), flush=True)   # no score printed


def queue(worker, nworkers):
    jobs = [(a, s) for s in SEEDS for a in ARMS][worker::nworkers]
    for a, s in jobs: fit(a, s, False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=["prereg", "dry", "fit", "queue"])
    ap.add_argument("a", nargs="?"); ap.add_argument("b", nargs="?", type=int); a = ap.parse_args()
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if a.command != "prereg":
        assert os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
        assert "3090" in torch.cuda.get_device_name(0)
    if a.command == "prereg": prereg()
    elif a.command == "queue": queue(int(a.a), a.b)
    else: fit(a.a, a.b, a.command == "dry")
