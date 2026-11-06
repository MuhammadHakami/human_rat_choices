"""ALL-optimal soft-target expert control (PREPARATION; no fit without a bound GO). Codex source-only audit 2026-10-13:
44.07% of the 6,000 human prior states have >1 equally short move and the original expert always takes the first in
MOVES order (direction counts 1565/2837/543/1055 vs uniform-over-optimal expectation 1031/2056.5/955/1957.5).
Teacher: q(a | recorded human state) = 1 / #optimal for every legal move that decreases the BFS distance to the goal by 1
(registered offline labeller geometry, as for the original expert; no online goal/geometry handle reaches the agent).
Same 6,000 rows x 3 views, same BC loop as the registered controls (RNG [20261012, 601, seed], batches, Adam, clip 0.5,
critic frozen, 20 epochs = 360,000 presentations; cross-entropy with soft targets), then the unchanged 524,288-frame PPO
and the input-ablation source probe. Seeds 1200-1209, GPU seed % 2, i.e. paired with the input study's FULL human arms.
Disclose: soft labels carry more information per row than hard recorded labels (different target entropy).
  python scripts/alloptimal_expert_claude.py check | dry | drygpu | fit SEED (GO) | calibrate (root source audit) | ood W N (OOD GO) | analyse (root audits)
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_prior_surrogate_v2_20261012 as sur
import input_ablation_claude as M

P, run, parent = M.P, M.run, M.parent
OUT = M.ROOT / "data/wacv/alloptimal_expert"; ARM = "human_alloptimal"; SEEDS = M.SEEDS


def targets_reference():
    """Reference (slow: surrogate_v2.features re-decompresses the npz per row); kept for the bit-exact check only."""
    feats = sur.features("human", run.load()[1], json.loads((M.ROOT / "data/unified/tartarus_trajectories.json").read_text()))
    opt = feats["legal"] & (feats["prog"] == 1); n = opt.sum(1); assert (n >= 1).all()
    return opt / n[:, None], feats["identity"], n


def targets():
    """Same optimal legal support as surrogate_v2.features (registered native cells, layouts 1-5 train individuals, BFS
    goal distances), with the npz loaded ONCE; no persistence feature (unused by this teacher)."""
    cache = run.load()[1]; native = json.loads((M.ROOT / "data/unified/tartarus_trajectories.json").read_text())
    with np.load(sur.D / "human_prior.npz") as z: ident = z["identity"].copy()
    par = json.loads(sur.PARENT.read_text()); subj, cfgs = set(par["native_subjects"]["human"]), set(par["native_configurations"])
    cells = {(t["subject"], t["configuration"], t["trial"]): [tuple(c) for c in t["cells"]] for t in native["trials"]
             if t["species"] == "human" and t["subject"] in subj and t["configuration"] in cfgs}
    dist = {cfg: sur.prior.goal_distances(cache.geometry(cfg)[0], cache.goal) for cfg in range(1, 6)}
    opt = np.zeros((len(ident), 4), bool)
    for r, (s, cfg, t, k) in enumerate(ident):
        here = cells[(s, cfg, t)][k]; d = dist[cfg]
        for a, (dr, dc) in enumerate(sur.MOVES):
            nxt = (here[0] + dr, here[1] + dc)
            if nxt in d and d[here] - d[nxt] == 1: opt[r, a] = True
    n = opt.sum(1); assert (n >= 1).all()
    return opt / n[:, None], ident, n


def pretrain_soft(net, seed, Pr, dry):
    meta = next(r for r in json.loads((run.BASE / "data_manifest.json").read_text())["datasets"] if r["species"] == "human")
    with np.load(M.ROOT / meta["path"]) as a: assert parent.sha(M.ROOT / meta["path"]) == meta["sha256"]; X, ident = a["X"], a["identity"]
    q, qid, _ = targets(); assert np.array_equal(ident, qid), "targets must be at exactly the prior rows"
    dev = next(net.net.parameters()).device
    x = torch.as_tensor(X.reshape(-1, 252), device=dev); y = torch.as_tensor(np.repeat(q, 3, axis=0), dtype=torch.float32, device=dev)
    rng = np.random.default_rng([20261012, 601, seed]); opt = torch.optim.Adam(net.net.parameters(), lr=Pr["bc"]["learning_rate"], eps=1e-5)
    critic = [p.detach().clone() for p in net.net.v.parameters()]; pres = upd = 0; stats = []
    for _ in range(1 if dry else Pr["bc"]["epochs"]):
        order = rng.permutation(len(x)); order = order[:512] if dry else order; tot = 0.
        for st in range(0, len(order), Pr["bc"]["batch_size"]):
            ii = torch.as_tensor(order[st:st + Pr["bc"]["batch_size"]], device=dev)
            loss = F.cross_entropy(net.net.pi(net.net.body(x[ii])), y[ii])          # soft-target cross-entropy
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.net.parameters(), Pr["bc"]["gradient_clip"]); opt.step()
            pres += len(ii); upd += 1; tot += float(loss.detach()) * len(ii)
        stats.append(tot / len(order))
    assert all(torch.equal(a, b) for a, b in zip(critic, net.net.v.parameters()))
    net.opt = torch.optim.Adam(net.net.parameters(), lr=.001, eps=1e-5)
    return dict(unique_native_moves=6000, labeled_presentations=pres, updates=upd, cross_entropy_per_epoch=stats, critic_weights_unchanged=True, label_arm=ARM, soft_targets=True)


PREP = OUT / "PROTOCOL_PREPARATION.json"; CAL = OUT / "calibrated"; OOD_OUT = M.ROOT / "data/audit/wacv_ood_alloptimal_claude"


def hashes():
    meta = next(r for r in json.loads((run.BASE / "data_manifest.json").read_text())["datasets"] if r["species"] == "human")
    return dict(wrapper=parent.sha(__file__), input_ablation=M.hashes(), surrogate_v2=parent.sha(sur.__file__),
                native=parent.sha(M.ROOT / "data/unified/tartarus_trajectories.json"), human_prior=parent.sha(M.ROOT / meta["path"]),
                data_manifest=parent.sha(run.BASE / "data_manifest.json"), temperature=parent.sha(M.ROOT / "scripts/temperature_expert_control_claude.py"), preparation=parent.sha(PREP) if PREP.exists() else None)


def fit(seed, dry, gpu_dry=False):
    import gzip
    assert dry or seed in SEEDS; device = "cuda:0" if (gpu_dry or not dry) else "cpu"
    if not dry:
        go = json.loads((OUT / "GO.json").read_text()); assert go["status"] == "GO" and go["hashes"] == hashes()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(seed % 2) and ("4090" if seed % 2 == 0 else "3090") in torch.cuda.get_device_name(0)
    Pr, cache, L = run.load(); d = OUT / (("dry_gpu" if gpu_dry else "dry") if dry else "cohort") / f"{ARM}__full" / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    if (d / "result.json").exists():
        old = json.loads((d / "result.json").read_text())
        assert old["status"] == "complete" and old["hashes"] == hashes() and parent.sha(d / "checkpoint_source.pt") == old["checkpoint_sha256"] \
            and parent.sha(d / "source_probe.json.gz") == old["source_probe_sha256"], f"invalid existing {d}"; return
    with (d / "started.json").open("x") as f: json.dump(dict(hashes=hashes()), f)
    t0 = time.perf_counter(); r = dict(arm=ARM, input="full", seed=seed, dry=dry, hashes=hashes(), status="running")
    try:
        net = run.build(seed, device, L); r["bc"] = pretrain_soft(net, seed, Pr, dry)
        torch.save(net, d / "checkpoint_prior.pt"); r["prior_checkpoint_sha256"] = parent.sha(d / "checkpoint_prior.pt")
        frames = 1024 if dry else Pr["rl"]["frames_each"]; env = parent.LongBatch(cache, n=8, seed=seed, frames=frames); parent.base.run_ppo_train(net, env)
        assert env.frames == net.count["frames"] == frames; torch.save(net, d / "checkpoint_source.pt")
        summary, records = M.source_probe(net, cache, seed, dry)
        (d / "source_probe.json.gz").write_bytes(gzip.compress(json.dumps(records, default=float).encode()))
        r.update(checkpoint_sha256=parent.sha(d / "checkpoint_source.pt"), source_competence=summary, source_probe_sha256=parent.sha(d / "source_probe.json.gz"),
                 train=dict(frames=frames, successes=sum(e["success"] for e in env.completed)), status="complete", wall_s=time.perf_counter() - t0,
                 gpu_name=torch.cuda.get_device_name(0) if device != "cpu" else "cpu")
    except Exception as e:
        import traceback; r.update(status="failed", error=repr(e), traceback=traceback.format_exc()); (d / "result.json").write_text(json.dumps(r, indent=2, default=str) + "\n"); raise
    (d / "result.json").write_text(json.dumps(r, indent=2) + "\n"); print(json.dumps({k: r[k] for k in ("arm", "seed", "status", "wall_s")}), flush=True)


def calibrate():
    """Source-only global temperature (verified temperature_expert_control math, bounds [.05, 20], 60 iterations): per seed,
    T so the all-optimal actor's mean entropy on the 18,000 human source views equals the input study's FULL human_bc of
    the SAME seed/GPU. Gated on the source GO, the 10-teacher root audit and the 90-case input-study source audit; every
    checkpoint verified against the audited digests; GPU parity asserted; bound failures retained and NOT called entropy
    matched; nothing is ever overwritten."""
    import temperature_expert_control_claude as T
    assert T.BOUNDS == (.05, 20.) and T.ITERS == 60
    go = json.loads((OUT / "GO.json").read_text()); assert go["status"] == "GO" and go["hashes"] == hashes()
    ta = json.loads((OUT / "root_source_audit.json").read_text()); assert ta["ok"] and ta["cases"] == 10
    ia = json.loads((M.OUT / "root_source_audit.json").read_text()); assert ia["ok"] and ia["cases"] == 90
    assert not (CAL / "calibration_lock.json").exists(), "never overwrite a calibration"
    X = T.views("human"); assert X.shape == (18000, 252); CAL.mkdir(parents=True, exist_ok=True); rows = {}
    for s in SEEDS:
        tdir, hdir = OUT / "cohort" / f"{ARM}__full" / f"seed{s}", M.OUT / "cohort" / "human_bc__full" / f"seed{s}"
        td, hd = parent.sha(tdir / "checkpoint_source.pt"), parent.sha(hdir / "checkpoint_source.pt")
        assert td == ta["checkpoints"][f"{ARM}__full/seed{s}"] and hd == ia["checkpoints"][f"human_bc__full/seed{s}"]
        gpu = "4090" if s % 2 == 0 else "3090"
        assert all(gpu in json.loads((x / "result.json").read_text())["gpu_name"] for x in (tdir, hdir))
        e = torch.load(tdir / "checkpoint_source.pt", map_location="cpu", weights_only=False); h = torch.load(hdir / "checkpoint_source.pt", map_location="cpu", weights_only=False)
        ze, zh = T.logits(e, X), T.logits(h, X); target = T.entropy(zh, 1.); t, status, got = T.solve(ze, target); m = T.scaled(e, t)
        se, sm = e.net.state_dict(), m.net.state_dict(); assert se.keys() == sm.keys() and all(torch.equal(se[k], sm[k]) for k in se if k not in ("pi.weight", "pi.bias"))
        d = CAL / f"{ARM}_T__full" / f"seed{s}"; d.mkdir(parents=True, exist_ok=False); torch.save(m, d / "checkpoint_source.pt")
        rows[f"seed{s}"] = dict(status="complete", seed=s, T=t, solve_status=status, entropy_matched=status == "solved", target_entropy=target,
                                achieved_entropy=got, teacher_entropy_T1=T.entropy(ze, 1.), views=len(X), teacher_source_sha256=td, human_source_sha256=hd,
                                non_pi_parameters_unchanged=True, gpu=gpu, checkpoint_sha256=parent.sha(d / "checkpoint_source.pt"))
        with (d / "result.json").open("x") as f: f.write(json.dumps(rows[f"seed{s}"], indent=1) + "\n")
    with (CAL / "calibration_lock.json").open("x") as f:
        f.write(json.dumps(dict(rows=rows, checkpoints={k: v["checkpoint_sha256"] for k, v in rows.items()}, hashes=hashes(), bounds=[.05, 20.], iterations=60), indent=1) + "\n")


def ood(worker, n):
    """16 E2 rooms, goal (3,3), 30 frozen trials; raw + calibrated teachers; full F.probe records; gated on OOD GO."""
    import gzip, wacv_ood_zero_shot_claude as O
    F = O.F; go = json.loads((OOD_OUT / "GO.json").read_text()); assert go["status"] == "GO" and go["hashes"] == hashes()
    assert go["calibration_lock_sha256"] == parent.sha(CAL / "calibration_lock.json")
    ca = json.loads((CAL / "root_calibration_audit.json").read_text()); assert ca["ok"] and go["calibration_audit_sha256"] == parent.sha(CAL / "root_calibration_audit.json")
    assert len(go["teacher_checkpoints"]) == 10 and len(go["calibrated_checkpoints"]) == 10
    F.environment = O.env_source_goal; method = next(x for x in json.loads(F.study.MANIFEST.read_text())["methods"] if x["id"] == "ppo")
    gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])
    jobs = [(root, c, s, p, st) for s in SEEDS if s % 2 == gpu for root, c in ((OUT / "cohort", f"{ARM}__full"), (CAL, f"{ARM}_T__full"))
            for p in range(8) for st in ("A", "B")][worker::n]
    for root, c, s, p, st in jobs:
        F.SEEDS30 = root; sha = parent.sha(root / c / f"seed{s}" / "checkpoint_source.pt")
        assert sha == (go["calibrated_checkpoints"] if root == CAL else go["teacher_checkpoints"])[f"seed{s}"]
        f = OOD_OUT / "cohort" / c / f"seed{s}" / f"pair{p}_{st}.json"; full = f.with_suffix(".full.json.gz")
        if f.exists():
            old = json.loads(f.read_text()); assert old["checkpoint_sha256"] == sha and parent.sha(full) == old["full_record_sha256"], f"invalid existing {f}"; continue
        F.study.source_model = lambda x, c=c: F.source_checkpoint(c, x); m = F.study.build(method, s); rec = F.probe(m, p, st, s, "clean")
        f.parent.mkdir(parents=True, exist_ok=True); full.write_bytes(gzip.compress(json.dumps(rec, default=float).encode()))
        f.write_text(json.dumps(dict(arm=c, seed=s, pair=p, room=st, goals=rec["goals"], trials=rec["trials"], checkpoint_sha256=sha,
                                     full_record_sha256=parent.sha(full), gpu_name=torch.cuda.get_device_name(0))) + "\n")


def analyse():
    """Primary: human_bc(full, input study) - T-calibrated all-optimal; secondary: - raw; paired over seeds 1200-1209."""
    from analyse_seeds30_claude import holm
    oa = json.loads((OOD_OUT / "root_audit.json").read_text()); assert oa["ok"] and oa["cases"] == 320
    go = json.loads((OOD_OUT / "GO.json").read_text()); assert go["status"] == "GO" and go["hashes"] == hashes()
    for c, key in ((f"{ARM}__full", "teacher_checkpoints"), (f"{ARM}_T__full", "calibrated_checkpoints")):
        for s in SEEDS:
            for p in range(8):
                for st in ("A", "B"):
                    f = OOD_OUT / "cohort" / c / f"seed{s}" / f"pair{p}_{st}.json"; r = json.loads(f.read_text())
                    assert r["checkpoint_sha256"] == go[key][f"seed{s}"] and parent.sha(f.with_suffix(".full.json.gz")) == r["full_record_sha256"] and r["trials"] == 30
    ia = json.loads((M.OOD_OUT / "root_audit.json").read_text()); assert ia["ok"] and ia["cases"] == 1440
    from analyse_ood_followup_claude import ood as score
    H = score(M.OOD_OUT, "human_bc__full", list(SEEDS)); R = score(OOD_OUT, f"{ARM}__full", list(SEEDS)); C = score(OOD_OUT, f"{ARM}_T__full", list(SEEDS))
    test = lambda d: dict(mean=float(np.mean(d)), ci95=None, exact_signflip_p=M.exact_signflip(d), n=len(d))
    out = dict(primary=test([H[s] - C[s] for s in SEEDS]), secondary_raw=test([H[s] - R[s] for s in SEEDS]), secondary_raw_status="descriptive (registered); primary unadjusted as registered",
               means=dict(human_bc=float(np.mean(list(H.values()))), alloptimal_raw=float(np.mean(list(R.values()))), alloptimal_T=float(np.mean(list(C.values())))),
               note="exposed rooms; equal-optimal soft teacher vs recorded human choices; soft labels carry more information per row")
    from scipy import stats
    for k in ("primary", "secondary_raw"):
        d = [H[s] - (C if k == "primary" else R)[s] for s in SEEDS]; se = np.std(d, ddof=1) / np.sqrt(len(d))
        out[k].update(t_p=float(stats.ttest_1samp(d, 0).pvalue) if se > 0 else (1.0 if np.mean(d) == 0 else 0.0),
                      ci95=[float(np.mean(d) - stats.t.ppf(.975, len(d) - 1) * se), float(np.mean(d) + stats.t.ppf(.975, len(d) - 1) * se)])
    (OOD_OUT / "analysis_alloptimal.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out)[:600])


def check():
    q, ident, n = targets(); qr, ir, nr = targets_reference()
    assert np.array_equal(q, qr) and np.array_equal(ident, ir) and np.array_equal(n, nr), "fast targets must equal the reference bit-exactly"
    print("bit-exact vs surrogate_v2 reference: rows", len(q), "q sha", __import__("hashlib").sha256(q.tobytes()).hexdigest()[:16])
    assert np.allclose(q.sum(1), 1) and abs((n > 1).mean() - .440667) < 1e-4, (n > 1).mean()
    print("check ok: rows", len(q), "tied fraction %.4f" % (n > 1).mean(), "expected direction counts", np.round(q.sum(0), 1).tolist())


if __name__ == "__main__":
    import importlib; Mod = importlib.import_module("alloptimal_expert_claude")
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    c = sys.argv[1]
    if c == "check": Mod.check()
    elif c == "dry": Mod.fit(0, True)
    elif c == "drygpu": Mod.fit(int(os.environ["CUDA_VISIBLE_DEVICES"]), True, gpu_dry=True)
    elif c == "calibrate": Mod.calibrate()
    elif c == "ood": Mod.ood(int(sys.argv[2]), int(sys.argv[3]))
    elif c == "analyse": Mod.analyse()
    elif c == "fit": Mod.fit(int(sys.argv[2]), False)
