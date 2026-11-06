"""MICrONS-perception refit of the E2 matched-label design (Claude, 2026-10-17; protocol
docs/claude_v4n_microns_protocol_20261017.md, frozen with this driver before any fit).

The agent's 144 camera channels are replaced by 144 features of the MICrONS foundation-model twin of mouse visual
cortex (Wang et al. 2025, Nature; cajal/fnn; scan 8-5, 9,941 recorded neurons) responding to the agent's own frame.
Everything else (BC rows/labels/RNG, PPO budget, network, feedback channels, camera-only mask, probes) is the
unchanged input_ablation_claude pipeline. Perceptions: pix (unchanged), twin, rcore (random core; trained
perspective/modulation/readout retained, so a partially pretrained control).
Frame -> twin: the pipeline's uint8 36x64 luminance frame -> 255*(L/255)^(1/2.2) -> 4x nearest upsample (144x256) ->
10 grey + 15 view frames -> mean predicted response over the last 10 frames (scripts/mouse_twin_e1.py).
Every frame any fit or probe can see is computed ONCE (source table + the 16 exposed E2 rooms; 20,820 unique frames).
The swap happens at the POLICY INPUT (v4n_microns_hook_claude_20261017.TwinSwap, keyed by the pipeline's own pooled
camera vector, unique per frame): environments, the pixel-space observer, the source table and the stored BC rows are
byte-identical for every perception. PCA: unique frames of source layouts 0-5 only; 6/7 and rooms are transformed.
  python scripts/v4n_microns_perception_claude_20261017.py views                               (CPU; unique frames)
  CUDA_VISIBLE_DEVICES=g python scripts/v4n_microns_perception_claude_20261017.py raw PERC [PART NPARTS]
  python scripts/v4n_microns_perception_claude_20261017.py derive PERC                          (PCA + lookup)
  python scripts/v4n_microns_perception_claude_20261017.py check
  CUDA_VISIBLE_DEVICES=seed%2 ... fit ARM PERC SEED | ood ARM PERC SEED ; analyse
"""
import hashlib, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve(); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1]))
import torch
import input_ablation_claude as S
import wacv_native_prior_confirm_env as CONFIRM
import wacv_native_behavior_prior_20261012 as RUN
import native_prior_controls_runner_20261012 as CTL  # noqa: F401 (BC path, unmodified)
import v4n_microns_hook_claude_20261017 as H

ROOT = HERE.parents[1]; OUT = ROOT / "data/wacv/microns_perception_20261017"
PERCS = ("pix", "twin", "rcore"); SEEDS = tuple(range(1300, 1310)); NPC = 144; FIT_CFGS = (0, 1, 2, 3, 4, 5)
_orig_pool = CONFIRM.pool_pixels
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ash = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def monitor(px):
    """uint8 [N,36,64] linear luminance -> uint8 [N,144,256] gamma-encoded monitor frames."""
    m = np.rint(255 * (px.astype(np.float32) / 255) ** (1 / 2.2)).astype(np.uint8)
    return m.repeat(4, 1).repeat(4, 2)


def views():
    """All unique uint8 frames: the source table (valid slots) + every pose of the 16 exposed E2 rooms."""
    import wacv_ood_zero_shot_claude as O
    from tartarus_visual_env import render
    from wacv_embodied_source_cache import HEADINGS
    _, cache, _ = S.run.load(); raw = cache.raw; valid = np.isfinite(raw.features[..., 0])
    frames = {}
    for i in np.argwhere(valid): frames.setdefault(raw.images[tuple(i)].tobytes(), raw.images[tuple(i)])
    n_src = len(frames); rc = O.env_source_goal(0, 1, 0, "clean", 65536).env.cache; room = 0
    for cfg in range(100, 116):
        w = rc.worlds[cfg]
        for r, c in np.argwhere(w):
            for h in HEADINGS:
                for ti in range(len(rc.rotations)):
                    img, _, _ = render(rc.view, w, (int(r), int(c)), tuple(h), rc.rotations[ti])
                    px, _ = _orig_pool(img, rc.view, rc.protocol["features"]); frames.setdefault(px.tobytes(), px); room += 1
    X = np.stack(list(frames.values())); np.save(OUT / "frames.npy", X)
    info = dict(unique_frames=len(X), unique_source_frames=n_src, source_slots=int(valid.sum()), room_poses=room, frames_sha256=ash(X),
                valid_by_layout={int(cfg): int(valid[gi].sum()) for cfg, gi in raw.index.items()})
    (OUT / "views.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info))


def raw_responses(perc, part=0, nparts=1):
    import mouse_twin_e1 as E1
    X = np.load(OUT / "frames.npy"); lo, hi = len(X) * part // nparts, len(X) * (part + 1) // nparts
    model = E1.load_twin(random_core=(perc == "rcore"))
    R = np.concatenate([E1.twin_features(model, monitor(X[i:min(i + E1.BATCH, hi)])) for i in range(lo, hi, E1.BATCH)])
    np.save(OUT / f"raw_{perc}_part{part}of{nparts}.npy", R)


def merge_raw(perc, nparts):
    import mouse_twin_e1 as E1
    R = np.concatenate([np.load(OUT / f"raw_{perc}_part{p}of{nparts}.npy") for p in range(nparts)]); X = np.load(OUT / "frames.npy")
    assert len(R) == len(X) and np.isfinite(R).all(); np.save(OUT / f"raw_{perc}.npy", R)
    (OUT / f"raw_{perc}.json").write_text(json.dumps(dict(shape=list(R.shape), raw_sha256=ash(R), frames_sha256=ash(X), parts=nparts,
        twin_params_sha256={f: sha(ROOT / E1.TWIN_DIR / f) for f in ("params_core.pt", f"params_{E1.SESSION}_{E1.SCAN}.pt")}), indent=1))


def derive(perc):
    """PCA on the unique frames of source layouts 0-5 only; lookup keyed by the pipeline's own pooled camera vector."""
    _, cache, _ = S.run.load(); raw = cache.raw; valid = np.isfinite(raw.features[..., 0])
    X = np.load(OUT / "frames.npy"); assert ash(X) == json.loads((OUT / "views.json").read_text())["frames_sha256"]
    gen = json.loads((OUT / "raw_generation.json").read_text()); assert gen["frames_sha256"] == sha(OUT / "frames.npy")
    R = np.load(OUT / f"raw_{perc}_part0of1.npy"); assert R.shape == (len(X), 9941) and np.isfinite(R).all()
    pos = {x.tobytes(): i for i, x in enumerate(X)}; fit_gi = {raw.index[c] for c in FIT_CFGS}
    fit_rows = sorted({pos[raw.images[tuple(i)].tobytes()] for i in np.argwhere(valid) if i[0] in fit_gi})
    A = R[fit_rows].astype(np.float64); mu = A.mean(0); _, s, Vt = np.linalg.svd(A - mu, full_matrices=False); W = Vt[:NPC].T
    sd = ((A - mu) @ W).std(0); assert (sd > 1e-6 * sd[0]).all(), "near-zero PC scale"
    Z = np.clip(((R - mu) @ W) / sd / 3, -1, 1).astype(np.float32)
    K = np.stack([_orig_pool(x / 255.0, raw_view(raw), raw.protocol["features"])[1] for x in X])
    assert len(np.unique(K, axis=0)) == len(K) and not (K == 0).all(1).any(), "pooled key not unique / zero"
    # the pipeline's stored source camera vectors (used by PPO/BC) must all be keys
    kset = {k.tobytes() for k in K}; assert all(raw.features[tuple(i)].tobytes() in kset for i in np.argwhere(valid))
    if not (OUT / "keys.npy").exists(): np.save(OUT / "keys.npy", K)
    assert np.array_equal(np.load(OUT / "keys.npy"), K)
    np.savez(OUT / f"pca_{perc}.npz", mu=mu, W=W, sd=sd); np.save(OUT / f"lookup_{perc}.npy", Z)
    info = dict(fit_layouts=list(FIT_CFGS), fit_valid_slots=int(sum(valid[g].sum() for g in fit_gi)), fit_unique_frames=len(fit_rows),
                var_explained=float((s[:NPC] ** 2).sum() / (s ** 2).sum()), clipped_fraction=float((np.abs(Z) >= 1).mean()),
                keys_sha256=ash(K), lookup_sha256=ash(Z), pca_sha256=sha(OUT / f"pca_{perc}.npz"), raw_sha256=ash(R),
                raw_generation_sha256=sha(OUT / "raw_generation.json"), frames_sha256=ash(X))
    (OUT / f"derive_{perc}.json").write_text(json.dumps(info, indent=1)); print(json.dumps(info))


def raw_view(raw):
    from tartarus_visual_env import View
    return View(**raw.protocol["view"])


def frozen(perc):
    return None if perc == "pix" else json.loads((OUT / f"derive_{perc}.json").read_text())


def setup(perc):
    """Redirect outputs and hashes; for twin/rcore attach the TwinSwap input hook after the camera-only mask.
    Environments, observer, source table and BC data are NOT modified for any perception."""
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    S.OUT = OUT / perc; S.OOD_OUT = OUT / perc / "ood"; S.SEEDS = SEEDS; S.INPUTS = ("no_posterior",)
    base = S.hashes; S.hashes = lambda: dict(base=base(), driver=sha(HERE), hook=sha(HERE.parent / "v4n_microns_hook_claude_20261017.py"),
                                              protocol=sha(ROOT / "docs/claude_v4n_microns_protocol_20261017.md"), perception=perc, frozen=frozen(perc))
    if perc != "pix":
        att = S.attach
        def attach(net, condition):
            net = att(net, condition); net.net.body.register_forward_pre_hook(H.TwinSwap(perc)); return net
        S.attach = attach


def check():
    """Before any fit: keys = pipeline camera vectors for every source slot and rendered room pose; BC rows are keys;
    hook output equals the table and leaves channels 144: unchanged; rcore deterministic; twin != rcore; batch invariance."""
    import mouse_twin_e1 as E1
    from tartarus_visual_env import render
    from wacv_embodied_source_cache import HEADINGS
    import wacv_ood_zero_shot_claude as O
    out = {}; _, cache, _ = S.run.load(); raw = cache.raw; valid = np.isfinite(raw.features[..., 0])
    K = np.load(OUT / "keys.npy"); kset = {k.tobytes(): i for i, k in enumerate(K)}
    out["source_slots_all_keys"] = all(raw.features[tuple(i)].tobytes() in kset for i in np.argwhere(valid))
    rc = O.env_source_goal(0, 1, 0, "clean", 65536).env.cache; miss = n = 0
    for cfg in range(100, 116):
        w = rc.worlds[cfg]
        for r, c in np.argwhere(w)[::3]:
            for h in HEADINGS:
                n += 1; ti = n % len(rc.rotations)
                miss += rc.frame(cfg, (int(r), int(c)), tuple(h), ti).tobytes() not in kset
    out["room_poses_checked"], out["room_poses_missing"] = n, miss
    for sp in ("human", "rat"):
        meta = next(x for x in json.loads((RUN.BASE / "data_manifest.json").read_text())["datasets"] if x["species"] == sp)
        with np.load(ROOT / meta["path"]) as a: B = a["X"]
        out[f"bc_{sp}_rows_all_keys"] = all(B[j, k, :144].tobytes() in kset for j, k in np.ndindex(B.shape[:2]))
    for perc in ("twin", "rcore"):
        hk = H.TwinSwap(perc); Z = np.load(OUT / f"lookup_{perc}.npy"); x = np.concatenate([K[:64], np.random.default_rng(0).normal(size=(64, 108)).astype(np.float32)], 1)
        y = hk(None, (torch.as_tensor(x)[None],))[0][0].numpy()
        out[f"{perc}_hook_matches_table"] = bool(np.array_equal(y[:, :144], Z[:64]) and np.array_equal(y[:, 144:], x[:, 144:]))
        hk2 = pickle_roundtrip(hk); out[f"{perc}_hook_pickles_without_table"] = hk2._map is None
    a, b = E1.load_twin(True).state_dict(), E1.load_twin(True).state_dict()
    out["rcore_deterministic"] = all(torch.equal(a[k], b[k]) for k in a)
    out["twin_vs_rcore_differ"] = bool(np.abs(np.load(OUT / "lookup_twin.npy") - np.load(OUT / "lookup_rcore.npy")).max() > 1e-3)
    X = np.load(OUT / "frames.npy")[:96]; m = E1.load_twin(); R = np.load(OUT / "raw_twin_part0of1.npy", mmap_mode="r")
    r1, r2 = E1.twin_features(m, monitor(X[:64])), E1.twin_features(m, monitor(X[32:96]))
    out["twin_batch_order_invariant_maxabs"] = float(np.abs(r1[32:] - r2[:32]).max()); out["twin_regen_vs_stored_maxabs"] = float(np.abs(r1 - R[:64]).max())
    out["ok"] = bool(out["source_slots_all_keys"] and out["room_poses_missing"] == 0 and out["bc_human_rows_all_keys"] and out["bc_rat_rows_all_keys"]
                     and all(out[f"{p}_hook_matches_table"] and out[f"{p}_hook_pickles_without_table"] for p in ("twin", "rcore"))
                     and out["rcore_deterministic"] and out["twin_vs_rcore_differ"] and out["twin_batch_order_invariant_maxabs"] < 1e-5
                     and out["twin_regen_vs_stored_maxabs"] < 1e-4)
    (OUT / "check.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)); assert out["ok"]


def pickle_roundtrip(o):
    import pickle; return pickle.loads(pickle.dumps(o))


def ood(arm, perc, seed):
    """The 16 exposed E2 rooms x 30 frozen trials via the unchanged F.probe (as input_ablation_claude.ood), one case."""
    import gzip, wacv_ood_zero_shot_claude as O
    setup(perc); F = O.F; dry = os.environ.get("DRY") == "1"; grp = "dry" if dry else "cohort"
    F.SEEDS30 = S.OUT / grp; F.environment = O.env_source_goal
    method = next(x for x in json.loads(F.study.MANIFEST.read_text())["methods"] if x["id"] == "ppo")
    c = S.case(arm, "no_posterior"); src = S.OUT / grp / c / f"seed{seed}"
    res = json.loads((src / "result.json").read_text()); assert res["status"] == "complete" and res["hashes"] == S.hashes()
    ck = S.parent.sha(src / "checkpoint_source.pt"); assert ck == res["checkpoint_sha256"]
    for p in ((0,) if dry else range(8)):
        for st in ("A", "B"):
            f = S.OOD_OUT / grp / c / f"seed{seed}" / f"pair{p}_{st}.json"
            if f.exists(): assert json.loads(f.read_text())["checkpoint_sha256"] == ck; continue
            F.study.source_model = lambda x, c=c: F.source_checkpoint(c, x); m = F.study.build(method, seed)
            hk = S.mask_of(F.study.source_model(seed)); assert hk is not None and hk.condition == "no_posterior"
            m.net.body.register_forward_pre_hook(S.InputMask(hk.condition))
            sw = H.swap_of(F.study.source_model(seed)); assert (sw is None) == (perc == "pix") and (sw is None or sw.perc == perc)
            if sw is not None: sw = H.TwinSwap(perc); m.net.body.register_forward_pre_hook(sw)
            rec = F.probe(m, p, st, seed, "clean"); f.parent.mkdir(parents=True, exist_ok=True)
            full = f.with_suffix(".full.json.gz"); full.write_bytes(gzip.compress(json.dumps(rec, default=float).encode()))
            f.write_text(json.dumps(dict(arm=c, perception=perc, seed=seed, pair=p, room=st, goals=rec["goals"], trials=rec["trials"],
                                         checkpoint_sha256=ck, full_record_sha256=S.parent.sha(full), twin_hook_attached=sw is not None)) + "\n")


E3_PERCS, E3_METHODS, E3_PAIR = ("pix", "twin"), ("ppo", "ppo_replay"), 1


def e3_setup(perc):
    """Unchanged continual runner (wacv_continual_factorial_claude.fit) from this cohort's camera-only checkpoints;
    InputMask and (twin) TwinSwap re-attached after build, as in ood()."""
    import wacv_continual_factorial_claude as F
    setup(perc); F.BASE = OUT / perc / "e3"; F.SEEDS30 = S.OUT / ("dry" if os.environ.get("DRY") == "1" else "cohort"); F.ARMS = tuple(S.case(a, "no_posterior") for a in S.ARMS)
    F.binding = lambda: dict(cohort=S.hashes(), factorial=sha(F.__file__), development=F.study.binding())
    build0 = F.study.build
    def build(method, seed):
        m = build0(method, seed); src = F.study.source_model(seed); hk = S.mask_of(src)
        assert hk is not None and hk.condition == "no_posterior" and (H.swap_of(src) is None) == (perc == "pix")
        m.net.body.register_forward_pre_hook(S.InputMask("no_posterior"))
        if perc != "pix": m.net.body.register_forward_pre_hook(H.TwinSwap(perc))
        return m
    F.study.build = build
    return F


def e3(arm, perc, method, seed):
    F = e3_setup(perc); F.fit(S.case(arm, "no_posterior"), method, seed, E3_PAIR, "clean", dry=(os.environ.get("DRY") == "1"))


def analyse_e3():
    """Secondary (exploratory): forgetting F = A successes after A-learning minus after B-learning (frozen probes);
    replay effect = F_ppo - F_replay per seed (mean over the 3 arms); exact sign-flip per perception."""
    out = {}
    for perc in E3_PERCS:
        Fg = {}
        for meth in E3_METHODS:
            for a in S.ARMS:
                for s in SEEDS:
                    r = json.loads((OUT / perc / "e3" / "cohort" / S.case(a, "no_posterior") / meth / f"pair{E3_PAIR}" / "clean" / f"seed{s}" / "result.json").read_text())
                    Fg[(meth, a, s)] = r["probes"]["A_after_A_learning"]["goals"] - r["probes"]["A_after_B_learning"]["goals"]
        d = np.array([np.mean([Fg[("ppo", a, s)] - Fg[("ppo_replay", a, s)] for a in S.ARMS]) for s in SEEDS])
        out[perc] = dict(forgetting_ppo={a: float(np.mean([Fg[("ppo", a, s)] for s in SEEDS])) for a in S.ARMS},
                         forgetting_replay={a: float(np.mean([Fg[("ppo_replay", a, s)] for s in SEEDS])) for a in S.ARMS},
                         replay_reduction_mean=float(d.mean()), replay_reduction_signflip_p=S.exact_signflip(d), per_seed=d.tolist())
    (OUT / "analysis_e3.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def analyse():
    """Primary: exact sign-flip on Delta = human_bc - human_legal_uniform (n=10 paired seeds), Holm over 3 perceptions.
    Secondary: t CI / t p with Holm. twin - rcore is descriptive only."""
    from scipy import stats
    out = {}
    for perc in PERCS:
        sc, comp = {}, {}
        for a in S.ARMS:
            c = S.case(a, "no_posterior"); base = OUT / perc / "ood" / "cohort" / c
            sc[a] = [float(np.mean([json.loads((base / f"seed{s}" / f"pair{p}_{st}.json").read_text())["goals"] for p in range(8) for st in "AB"])) for s in SEEDS]
            comp[a] = [json.loads((OUT / perc / "cohort" / c / f"seed{s}" / "result.json").read_text())["source_competence"]["successes"] for s in SEEDS]
        d = np.subtract(sc["human_bc"], sc["human_legal_uniform"]); se = d.std(ddof=1) / np.sqrt(len(d))
        tp, ci = (float(stats.ttest_1samp(d, 0).pvalue), [float(x) for x in stats.t.interval(.95, len(d) - 1, d.mean(), se)]) if se > 0 \
            else ((1.0 if d.mean() == 0 else 0.0), [float(d.mean())] * 2)        # zero variance: deterministic, kept
        out[perc] = dict(means={a: float(np.mean(v)) for a, v in sc.items()}, delta_mean=float(d.mean()), per_seed_delta=d.tolist(),
                         signflip_p=S.exact_signflip(d), t_p=tp, t_ci95=ci, zero_variance=bool(se == 0),
                         source_competence=comp, qualified=all(sum(x >= 75 for x in comp[a]) >= 8 for a in S.ARMS))
    for key in ("signflip_p", "t_p"):
        ps = [out[p][key] for p in PERCS]; run = 0.0
        for r, i in enumerate(np.argsort(ps)): run = max(run, min(1.0, (3 - r) * ps[i])); out[PERCS[i]][f"holm_{key}"] = run
    out["descriptive_twin_minus_rcore"] = float(out["twin"]["delta_mean"] - out["rcore"]["delta_mean"])
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]; OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "views": views()
    elif cmd == "raw": raw_responses(args[0], *(int(x) for x in args[1:3]))
    elif cmd == "merge": merge_raw(args[0], int(args[1]))
    elif cmd == "derive": derive(args[0])
    elif cmd == "check": check()
    elif cmd == "fit": setup(args[1]); S.fit(args[0], "no_posterior", int(args[2]), dry=(os.environ.get("DRY") == "1"))
    elif cmd == "ood": ood(args[0], args[1], int(args[2]))
    elif cmd == "analyse": analyse()
    elif cmd == "e3": e3(args[0], args[1], args[2], int(args[3]))
    elif cmd == "analyse_e3": analyse_e3()
    elif cmd == "go":                       # CORE GO only (90 fits); E3 needs its own audit and GO, written separately later
        note = args[0]; assert json.loads((OUT / "check.json").read_text())["ok"]
        for perc in PERCS:
            import importlib; importlib.reload(S); setup(perc); (OUT / perc).mkdir(exist_ok=True)
            (OUT / perc / "GO.json").write_text(json.dumps(dict(status="GO", hashes=S.hashes(), reviewer_note=note,
                                                               check_sha256=sha(OUT / "check.json")), indent=1))
        (OUT / "GO.json").write_text(json.dumps(dict(status="GO", reviewer_note=note, perceptions=list(PERCS)), indent=1))
    else: raise SystemExit(f"unknown {cmd}")
