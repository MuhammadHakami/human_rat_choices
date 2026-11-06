"""Retrained input ablation of the E2 human-choice advantage (Codex/Claude decision 2026-10-13; PREPARATION, no fit
without a bound GO). Arms {human_bc, human_legal_uniform, cold_ppo} x inputs {full, no_pixels, no_posterior} x NEW seeds
1200-1209 = 90 source fits; all 9 cases of a seed on GPU (seed % 2). Identical recipe/dose to E1 (BC 6000 moves x 3 views
x 20 epochs = 360,000 presentations; 524,288 PPO frames; same source curriculum/architecture), via the unchanged
wacv_native_behavior_prior_20261012 / prior_seeds30_claude / controls code.
Masking: a picklable forward-pre-hook on the policy's shared body zeroes input channels for BC, PPO training and every
frozen probe (the hook is saved in the checkpoint and re-attached when the OOD runner rebuilds the learner).
  no_pixels    zero 0:144   (policy loses the DIRECT camera; the posterior is still visually derived by the observer)
  no_posterior zero 150:252 (policy loses the calibrated place posterior + coordinates; pixels kept)
  public feedback 144:150 (contact, reward, last action) is always kept. The observer runs in every condition, so no
  observer-cost saving is claimed.
Source competence: frozen policy on TRAINING layouts 1-5, all known starts x 3 rotations (150 trials); representation-
qualified iff >= 75/150 in >= 8/10 seeds for EACH arm of that input. All 1,440 OOD probes run regardless (stratifies
interpretation only; no model/hyperparameter selection). The 16 E2 rooms are exposed: a sensitivity follow-up.
  python scripts/input_ablation_claude.py dry ARM INPUT          (CPU, 1 BC epoch/512 rows, 1024 PPO frames, 2 probe trials)
  python scripts/input_ablation_claude.py drygpu ARM INPUT       (CUDA plumbing with dry budgets -> dry_gpu/, seed = GPU index)
  python scripts/input_ablation_claude.py fit ARM INPUT SEED     (needs GO; CUDA_VISIBLE_DEVICES == seed % 2)
  python scripts/input_ablation_claude.py queue WORKER NWORKERS  (only the seeds of the visible GPU)
  python scripts/input_ablation_claude.py ood WORKER NWORKERS | analyse
"""
import copy, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prior_seeds30_claude as P

run, parent = P.run, P.parent
ROOT = P.ROOT; OUT = ROOT / "data/wacv/input_ablation"; OOD_OUT = ROOT / "data/audit/wacv_ood_input_ablation_claude"
ARMS = ("human_bc", "human_legal_uniform", "cold_ppo"); INPUTS = ("full", "no_pixels", "no_posterior")
SEEDS = tuple(range(1200, 1210)); SPANS = dict(full=(), no_pixels=((0, 144),), no_posterior=((150, 252),))


class InputMask:
    """Picklable forward-pre-hook: multiply the 252-d input by a fixed 0/1 mask (BC, PPO and probes alike)."""
    def __init__(self, condition):
        self.condition = condition; m = np.ones(252, np.float32)
        for a, b in SPANS[condition]: m[a:b] = 0
        self.mask = m

    def __call__(self, module, inputs):
        x = inputs[0]; return (x * torch.as_tensor(self.mask, device=x.device, dtype=x.dtype),) + tuple(inputs[1:])


def attach(net, condition):
    if condition != "full": net.net.body.register_forward_pre_hook(InputMask(condition))
    return net


def mask_of(net):
    return next((h for h in net.net.body._forward_pre_hooks.values() if isinstance(h, InputMask)), None)


def case(arm, inp): return f"{arm}__{inp}"


def source_probe(net, cache, seed, dry):
    """Frozen competence on training layouts 1-5: all known starts x 3 rotations (150 trials)."""
    sched = [dict(configuration=c, spawn=tuple(s), rotation_index=r) for c in range(1, 6) for r in range(3) for s in cache.geometry(c)[1]]
    if dry: sched = sched[:2]
    m = copy.deepcopy(net); m.rng = np.random.default_rng([20261013, 1200, seed]); m.tg = torch.Generator().manual_seed(int(m.rng.integers(2**31)))
    before = [p.detach().clone() for p in m.net.parameters()]
    env = parent.LongBatch(cache, n=1, seed=0, frames=45 * len(sched), schedule=sched)
    rng0 = m.tg.get_state().tolist(); counters0 = dict(m.count)
    probs = parent.base.run_ppo_select(m, env, False); eps = parent.base.episodes(env, probs, m._values_log)
    assert len(eps) == len(sched) and all(torch.equal(a, b) for a, b in zip(before, m.net.parameters()))
    summary = dict(successes=sum(e["success"] for e in eps), trials=len(eps), mean_spl=float(np.mean([e["spl"] for e in eps])))
    return summary, dict(schedule=sched, episodes=eps, action_rng_initial=rng0, counters_before=counters0, counters_after=dict(m.count))


def hashes():
    return dict(wrapper=parent.sha(__file__), seeds30=P.hashes())


def fit(arm, inp, seed, dry, gpu_dry=False):
    assert arm in ARMS and inp in INPUTS and (dry or seed in SEEDS)
    device = "cuda:0" if (gpu_dry or not dry) else "cpu"
    if not dry:
        go = json.loads((OUT / "GO.json").read_text()); assert go["status"] == "GO" and go["hashes"] == hashes()
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(seed % 2), "all 9 cases of a seed on GPU seed % 2"
        assert ("4090" if seed % 2 == 0 else "3090") in torch.cuda.get_device_name(0)
    Pr, cache, L = run.load(); d = OUT / (("dry_gpu" if gpu_dry else "dry") if dry else "cohort") / case(arm, inp) / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    if (d / "result.json").exists():                                      # never return quietly on an invalid prior result
        old = json.loads((d / "result.json").read_text())
        assert old["status"] == "complete" and old["hashes"] == hashes() and parent.sha(d / "checkpoint_source.pt") == old["checkpoint_sha256"], f"invalid existing result {d}"
        return
    with (d / "started.json").open("x") as f: json.dump(dict(hashes=hashes()), f)
    import gzip
    t0 = time.perf_counter(); r = dict(arm=arm, input=inp, seed=seed, dry=dry, hashes=hashes(), mask_zero_spans=SPANS[inp], status="running")
    try:
        net = attach(run.build(seed, device, L), inp)
        r["bc"] = P.pretrain(net, arm, seed, Pr, dry)
        torch.save(net, d / "checkpoint_prior.pt"); r["prior_checkpoint_sha256"] = parent.sha(d / "checkpoint_prior.pt")
        frames = 1024 if dry else Pr["rl"]["frames_each"]
        env = parent.LongBatch(cache, n=8, seed=seed, frames=frames); parent.base.run_ppo_train(net, env)
        assert env.frames == net.count["frames"] == frames and (mask_of(net) is not None) == (inp != "full")
        torch.save(net, d / "checkpoint_source.pt")
        summary, records = source_probe(net, cache, seed, dry)
        (d / "source_probe.json.gz").write_bytes(gzip.compress(json.dumps(records, default=float).encode()))
        r.update(train=dict(frames=frames, successes=sum(e["success"] for e in env.completed), episodes=len(env.completed)),
                 checkpoint_sha256=parent.sha(d / "checkpoint_source.pt"), source_competence=summary,
                 source_probe_sha256=parent.sha(d / "source_probe.json.gz"),
                 status="complete", wall_s=time.perf_counter() - t0, gpu_name=torch.cuda.get_device_name(0) if device != "cpu" else "cpu")
    except Exception as e:                                                # retained failed result, then re-raise
        import traceback; r.update(status="failed", error=repr(e), traceback=traceback.format_exc())
        (d / "result.json").write_text(json.dumps(r, indent=2, default=str) + "\n"); raise
    (d / "result.json").write_text(json.dumps(r, indent=2) + "\n"); print(json.dumps({k: r[k] for k in ("arm", "input", "seed", "status", "wall_s")}), flush=True)


def queue(worker, n):
    g = int(os.environ["CUDA_VISIBLE_DEVICES"]); jobs = [(a, i, s) for s in SEEDS if s % 2 == g for a in ARMS for i in INPUTS][worker::n]
    for a, i, s in jobs: fit(a, i, s, False)


def ood(worker, n):
    """Same 16 E2 rooms / goal (3,3) / 30 frozen trials; F.probe called directly so FULL records (paths, actions,
    probabilities, pose audit, RNG) are kept; the source mask is re-attached on rebuild. Needs OOD_OUT/GO.json binding all
    90 source checkpoint SHAs and a passing root source audit (all inputs retained, none dropped for incompetence)."""
    import gzip, wacv_ood_zero_shot_claude as O
    F = O.F; go = json.loads((OOD_OUT / "GO.json").read_text()); assert go["status"] == "GO" and go["hashes"] == hashes() and len(go["checkpoints"]) == 90
    aud = json.loads((OUT / "root_source_audit.json").read_text()); assert aud["ok"] and aud["cases"] == 90
    assert go["root_source_audit_sha256"] == parent.sha(OUT / "root_source_audit.json")
    assert int(os.environ["CUDA_VISIBLE_DEVICES"]) in (0, 1)
    F.SEEDS30 = OUT / "cohort"; F.environment = O.env_source_goal
    method = next(x for x in json.loads(F.study.MANIFEST.read_text())["methods"] if x["id"] == "ppo")
    jobs = [(a, i, s, p, st) for s in SEEDS if s % 2 == int(os.environ["CUDA_VISIBLE_DEVICES"]) for a in ARMS for i in INPUTS for p in range(8) for st in ("A", "B")][worker::n]
    for a, i, s, p, st in jobs:
        c = case(a, i); f = OOD_OUT / "cohort" / c / f"seed{s}" / f"pair{p}_{st}.json"
        src = OUT / "cohort" / c / f"seed{s}"; sha = parent.sha(src / "checkpoint_source.pt"); assert sha == go["checkpoints"][f"{c}/seed{s}"]
        if f.exists():                                                    # never skip quietly: an existing record must verify
            old = json.loads(f.read_text()); full = f.with_suffix(".full.json.gz")
            assert old["checkpoint_sha256"] == sha and full.exists() and parent.sha(full) == old["full_record_sha256"], f"invalid existing {f}"
            continue
        F.study.source_model = lambda x, c=c: F.source_checkpoint(c, x); m = F.study.build(method, s)
        hk = mask_of(F.study.source_model(s))
        if hk is not None: m.net.body.register_forward_pre_hook(InputMask(hk.condition))
        rec = F.probe(m, p, st, s, "clean"); f.parent.mkdir(parents=True, exist_ok=True)
        full = f.with_suffix(".full.json.gz"); full.write_bytes(gzip.compress(json.dumps(rec, default=float).encode()))
        f.write_text(json.dumps(dict(arm=c, seed=s, pair=p, room=st, goal=[3, 3], goals=rec["goals"], trials=rec["trials"], checkpoint_sha256=sha,
                                     full_record_sha256=parent.sha(full), gpu_name=torch.cuda.get_device_name(0))) + "\n")


def exact_signflip(d):
    """Two-sided exact sign-flip p over all 2^n sign patterns (n = 10 -> 1024)."""
    import itertools
    d = np.asarray(d, float); obs = abs(d.mean()); n = len(d)
    return float(np.mean([abs((d * np.array(sg)).mean()) >= obs - 1e-12 for sg in itertools.product((1, -1), repeat=n)]))


def analyse():
    from scipy import stats
    from analyse_seeds30_claude import holm
    src = json.loads((OUT / "root_source_audit.json").read_text()); oa = json.loads((OOD_OUT / "root_audit.json").read_text())
    assert src["ok"] and src["cases"] == 90 and oa["ok"] and oa["cases"] == 1440, "root source + OOD audits first"
    go = json.loads((OOD_OUT / "GO.json").read_text()); S = {}
    assert go["status"] == "GO" and go["hashes"] == hashes() and len(go["checkpoints"]) == 90
    assert go["root_source_audit_sha256"] == parent.sha(OUT / "root_source_audit.json")
    for a in ARMS:
        for i in INPUTS:
            c = case(a, i); S[c] = {}
            for s in SEEDS:
                rows = [json.loads((OOD_OUT / "cohort" / c / f"seed{s}" / f"pair{p}_{st}.json").read_text()) for p in range(8) for st in ("A", "B")]
                assert len(rows) == 16 and all(r["checkpoint_sha256"] == go["checkpoints"][f"{c}/seed{s}"] and r["trials"] == 30 for r in rows)
                assert all(parent.sha((OOD_OUT / "cohort" / c / f"seed{s}" / f"pair{p}_{st}.full.json.gz")) == r["full_record_sha256"]
                           for r, (p, st) in zip(rows, [(p, st) for p in range(8) for st in ("A", "B")]))
                S[c][s] = float(np.mean([r["goals"] for r in rows]))
    comp = {i: {a: [json.loads((OUT / "cohort" / case(a, i) / f"seed{s}" / "result.json").read_text())["source_competence"]["successes"] for s in SEEDS] for a in ARMS} for i in INPUTS}
    qualified = {i: all(sum(c >= 75 for c in comp[i][a]) >= 8 for a in ARMS) for i in INPUTS}
    def test(d):
        d = np.asarray(d, float); m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(.975, len(d) - 1)
        if se == 0:                                                      # collapsed condition: deterministic, kept (never dropped)
            return dict(n=len(d), mean=float(m), ci95=[float(m), float(m)], t_p=1.0 if m == 0 else 0.0, exact_signflip_p=exact_signflip(d),
                        tost_p=0.0 if abs(m) < 3 else 1.0, zero_variance=True)
        tp = float(stats.ttest_1samp(d, 0).pvalue); lo = stats.t.sf((m + 3) / se, len(d) - 1); hi = stats.t.cdf((m - 3) / se, len(d) - 1)
        return dict(n=len(d), mean=float(m), ci95=[float(m - t * se), float(m + t * se)], t_p=tp, exact_signflip_p=exact_signflip(d), tost_p=float(max(lo, hi)), zero_variance=False)
    con = {}
    for i in INPUTS:
        con[f"bc-legal@{i}"] = test([S[case("human_bc", i)][s] - S[case("human_legal_uniform", i)][s] for s in SEEDS])
        con[f"bc-cold@{i}"] = test([S[case("human_bc", i)][s] - S[case("cold_ppo", i)][s] for s in SEEDS])
    D = {i: [S[case("human_bc", i)][s] - S[case("human_legal_uniform", i)][s] for s in SEEDS] for i in INPUTS}
    for i in ("no_pixels", "no_posterior"): con[f"interaction full-{i}"] = test(np.subtract(D["full"], D[i]))
    for key, k2 in (("holm_t_p", "t_p"), ("holm_exact_signflip_p", "exact_signflip_p")):
        for k, h in zip(con, holm([v[k2] for v in con.values()])): con[k][key] = h
    inter = [k for k in con if k.startswith("interaction")]
    for k, h in zip(inter, holm([con[k]["tost_p"] for k in inter])): con[k]["holm2_tost_p"] = h
    for k in con:
        if not k.startswith("interaction"): con[k].pop("tost_p")
    out = dict(contrasts=con, source_competence=comp, representation_qualified=qualified, mean_ood_per30={k: float(np.mean(list(v.values()))) for k, v in S.items()},
               reading="Holm over 8 separately for t and exact sign-flip; interaction equivalence only via Holm-2 TOST +-3/30; unresolved interaction is not 'pixels irrelevant'; OOD interpreted only for representation-qualified inputs; exposed rooms (sensitivity follow-up)")
    (OOD_OUT / "analysis_input_ablation.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(dict(qualified=qualified)))


def selftest():
    assert exact_signflip([1.0] * 10) == 2 / 1024 and exact_signflip([1, -1] * 5) == 1.0
    print("selftest ok: exact sign-flip")


if __name__ == "__main__":
    # dispatch through the importable module so pickled hooks are input_ablation_claude.InputMask, never __main__.InputMask
    import importlib
    M = importlib.import_module("input_ablation_claude")
    fit, queue, ood, analyse, selftest = M.fit, M.queue, M.ood, M.analyse, M.selftest
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    c = sys.argv[1]
    if c == "dry": fit(sys.argv[2], sys.argv[3], 0, True)
    elif c == "drygpu": fit(sys.argv[2], sys.argv[3], int(os.environ["CUDA_VISIBLE_DEVICES"]), True, gpu_dry=True)   # GPU plumbing, dry budgets
    elif c == "fit": fit(sys.argv[2], sys.argv[3], int(sys.argv[4]), False)
    elif c == "queue": queue(int(sys.argv[2]), int(sys.argv[3]))
    elif c == "ood": ood(int(sys.argv[2]), int(sys.argv[3]))
    elif c == "analyse": analyse()
    elif c == "selftest": selftest()
