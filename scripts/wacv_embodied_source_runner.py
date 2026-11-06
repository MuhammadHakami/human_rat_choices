"""Source-development runner for docs/WACV_EMBODIED_SOURCE_PROTOCOL_20261012.json (registered; sha 2ae1b1c6...).

Reads every setting from the registered protocol; nothing is tuned here. One fit = (method, rate index, seed):
  1. train on the source curriculum for exactly 32,768 frames (SourceBatch, n=1; recurrent PPO n=8, 32 rollouts of
     128 x 8), curriculum/final boundaries forced terminal by the environment;
  2. save the source-trained checkpoint (preserved before any selection);
  3. selection on configurations 6 then 7 (selection_schedule, 20 native starts, 45 decisions each), 3 repeats x
     {frozen, adaptive}, each from a FRESH deep copy of the checkpoint with deterministic per-(fit, arm, repeat) RNGs;
     state reset every trial; frozen = no weight update of any kind (entropy included); adaptive = weights kept
     across trials and layouts, no layout/boundary information;
  4. write result.json (+ episodes.json.gz with paths, actions, sampled probabilities, rewards) with frame/update/
     state/RSS/wall-time counts and protocol/cache/code hashes. Non-finite parameters or probabilities end the fit
     as status "failed" (recorded, never retried).
Recurrent PPO in selection: one environment slot, 128-step segments (final segment stops at schedule exhaustion),
stored LSTM state at segment start; minibatches 2 -> 1 because one slot cannot be split; objective,
epochs, clip, rate, entropy, value coefficient and gradient clipping unchanged; every interaction counted.
Rate-BPTT Adam replaces the module's SGD optimiser explicitly (eps from protocol); optimiser tensors are counted
from the actual optimiser state.
Commands (CPU, 1 thread; CUDA_VISIBLE_DEVICES empty):
  dry    METHOD --out DIR             technical check only: 128 training frames (PPO: one 128x8 rollout = 1024 frames,
                                      documented separately) + 2 selection trials per arm; NO score interpretation
  fit    METHOD RATE_INDEX SEED --out DIR     one registered source fit (do not launch the cohort without root GO)
  audit  --out DIR                    re-verify every result against its own episode records and hashes
  select --out DIR                    requires all 63 audited results; registered rule only
"""
import argparse, copy, gzip, hashlib, json, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wacv_embodied_source_cache import SourceCache
from wacv_embodied_source_env import SourceBatch, selection_schedule
from eprop_actor_critic import EpropAC
from rate_rnn_actor_critic import RateBPTTAC
from spiking_bptt_actor_critic import SpikingBPTTAC
from recurrent_ppo import RecurrentPPO, gae

PROTOCOL = ROOT / "docs/WACV_EMBODIED_SOURCE_PROTOCOL_20261012.json"
PROTOCOL_SHA = "2ae1b1c6db84298ecc583e13c73559ad18c3ec054caeb2dbcd91eaef0eccedda"
MANIFEST = ROOT / "data/audit/wacv_embodied_source_20261012/cache/source_visual_cache.json"
CODE = ["scripts/wacv_embodied_source_runner.py", "scripts/eprop_actor_critic.py", "scripts/rate_rnn_actor_critic.py",
        "scripts/spiking_bptt_actor_critic.py", "scripts/recurrent_ppo.py", "scripts/wacv_embodied_source_env.py",
        "scripts/wacv_embodied_source_cache.py", "scripts/tartarus_visual_env.py"]
GO = ROOT / "data/audit/wacv_embodied_source_20261012/source_GO.json"
ARMS = ("frozen", "adaptive")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise SystemExit("protocol file differs from the registered hash")
    cache = SourceCache(MANIFEST, PROTOCOL); return cache, cache.protocol


def require_go(out):
    if not GO.exists(): raise RuntimeError("Root source GO missing")
    g = json.loads(GO.read_text())
    current = hashes()
    assert g["status"] == "GO" and g["hashes"] == current, "Bound source code/data changed"
    assert Path(out).resolve() == Path(g["output"]).resolve(), "Unregistered output cohort"


def hashes():
    return dict(protocol=sha(PROTOCOL), cache_manifest=sha(MANIFEST), cache_array=sha(MANIFEST.with_suffix(".npz")),
                code={c: sha(ROOT / c) for c in CODE})


# ---- learners built only from protocol values ---------------------------------------------------------------------
def build(method, rate, seed, P, obs_dim):
    cl = P["common_learning"]; m = next(x for x in P["methods"] if x["name"] == method); f32 = torch.float32
    assert cl["dtype"] == "float32" and cl["readout_kappa"] == "exp(-1/20)" and cl["tau_m"] == 20.0   # kappa = alpha default
    if method.startswith("eprop"):
        e = P["eprop"]; assert e["entropy_mode"] == "independent" and e["plastic_core"] and not e["self_synapses"]
        return EpropAC(obs_dim, m["hidden"], 4, frac_alif=cl["alif_fraction"], tau_m=cl["tau_m"], tau_a=cl["tau_a"],
                       beta=cl["adaptive_threshold_beta"], v_th=cl["threshold"], gamma_pd=cl["pseudo_derivative_gain"],
                       gamma=cl["gamma"], c_v=cl["value_coefficient"], lr=rate, lr_core=m["core_rate_ratio"] * rate,
                       feedback={"eprop_symmetric": "symmetric", "eprop_random": "random"}[method],
                       entropy=cl["entropy"], entropy_mode="independent", dtype=f32, seed=seed)
    if method.startswith("rate_bptt"):
        net = RateBPTTAC(obs_dim, m["hidden"], 4, tau=cl["tau_m"], gamma=cl["gamma"], c_v=cl["value_coefficient"],
                         lr=rate, entropy=cl["entropy"], dtype=f32, seed=seed)
        if method.endswith("adam"):
            net.opt = torch.optim.Adam(net.P.values(), lr=rate, eps=P["bptt"]["adam_epsilon"])
        return net
    if method.startswith("spike_bptt"):
        net = SpikingBPTTAC(obs_dim, m["hidden"], 4, frac_alif=cl["alif_fraction"], tau_m=cl["tau_m"], tau_a=cl["tau_a"],
                            beta=cl["adaptive_threshold_beta"], v_th=cl["threshold"], gamma_pd=cl["pseudo_derivative_gain"],
                            gamma=cl["gamma"], c_v=cl["value_coefficient"], lr=rate, entropy=cl["entropy"],
                            optimizer=method.rsplit("_", 1)[1], dtype=f32, seed=seed)
        if method.endswith("adam"): assert net.opt.defaults["eps"] == P["bptt"]["adam_epsilon"]
        return net
    if method == "recurrent_ppo":
        p = P["ppo"]
        net = RecurrentPPO(obs_dim, 4, hidden=m["hidden"], n_envs=p["n_envs"], n_steps=p["n_steps"], gamma=cl["gamma"],
                           lam=p["lambda"], lr=rate, clip=p["clip"], ent=cl["entropy"], vf=cl["value_coefficient"],
                           epochs=p["epochs"], minibatches=p["minibatches"], seed=seed)
        assert net.opt.defaults["eps"] == p["adam_epsilon"] and p["max_grad_norm"] == 0.5 and p["advantage_normalization"]
        return net
    raise ValueError(method)


def params(net):
    if isinstance(net, EpropAC): return [net.W, net.W_out, net.b_out, net.w_V, net.b_V]
    if isinstance(net, RecurrentPPO): return list(net.net.parameters())
    return list(net.P.values())


def finite(net):
    return all(bool(torch.isfinite(t).all()) for t in params(net))


def resources(net, T=45):
    b = lambda ts: int(sum(t.element_size() * t.numel() for t in ts))
    opt = getattr(net, "opt", None)
    opt_b = b([v for st in opt.state.values() for v in st.values() if torch.is_tensor(v) and v.dim() > 0]) if opt else 0
    out = dict(weights_bytes=b(params(net)), optimiser_bytes=opt_b,
               plastic_parameters=int(sum(t.numel() for t in params(net))), counters=dict(net.count))
    if isinstance(net, EpropAC):
        m = net.memory_bytes(); out.update(fixed_bytes=m["fixed_structure"] + m["fixed_feedback"], state_bytes=m["persistent_state"],
                                           effective_core_synapses=int(net.mask.sum()))
    elif isinstance(net, RecurrentPPO):
        m = net.memory_bytes(); out.update(state_bytes=m["persistent_state"], buffer_bytes=m["experience_buffer"])
    else:
        m = net.memory_bytes(T); out.update(state_bytes=m["persistent_state"],
                                            bptt_buffer_lower_bound_bytes=m["experience_buffer_lower_bound"])
    out["dense_ops"] = dense_ops(net); out["maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return out


def dense_ops(net):
    """Dense multiply-accumulate counts (our torch matmuls are dense; spike events are only a sparsity proxy).
    Forward per decision: core matvec + readouts. Learning: e-prop ~ 7 n*P elementwise per learning decision
    (eps_v, e, eps_a, ebar, G, TD apply, entropy apply); BPTT backward ~ 2x forward per learned decision;
    PPO update ~ epochs x 3x forward per buffered decision. Formulas are estimates, labelled as such."""
    c = net.count
    if isinstance(net, RecurrentPPO):
        lstm = net.net.lstm; h = lstm.hidden_size; i = lstm.input_size
        fwd = 4 * h * (i + h) + 5 * h
        learned = c.get("updates", 0) * net.cfg["n_steps"] * net.cfg["n_envs"]
        return dict(forward_macs_per_decision=fwd, forward_macs_total=fwd * c["frames"],
                    learning_macs_estimate=net.cfg["epochs"] * 3 * fwd * learned, formula="LSTM 4h(i+h)+heads; update epochs*3*fwd*n_steps*n_envs per rollout (training config)")
    if isinstance(net, EpropAC):
        n, P = net.n, net.n_in + net.n; fwd = n * P + (net.k + 1) * n
        return dict(forward_macs_per_decision=fwd, forward_macs_total=fwd * c["frames"],
                    learning_ops_estimate=7 * n * P * c["td_errors"], formula="core n*P + readouts; learning ~7 n*P per TD step")
    n = net.n; P = (net.P["W_in"].shape[1] + n) if "W_in" in net.P else net.P["W"].shape[1]
    fwd = n * P + (net.k + 1) * n
    return dict(forward_macs_per_decision=fwd, forward_macs_total=fwd * c["frames"],
                learning_macs_estimate=2 * fwd * c["frames"], formula="core n*P + readouts; BPTT backward ~2x forward per decision")


# ---- interaction loops ----------------------------------------------------------------------------------------------
def run_single(net, env, learn):
    """One-slot loop for e-prop / rate-BPTT / spiking-BPTT. learn=False: no weight change of any kind."""
    probs = []; acc = None if learn else {}; net._values_log = values = []
    is_ep = isinstance(net, EpropAC)

    def act(x, r):
        a = net.step(x, reward_prev=r, accumulate=acc) if is_ep else net.step(x, reward_prev=r)
        pi = net.last_pi if is_ep else net.buf[-1][0]                    # pre-update snapshot used for sampling
        values.append(net.last_value if is_ep else float(net.buf[-1][1].detach()))
        probs.append([float(v) for v in pi.detach()]);
        if not all(np.isfinite(probs[-1])): raise FloatingPointError("non-finite policy")
        return a

    def finish(r):
        if is_ep: net.end(r, accumulate=acc)
        elif learn: net.end(r)
        else: net.reset()
        if not finite(net): raise FloatingPointError("non-finite parameters")

    obs = env.reset(); a = act(obs[0], 0.0)
    while True:
        obs, r, done = env.step(np.array([a]))
        if done[0]:
            finish(float(r[0]))
            if env.frames >= env.limit or env.exhausted: break
            a = act(obs[0], 0.0)
        else:
            a = act(obs[0], float(r[0]))
    return probs


def run_ppo_train(net, env):
    probs = []; net._values_log = values = []; obs = env.reset(); n = env.n; h = net.hidden
    st = (torch.zeros(1, n, h), torch.zeros(1, n, h)); dp = np.zeros(n, np.float32)
    while env.frames < env.limit:
        B, st0, obs, st, dp = net.rollout(env, obs, st, dp); net.update(B, st0)
        probs.extend(B["probs"].reshape(-1, 4).tolist()); values.extend(B["val"].reshape(-1).tolist())
        if not finite(net): raise FloatingPointError("non-finite parameters")
    return probs


def run_ppo_select(net, env, learn):
    """One slot, n_steps (128) segments exactly as in training, minibatches 1 (one slot cannot be split); the collector
    stops early only when the fixed 20-trial schedule is exhausted (no extra uncharged action). LSTM state carries
    across segment boundaries inside a trial and resets at trial starts via done flags; the last segment bootstraps
    from the actual done flag (value of the next observation if not done). A one-element segment would make the
    advantage std undefined: recurrent_ppo then uses std 0 (normalised advantage 0) and counts it in
    count['singleton_advantage_minibatches'], reported with the fit."""
    probs = []; net._values_log = values = []; obs = env.reset(); h = net.hidden; n_steps = net.cfg["n_steps"]
    net.cfg["mb"] = 1; net.cfg["n_envs"] = 1
    st = (torch.zeros(1, 1, h), torch.zeros(1, 1, h)); done_prev = np.ones(1, np.float32)   # first step starts a trial
    while not env.exhausted and env.frames < env.limit:
        st0 = (st[0].clone(), st[1].clone())
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done_before", "done_after")}
        with torch.no_grad():
            for _ in range(n_steps):
                x = torch.as_tensor(obs)[None]; d0 = torch.as_tensor(done_prev)[None]
                o, st = net.net.unroll(x, st, d0); dist, v = net.net.heads(o[0])
                a = torch.multinomial(dist.probs, 1, generator=net.tg).squeeze(-1); probs.append(dist.probs[0].tolist()); values.append(float(v[0]))
                obs, r, dn = env.step(a.numpy()); net.count["frames"] += 1
                for k, val in (("obs", x[0]), ("act", a), ("logp", dist.log_prob(a)), ("val", v), ("rew", torch.as_tensor(r)),
                               ("done_before", d0[0]), ("done_after", torch.as_tensor(dn, dtype=torch.float32))):
                    buf[k].append(val)
                done_prev = dn.astype(np.float32)
                if env.exhausted or env.frames >= env.limit: break
            if done_prev[0]: last_v = torch.zeros(1)
            else:
                o, _ = net.net.unroll(torch.as_tensor(obs)[None], (st[0].clone(), st[1].clone()), torch.zeros(1, 1))
                last_v = net.net.heads(o[0])[1]
        if learn:
            B = {k: torch.stack(v) for k, v in buf.items()}
            B["adv"], B["ret"] = gae(B["rew"], B["val"], B["done_after"], last_v, net.cfg["gamma"], net.cfg["lam"])
            net.update(B, st0)
            if not finite(net): raise FloatingPointError("non-finite parameters")
    return probs


def episodes(env, probs, values=None):
    """Episode records with their per-decision sampled probabilities, value estimates and rewards."""
    out, i = [], 0
    for e in sorted(env.completed, key=lambda x: (x["episode"], x["slot"])):
        out.append(dict(e, rewards=[0.0] * (e["decisions"] - 1) + [e["reward_sum"]]))
    if env.n == 1:
        for e in out:
            e["probabilities"] = probs[i:i + e["decisions"]]
            if values is not None: e["values"] = values[i:i + e["decisions"]]
            i += e["decisions"]
        assert i == len(probs), (i, len(probs))
    else:
        out.append(dict(note="PPO 8-slot training: probabilities/values in time-major slot order", rollout_probabilities=probs, rollout_values=values))
    return out


def arm_seed(seed, rate_idx, method, arm, repeat):
    m = [x["name"] for x in json.loads(PROTOCOL.read_text())["methods"]].index(method)
    return [20261012, m, rate_idx, seed, ARMS.index(arm), repeat]


def reseed(net, ss):
    net.rng = np.random.default_rng(ss)
    if isinstance(net, RecurrentPPO): net.tg = torch.Generator().manual_seed(int(np.random.SeedSequence(ss).generate_state(1)[0]))


def selection(trained, cache, P, method, rate_idx, seed, trials=None):
    res = {}
    for arm in ARMS:
        for rep in range(P["source"]["select_protocol"]["repeats"]):
            net = copy.deepcopy(trained); reseed(net, arm_seed(seed, rate_idx, method, arm, rep)); count0 = dict(net.count); arm_t0 = time.perf_counter()
            sched = selection_schedule(cache, rep)[:trials] if trials else selection_schedule(cache, rep)
            env = SourceBatch(cache, n=1, seed=0, frames=len(sched) * P["source"]["select_protocol"]["max_decisions_per_trial"], schedule=sched)
            learn = arm == "adaptive"; w0 = [t.clone() for t in params(net)]
            probs = run_ppo_select(net, env, learn) if isinstance(net, RecurrentPPO) else run_single(net, env, learn)
            if not learn: assert all(torch.equal(a, b) for a, b in zip(w0, params(net))), "frozen arm changed weights"
            assert len(env.completed) == len(sched) and (env.exhausted or env.frames == env.limit)   # all trials 45 -> limit, else exhausted
            eps = episodes(env, probs, net._values_log)
            res[f"{arm}_{rep}"] = dict(success=float(np.mean([e["success"] for e in eps])), trials=len(eps),
                                       decisions=env.frames, episodes=eps, wall_s=time.perf_counter()-arm_t0,
                                       counters_delta={k:net.count.get(k,0)-count0.get(k,0) for k in set(net.count)|set(count0)},
                                       resources=resources(net), frozen_weights_identical=not learn)
    return res


def fit(method, rate_idx, seed, out, dry=False):
    torch.set_num_threads(1); t0 = time.time()
    if not dry: require_go(out)
    cache, P = load(); m = next(x for x in P["methods"] if x["name"] == method)
    assert seed in P["source"]["seeds"] and 0 <= rate_idx < len(m["rates"])
    rate = m["rates"][rate_idx]; ppo = method == "recurrent_ppo"
    d = Path(out) / method / f"rate{rate_idx}" / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    frames = (P["ppo"]["n_steps"] * P["ppo"]["n_envs"] if ppo else 128) if dry else P["source"]["training_frames"]
    rec = dict(method=method, rate_index=rate_idx, rate=rate, seed=seed, dry=dry, frames_planned=frames,
               hashes=hashes(), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status="running")
    try:
        net = build(method, rate, seed, P, cache.obs_dim)
        env = SourceBatch(cache, n=P["ppo"]["n_envs"] if ppo else 1, seed=seed, frames=frames)
        probs = run_ppo_train(net, env) if ppo else run_single(net, env, learn=True)
        assert env.frames == frames, (env.frames, frames)
        train_eps = episodes(env, probs, net._values_log)
        torch.save(net, d / "checkpoint_source.pt")                       # preserved before any selection
        rec["checkpoint_sha256"] = sha(d / "checkpoint_source.pt")
        rec["train"] = dict(frames=env.frames, episodes=len(env.completed), resources=resources(net),
                            success_rate_technical=float(np.mean([e["success"] for e in env.completed])))
        sel = selection(net, cache, P, method, rate_idx, seed, trials=2 if dry else None)
        rec["selection"] = {k: {q: v[q] for q in ("success", "trials", "decisions", "wall_s", "counters_delta", "resources", "frozen_weights_identical")} for k, v in sel.items()}
        rec["adaptive_success"] = float(np.mean([sel[f"adaptive_{r}"]["success"] for r in range(3)]))
        rec["frozen_success"] = float(np.mean([sel[f"frozen_{r}"]["success"] for r in range(3)]))
        with open(d / "episodes.json.gz", "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:   # byte-reproducible
            gz.write(json.dumps(dict(train=train_eps, selection={k: v["episodes"] for k, v in sel.items()})).encode())
        rec["episodes_sha256"] = sha(d / "episodes.json.gz"); rec["status"] = "complete"
    except (FloatingPointError, ValueError) as err:
        rec.update(status="failed", error=repr(err), traceback=traceback.format_exc()[-2000:],
                   failure_phase="selection" if "checkpoint_sha256" in rec else "training",
                   training_frames_completed=env.frames if "env" in locals() else 0,
                   partial_resources=resources(net) if "net" in locals() else None)
        if "env" in locals():
            with open(d / "failed_partial_episodes.json.gz", "wb") as raw, gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as gz:
                gz.write(json.dumps(env.completed).encode())
            rec["partial_episodes_sha256"]=sha(d / "failed_partial_episodes.json.gz")
    rec.update(wall_s=round(time.time() - t0, 2), maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               note="technical dry run; no scientific interpretation" if dry else "registered source fit")
    (d / "result.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec.get(k) for k in ("method", "rate", "seed", "dry", "status", "wall_s", "adaptive_success", "frozen_success")}))
    return rec


def audit(out):
    cache, P = load(); rows = []; ok = True
    for m in P["methods"]:
        for ri in range(len(m["rates"])):
            for s in P["source"]["seeds"]:
                d = Path(out) / m["name"] / f"rate{ri}" / f"seed{s}"; f = d / "result.json"
                if not f.exists(): rows.append(dict(method=m["name"], rate_index=ri, seed=s, present=False)); ok = False; continue
                r = json.loads(f.read_text()); row = dict(method=m["name"], rate_index=ri, seed=s, present=True, status=r["status"], dry=r["dry"])
                if r["dry"]: ok = False; row["error"] = "dry run in cohort"
                if r["status"] == "complete":
                  try:
                    assert sha(d / "episodes.json.gz") == r["episodes_sha256"] and sha(d / "checkpoint_source.pt") == r["checkpoint_sha256"]
                    E = json.load(gzip.open(d / "episodes.json.gz", "rt"))
                    tr = [e for e in E["train"] if "decisions" in e]
                    assert sum(e["decisions"] for e in tr) == P["source"]["training_frames"] == r["train"]["frames"]
                    for k, v in r["selection"].items():
                        eps = E["selection"][k]; assert len(eps) == 20 and abs(np.mean([e["success"] for e in eps]) - v["success"]) < 1e-12
                        assert all(e["configuration"] in (6, 7) and e["decisions"] <= 45 for e in eps)
                    assert abs(np.mean([r["selection"][f"adaptive_{q}"]["success"] for q in range(3)]) - r["adaptive_success"]) < 1e-12
                    row["code"] = r["hashes"]["code"]; row["protocol"] = r["hashes"]["protocol"]; row["verified"] = True
                  except AssertionError as err:
                    row.update(verified=False, error=f"verification failed: {err!r}"[:300]); ok = False
                rows.append(row)
    codes = {json.dumps(x.get("code"), sort_keys=True) for x in rows if x.get("code")}
    rep = dict(ok=ok and len(codes) <= 1, fits_expected=P["source"]["fits_planned"], present=sum(x["present"] for x in rows),
               complete=sum(x.get("status") == "complete" for x in rows), verified=sum(bool(x.get("verified")) for x in rows), failed=sum(x.get("status") == "failed" for x in rows),
               single_code_version=len(codes) <= 1, rows=rows)
    (Path(out) / "audit.json").write_text(json.dumps(rep, indent=1)); print(json.dumps({k: v for k, v in rep.items() if k != "rows"}))
    return rep


def select(out):
    cache, P = load(); a = json.loads((Path(out) / "audit.json").read_text())
    root_audit=json.loads((Path(out) / "root_audit.json").read_text())
    if not root_audit["ok"] or root_audit["dry"] or root_audit["cases"]!=63:
        raise SystemExit("selection requires complete independent source audit")
    if not a["ok"] or a["present"] != P["source"]["fits_planned"]:
        raise SystemExit("selection requires a complete, audited 63-fit cohort")
    choice = {}
    for m in P["methods"]:
        cand = []
        for ri in range(len(m["rates"])):
            R = [json.loads((Path(out) / m["name"] / f"rate{ri}" / f"seed{s}" / "result.json").read_text()) for s in P["source"]["seeds"]]
            ad = float(np.mean([r.get("adaptive_success", 0.0) if r["status"] == "complete" else 0.0 for r in R]))   # failed = 0 (declared)
            fr = float(np.mean([r.get("frozen_success", 0.0) if r["status"] == "complete" else 0.0 for r in R]))
            cand.append((-ad, -fr, ri, ad, fr))
        best = sorted(cand)[0]
        choice[m["name"]] = dict(rate_index=best[2], rate=m["rates"][best[2]], adaptive=best[3], frozen=best[4],
                                 candidates=[dict(rate_index=c[2], adaptive=c[3], frozen=c[4]) for c in cand])
    (Path(out) / "selection.json").write_text(json.dumps(dict(rule=P["source"]["select_protocol"], choice=choice), indent=1))
    print(json.dumps({k: (v["rate"], v["adaptive"]) for k, v in choice.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["dry", "fit", "audit", "select"])
    ap.add_argument("method", nargs="?"); ap.add_argument("rate_index", nargs="?", type=int); ap.add_argument("seed", nargs="?", type=int)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "": raise SystemExit("run with CUDA_VISIBLE_DEVICES= (CPU only)")
    if a.cmd == "dry": fit(a.method, 0, 0, a.out, dry=True)
    elif a.cmd == "fit": fit(a.method, a.rate_index, a.seed, a.out)
    elif a.cmd == "audit": audit(a.out)
    else: select(a.out)
