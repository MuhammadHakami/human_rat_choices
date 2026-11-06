"""Export what the project page (site/) needs to run the paper's agents in the browser, from this repository's
registered code and records. Nothing here is a new result.
  python tools/build_site_data.py [--cohort DIR] [--seeds 1000 1001 ...]
--cohort: a seeds30 cohort directory holding more checkpoints (default: this repo's, seed 1000 only; `bash setup.sh
--models` adds all 30). Writes site/data/: env.json (renderer, rooms, localisation atlas), models/*.bin (prior weights),
kc/*.bin (fixed sparse expansions), circuits.json (searched circuit settings), results.json (chart data), golden.json
(Python reference traces checked by site/test/parity.mjs).
"""
import argparse, collections, glob, json, os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""                 # CPU only: nothing here needs a GPU
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
OUT = ROOT / "site/data"
PRIORS = ("human_bc", "human_legal_uniform", "rat_bc", "rat_legal_uniform", "rat_sham", "cold_ppo")
A = ROOT / "data/audit"


def imports():
    from play import cpu_fallback
    cpu_fallback()
    import fresh_rooms_confirm_claude as FR
    import v4n_moved_goal_learning_claude_20261031 as M
    import v4n_camera_ready_claude_20261031 as CR
    return FR, M, CR


def build(M, prior, seed, cohort):
    """M.build with the checkpoint read from `cohort` (registered loader, hash-checked)."""
    F = M.F; method = next(x for x in json.loads(F.study.MANIFEST.read_text())["methods"] if x["id"] == "ppo")
    F.SEEDS30 = Path(cohort); F.study.source_model = lambda s: F.source_checkpoint(prior, s)
    return F.study.build(method, seed)


def env_json(FR, M, CR):
    W = FR.worlds; _, _, source = W.fresh.load(); man = json.loads(FR.MAN.read_text()); gm = M.goal_map(FR.MAN)
    view = source.protocol["view"]; feat = source.protocol["features"]
    atlas = source.raw.features[0, :, :, :, 0, :64].reshape(100, 4, 64)
    rooms = [dict(configuration=r["configuration"], pair=r["pair"], stage=r["stage"], walkable=r["walkable"],
                  familiar_goal=list(FR.GOAL), moved_goal=list(map(int, gm[r["configuration"]]))) for r in man["rows"]]
    return dict(view=view, pooled=[feat["pooled_height"], feat["pooled_width"]], rotations=list(source.rotations),
                starts=man["starts"], moves=[list(m) for m in W.fresh.source.MOVES], coord=W.fresh.source.parent.COORD.tolist(),
                atlas=np.round(atlas.astype(np.float64), 7).ravel().tolist(), sigma=.03, max_decisions=45, gamma=M.GAMMA,
                n_env=M.N_ENV, budget=M.BUDGET, pre=2048, target=list(M.TARGET), rooms=rooms)


def weights(net):
    sd = {k: v.detach().cpu().float() for k, v in net.state_dict().items()}
    keys = ("body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias", "pi.weight", "pi.bias", "v.weight", "v.bias")
    return np.concatenate([sd[k].numpy().ravel() for k in keys]).astype("<f4")


def kc(seed, K=2048, c=7, H=64):
    """The registered sparse expansion (v4n_brainlike_claude_20261103.KC): each of K cells sees c random hidden units."""
    g = torch.Generator().manual_seed(9000 + seed); idx = np.zeros((K, c), "<i4"); w = np.zeros((K, c), "<f4")
    for j in range(K):
        v = torch.randn(c, generator=g) / np.sqrt(c); r = torch.randperm(H, generator=g)[:c]   # Python evaluates the right-hand side first
        idx[j], w[j] = r.numpy(), v.numpy()
    return idx, w


def circuits():
    s3 = json.loads((A / "v4n_brainlike3_20261105/BEST.json").read_text())["configs"]
    out = {}
    for g, c in s3.items():
        c = dict(c); c["eta_v"] = c["rho"] * (1 - .95 * c["lam"]); out[g] = c        # stage 3 samples rho = eta_v / (1 - gamma lambda)
    return out


def ci(v):
    from scipy import stats
    v = np.asarray(v, float); h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.
    return [round(float(v.mean()), 2), round(float(h), 2)]


def results():
    """Chart data, read the same way as the paper's figure scripts (scripts/figs/)."""
    sys.path.insert(0, str(ROOT / "scripts/figs"))
    import fig_camera_kpi_claude as KF
    CRA = json.loads((A / "v4n_camera_ready_20261031/analysis.json").read_text()); ps = CRA["per_seed"]
    ood = {fam: {a: dict(success=[round(x["success"] * 100 / 30, 2) for x in ps[fam][a].values()],
                         revisit=[round(100 * x["revisit_rate"], 2) for x in ps[fam][a].values()]) for a in ps[fam]}
           for fam in ("registered_sample_source", "sample_moved") if fam in ps}
    T = json.loads((ROOT / "data/unified/tartarus_trajectories.json").read_text())["trials"]; nat = {}
    for t in T:
        c = [tuple(x) for x in t["cells"]]; seen = {c[0]}; rv = 0
        for x in c[1:]: rv += x in seen; seen.add(x)
        nat.setdefault(t["species"], {}).setdefault(t["subject"], []).append(rv / max(len(c) - 1, 1))
    native = {sp: [round(100 * float(np.mean(v)), 2) for v in d.values()] for sp, d in nat.items()}
    L = A / "v4n_moved_goal_learning_20261031/test2"; curves = {}
    for l in ("ppo", "td_head", "td_fa"):
        for p in ("human_bc", "human_legal_uniform", "cold_ppo"):
            C = [json.loads(f.read_text())["curve"] for f in sorted((L / l).glob(f"*/{p}/seed*.json"))]
            if C: ks = sorted(int(k) for k in C[0]); curves[f"{l}/{p}"] = dict(x=ks, y=[ci([c[str(k)] * 100 / 30 for c in C]) for k in ks], n=len(C))
    kpi = {}
    for l in KF.NICE:
        for p, _, _ in KF.PRI:
            for key in ("new", "familiar_trained_room", "moved_heldout_room", "familiar_heldout_room"):
                v = list(KF.load(l, p, key).values()) if l != "before" or key != "new" else []
                if v: kpi[f"{l}/{p}/{key}"] = ci(v)
    s4 = A / "v4n_brainlike4_20261106/test/bl4_none"
    snn = {}
    for p, _, _ in KF.PRI:
        for Tn in (1, 2, 4, 8, 16, 32):
            R = [json.loads(f.read_text()) for f in sorted((KF.K / "snn" / p / f"T{Tn}").glob("*.json"))]
            if R:
                by = collections.defaultdict(list)
                for r in R: by[r["seed"]].append(r["goals"] * 100 / 30)
                snn[f"{p}/T{Tn}"] = dict(synops=round(float(np.mean([r["synops_per_decision"] for r in R])), 1), success=ci([np.mean(v) for v in by.values()]))
    D = A / "v4n_rosetta_matched_20261031"; G = json.loads((D / "brains/geometry.json").read_text()); geo = collections.defaultdict(list)
    for k, v in G.items():
        kind, pair, net, _ = json.loads(k)
        if kind == "P": geo[f"{pair}|{(net or 'pooled').split('/')[0]}"].append(v["geometry"]["cka"])
    B = json.loads((D / "lexfree/analysis.json").read_text())
    closed = {a: [100 * r["goals"] / r["trials"] for r in (json.loads(Path(f).read_text()) for f in glob.glob(str(D / f"lexfree/closed/{a}/seed*/cfg*.json")))]
              for a in ("human_bc", "human_legal_uniform", "rat_bc", "cold_ppo")}
    SL = A / "v4n_snn_learned_20261103/summary.json"                     # learned networks converted to spikes (PPO, local rules)
    snn_learned = {k: dict(synops=round(v["synops"], 1), spiking=round(v["spiking"], 2)) for k, v in json.loads(SL.read_text()).items()}
    return dict(proofs=proofs(), snn_learned=snn_learned, ood=ood, native=native, curves=curves, kpi=kpi, snn=snn, nice=KF.NICE,
                cka={k: ci(v) for k, v in geo.items()}, closed_loop={a: ci(v) for a, v in closed.items() if v},
                text_to_move={a: B["C"][a]["composed"] for a in B["C"]})


def animals():
    """Human vs rat, per maze configuration and start: occupancy share per cell, trials, successes and revisit rate.
    Aggregates only; the raw trajectories (de Cothi et al. 2022) are not redistributed. One example trial per species
    (configuration 16, start 9) is the one already shown in the README."""
    from arc_fly.unified_env import load_tasks
    tasks = {t.id: t for t in load_tasks(ROOT / "data/unified/tartarus/tasks.json")}
    T = json.loads((ROOT / "data/unified/tartarus_trajectories.json").read_text())["trials"]; agg = {}
    for t in T:
        if t["task_id"] is None: continue                     # start not among the registered tasks (543 trials)
        a = agg.setdefault(t["task_id"], {}).setdefault(t["species"], dict(n=0, success=0, occ=[0] * 100, rv=[]))
        c = [tuple(x) for x in t["cells"]]; a["n"] += 1; a["success"] += int(t["success"])
        for x in set(c): a["occ"][x[0] * 10 + x[1]] += 1
        a["rv"].append(sum(x in c[:i] for i, x in enumerate(c) if i) / max(len(c) - 1, 1))
    out = {}
    for k, sp in agg.items():
        tk = tasks[k]; out[k] = dict(walkable=tk.walkable.astype(int).tolist(), goal=list(tk.goal), spawn=list(tk.spawn),
                                     species={s: dict(n=v["n"], success=v["success"], occ=[round(o / v["n"], 3) for o in v["occ"]],
                                                      revisit=round(100 * float(np.mean(v["rv"])), 1)) for s, v in sp.items()})
    ex = {s: next(t["cells"] for t in T if t["task_id"] == "tartarus/16/9" and t["species"] == s and t["success"]) for s in ("human", "rat")}
    return dict(tasks=out, example=dict(task="tartarus/16/9", paths=ex))


def proofs():
    """The theorems the page lists, with their exact Lean statements (read from proofs/ and proofs_ml/)."""
    import re
    src = "".join(f.read_text() for d in ("proofs/Proofs", "proofs_ml/ProofsML") for f in sorted((ROOT / d).glob("*.lean")))
    out = {}
    for name in ("trapped_forever", "relabel_keeps_states", "species_gap_decomposition", "trace_bounded", "softmax_logit_grad"):
        m = re.search(rf"theorem {name}\b(.*?):=", src, re.S); out[name] = re.sub(r"\s+", " ", f"theorem {name}{m.group(1)}").strip()
    return out


LEARNED = {"ppo": "data/audit/v4n_moved_goal_learning_20261031/test2/ppo/registered/{p}/seed{s}.final.pt",
           "bl3": "data/audit/v4n_brainlike3_20261105/test/bl3_none/{p}/seed{s}.final.pt"}


def learned(root, seeds):
    """Weights after learning the moved goal, as tested in the paper: PPO (whole network, backprop) and the stage-3 brain
    circuit (plastic KC->action / KC->value synapses only; its fixed expansion must equal kc/<seed>.bin)."""
    import hashlib
    (OUT / "learned").mkdir(exist_ok=True); out = []
    for s in seeds:
        for p in ("human_bc", "human_legal_uniform", "cold_ppo"):
            for k, pat in LEARNED.items():
                f = Path(root) / pat.format(p=p, s=s)
                if not f.exists(): continue
                sd = torch.load(f, map_location="cpu", weights_only=False)
                if k == "ppo":
                    keys = ("body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias", "pi.weight", "pi.bias", "v.weight", "v.bias")
                    b = np.concatenate([sd[q].float().numpy().ravel() for q in keys]).astype("<f4")
                else:
                    i, w = kc(s); P = np.zeros((64, 2048), np.float32)
                    for j in range(2048): P[i[j], j] = w[j]
                    assert np.array_equal(P, sd["pi.P"].numpy()), f"expansion differs for seed {s}"
                    b = np.concatenate([sd["pi.R.weight"].float().numpy().ravel(), sd["v.R.weight"].float().numpy().ravel()]).astype("<f4")
                (OUT / "learned" / f"{k}_{p}_{s}.bin").write_bytes(b.tobytes())
                rec = json.loads(f.with_name(f.name.replace(".final.pt", ".json")).read_text())
                out.append(dict(kind=k, prior=p, seed=s, source=pat.format(p=p, s=s), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                pair=rec.get("pair", s % 8), recorded=rec.get("moved_trained_room") if k == "bl3" else rec["curve"][max(rec["curve"], key=int)]))
    return out


def golden(FR, M, cohort):
    """Python reference: one frozen probe episode (obs, logits per step) and 64 steps of the stage-3 circuit learning
    from recorded actions (final plastic synapses). The browser code must reproduce both."""
    import v4n_brainlike_claude_20261103 as B, v4n_brainlike2_claude_20261105 as B2
    gm = M.goal_map(FR.MAN); m = build(M, "human_bc", 1000, cohort); net = m.net
    sched = [dict(configuration=100, spawn=json.loads(FR.MAN.read_text())["starts"][3], rotation_index=1)]
    env = M.env_for(FR.MAN, gm, 0, 1, 1000, "clean", 45, sched); obs = env.reset(); rng = np.random.default_rng(5); steps = []
    while not env.exhausted:
        with torch.no_grad(): lg = net.pi(net.body(torch.as_tensor(obs))).numpy()[0]
        a = int(rng.choice(4, p=np.exp(lg - lg.max()) / np.exp(lg - lg.max()).sum())); steps.append(dict(obs=np.round(obs[0], 6).tolist(), logits=lg.round(6).tolist(), a=a))
        obs, r, d = env.step(np.array([a]))
        if d[0]: break
    cfg = circuits()["none"]; B.KC, B.Circuit = B.KC, B2.Circuit2
    mb = B.build("human_bc", 1000, cfg); L = B2.Circuit2(mb.net, cfg)
    env = M.env_for(FR.MAN, {c: FR.GOAL for c in gm}, 0, M.N_ENV, 1500, "clean", 2048); worlds = []; phys = env.env; nw = phys._new_world
    def logged():                                       # the browser replays these spawns instead of numpy's RNG
        w = nw(); worlds.append(dict(spawn=list(w["spawn"]), rotation_index=w["rotation_index"])); return w
    phys._new_world = logged; obs = env.reset(); acts = []
    z, p, v = L.forward(obs); p0 = p[0].tolist(); gen = torch.Generator().manual_seed(1)
    for t in range(64):
        a = torch.multinomial(p.cpu(), 1, generator=gen).squeeze(-1); acts.append(a.tolist())
        obs, r, done = env.step(a.numpy()); z2, p2, v2 = L.forward(obs); L.step(z, p, v, a, r, done, v2); z, p, v = z2, p2, v2
    import v4n_context_gating_claude_20261103 as G
    cg = circuits()["inferred"]; mg = B.build("human_bc", 1000, cg); h = mg.net.body(torch.as_tensor(obs[:1])); gz = {}
    for c in (0, 1): G.CTX["c"] = c; gz[c] = torch.nonzero(mg.net.pi.code(h)[0]).ravel().tolist()
    G.CTX["c"] = 0
    return dict(probe=dict(prior="human_bc", seed=1000, configuration=100, schedule=sched, steps=steps),
                gate=dict(obs=np.round(obs[0], 6).tolist(), active={str(c): v for c, v in gz.items()}),
                circuit=dict(prior="human_bc", seed=1000, cfg=cfg, env_seed=1500, actions=acts, worlds=worlds, p0=p0,
                             Wa=L.Wa.detach().cpu().numpy().round(7).ravel().tolist(), Wv=L.Wv.detach().cpu().numpy().round(7).ravel().tolist(),
                             rms=None if L.rms is None else L.rms.detach().cpu().numpy().tolist()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort", default=str(ROOT / "data/wacv/seeds30/cohort"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    ap.add_argument("--learned-root", default=str(ROOT), help="tree holding the learned .final.pt files (release records)"); q = ap.parse_args()
    FR, M, CR = imports(); (OUT / "models").mkdir(parents=True, exist_ok=True); (OUT / "kc").mkdir(exist_ok=True)
    (OUT / "env.json").write_text(json.dumps(env_json(FR, M, CR), separators=(",", ":")))
    have = []
    for s in q.seeds:
        if not all((Path(q.cohort) / p / f"seed{s}/checkpoint_source.pt").exists() for p in PRIORS): continue
        for p in PRIORS: (OUT / "models" / f"{p}_{s}.bin").write_bytes(weights(build(M, p, s, q.cohort).net).tobytes())
        i, w = kc(s); (OUT / "kc" / f"{s}.bin").write_bytes(i.astype("<i4").tobytes() + w.tobytes()); have.append(s)
    (OUT / "circuits.json").write_text(json.dumps(dict(seeds=have, priors=PRIORS, configs=circuits(), learned=learned(q.learned_root, have),
        gate_half0=sorted(torch.randperm(2048, generator=torch.Generator().manual_seed(31337))[:1024].tolist())), indent=1))   # G.Gated: context 0 uses this half of the cells
    (OUT / "results.json").write_text(json.dumps(results(), separators=(",", ":")))
    (OUT / "animals.json").write_text(json.dumps(animals(), separators=(",", ":")))
    (OUT / "golden.json").write_text(json.dumps(golden(FR, M, q.cohort), separators=(",", ":")))
    print("seeds", have, "->", OUT)


if __name__ == "__main__":
    main()
