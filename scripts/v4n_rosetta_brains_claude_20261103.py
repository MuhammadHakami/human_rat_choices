"""Rosetta protocol across views, words, actions, predicted cortex and network 'brains' (camera-ready, author 2026-11-03:
"the same rosetta paper strategy with action-text and action-image, then action-text-image; the geometrical one also to
rat/human brain with also PPO and bio plausible ones"). Descriptive, secondary; registered before any outcome. GPU 1 for
encoders and networks, CPU for the authors' solver (upstream unpaired-rosetta fdfad84, unmodified).
Moments (common rows for every network): the four frozen priors (human_bc, human_legal_uniform, rat_bc, cold_ppo; seeds
1000-1002) act in the 16 fresh rooms with their familiar goals (the probe schedule, sampled actions); at each decision we
record the policy input (252: pixels, flags, pose belief), pose and frame key; 1,000 per arm x seed (12,000 rows).
Rooms 100-107 fit, 108-115 test. Modalities per row:
  image   DINOv2-base CLS of the rendered view      text    MPNet of the template caption (what is visible)
  cortex  MICrONS foundation-model twin of mouse visual cortex (Wang et al. 2025) for the same frame, PCA-144 fit on
          fit-room frames (predicted responses; there are no rat or human recordings of these views)
  hidden  a network's last hidden layer on the row's input       action  that network's action probabilities
Networks: priors (4 arms x seeds 1000-1002) and networks after moved-goal learning from the human_bc prior, seeds
1000-1002: ppo, td_head, s2_dfa, s2_ed, s2_head_strat.
Tests (the Rosetta authors' protocol: geometry_scores on 1,000 held-out rows vs shuffled; WassersteinProcrustes on
disjoint fit halves with 0 and 16 pairs; FOSCTTM on test pairs), 3 Rosetta seeds per cell:
  P  pairs: image-text, image-cortex, text-cortex; per network image-action, text-action, cortex-action, image-hidden,
     text-hidden, cortex-hidden, hidden-action
  B  three-way bridge (0 pairs): image->text and text->action fitted unpaired, composed, FOSCTTM image->action, vs the
     direct unpaired image->action; same with cortex->text->action
  N  network to network (as the paper's brain-to-brain NSD alignment): hidden of human_bc vs rat_bc / human_legal /
     cold_ppo (same seed), human_bc vs human_bc of another seed (control), human_bc prior vs its ppo / s2_dfa learned
     network, ppo-learned vs s2_dfa-learned
  python scripts/v4n_rosetta_brains_claude_20261103.py prereg | collect | features | jobs W | summary
"""
import gzip, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_rosetta_matched_claude_20261031 as R

OUT = R.BASE / "brains"; PROTOCOL = OUT / "PROTOCOL.json"; ARMS = R.ARMS; SEEDS = (1000, 1001, 1002); PER = 1000; RSEEDS = 3
LEARNED = ("ppo", "td_head", "s2_dfa", "s2_ed", "s2_head_strat")


def nets():
    """name -> (prior, seed, learned dir or None)."""
    import v4n_snn_learned_claude_20261103 as N
    out = {f"{a}/{s}": (a, s, None) for a in ARMS for s in SEEDS}
    out.update({f"{l}@human_bc/{s}": ("human_bc", s, N.DIRS[l]) for l in LEARNED for s in SEEDS}); return out


def load_net(prior, seed, d):
    import torch, v4n_moved_goal_learning_claude_20261031 as M, v4n_local_search2_claude_20261102 as S2
    m = M.build(prior, seed)
    if d is not None:
        src = d / prior / f"seed{seed}.json"; cfg = json.loads(src.read_text()).get("cfg")
        if cfg and cfg.get("expand"): S2.expand(m.net, cfg["expand"], cfg["kwta"], seed)
        m.net.load_state_dict({k: v.to(next(m.net.parameters()).device) for k, v in torch.load(src.with_suffix(".final.pt")).items()})
    return m.net


def collect():
    import torch, v4n_moved_goal_learning_claude_20261031 as M
    assert PROTOCOL.exists(); FR, cache = R._env(); starts = json.loads((M.worlds.fresh.BASE / "geometry_manifest.json").read_text())["starts"]; rows = []
    for a in ARMS:
        for s in SEEDS:
            net = load_net(a, s, None); dev = next(net.parameters()).device; g = torch.Generator().manual_seed([20261103, s, ARMS.index(a)][0] + s * 7 + ARMS.index(a)); ms = []
            for pair in range(8):
                sched = [dict(configuration=100 + 2 * pair + b, spawn=st, rotation_index=rot) for b in (0, 1) for rot in range(3) for st in starts]
                env = M.env_for(M.FR.MAN, {}, pair, 1, s, "clean", 45 * len(sched) // 2, sched); obs = env.reset()
                while not env.exhausted and env.frames < env.limit:
                    w = env.env.worlds[0]
                    if w is not None: ms.append(dict(arm=a, seed=s, cfg=int(w["configuration"]), cell=list(map(int, w["position"])), heading=list(map(int, w["heading"])),
                                                    rot=int(w["rotation_index"]), obs=obs[0].tolist()))
                    with torch.no_grad(): p = torch.softmax(net.pi(net.body(torch.as_tensor(obs, device=dev))), -1).cpu()
                    obs, _, _ = env.step(torch.multinomial(p, 1, generator=g).squeeze(-1).numpy())
            idx = np.random.default_rng([20261103, s, ARMS.index(a)]).permutation(len(ms))[:PER]; rows += [ms[i] for i in sorted(idx)]
            print(a, s, len(ms), flush=True)
    for r in rows: r["caption"] = R.caption(cache.worlds[r["cfg"]], tuple(r["cell"]), tuple(r["heading"]), lexfree=False)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "moments.json.gz").write_bytes(gzip.compress(json.dumps(rows).encode())); print(len(rows), "rows")


def features():
    import torch, mouse_twin_e1 as E1
    import v4n_microns_perception_claude_20261017 as MP
    from sklearn.decomposition import PCA
    rows = json.loads(gzip.decompress((OUT / "moments.json.gz").read_bytes())); FR, cache = R._env(); E = R.Encoders("cuda:0")
    keys = sorted({(r["cfg"], *r["cell"], *r["heading"], r["rot"]) for r in rows}); kid = {k: i for i, k in enumerate(keys)}
    frames = np.stack([R.render_frame(cache, k[0], k[1:3], k[3:5], k[5]) for k in keys])
    img = np.concatenate([E.image(list(frames[i:i + 128])) for i in range(0, len(keys), 128)])
    tw = E1.load_twin(); px = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    raw = np.concatenate([E1.twin_features(tw, MP.monitor(px[i:i + E1.BATCH])) for i in range(0, len(px), E1.BATCH)])
    fitk = np.array([k[0] in R.FIT_CFG for k in keys]); ctx = PCA(144, random_state=0).fit(raw[fitk]).transform(raw).astype(np.float32)
    caps = sorted({r["caption"] for r in rows}); txt = E.text(caps)
    fi = np.array([kid[(r["cfg"], *r["cell"], *r["heading"], r["rot"])] for r in rows]); ci = np.array([caps.index(r["caption"]) for r in rows])
    X = torch.as_tensor(np.array([r["obs"] for r in rows], np.float32)); H, P = {}, {}
    for name, (p, s, d) in nets().items():
        net = load_net(p, s, d); dev = next(net.parameters()).device
        with torch.no_grad(): h = net.body(X.to(dev)); H[name] = h.cpu().numpy(); P[name] = torch.softmax(net.pi(h), -1).cpu().numpy()
    np.savez_compressed(OUT / "features.npz", img=img[fi], txt=txt[ci], cortex=ctx[fi], **{f"hid:{k}": v for k, v in H.items()}, **{f"act:{k}": v for k, v in P.items()})
    print("features", img.shape, ctx.shape, raw.shape, len(H), "networks")


def cortex_full():
    """Amendment 1 (reviewer: '9K neurons reduced by PCA to 144 PCs? why?'): the full 9,941 predicted responses, no PCA,
    and the variance the source-free PCA-144 keeps."""
    import mouse_twin_e1 as E1, v4n_microns_perception_claude_20261017 as MP
    from sklearn.decomposition import PCA
    rows = json.loads(gzip.decompress((OUT / "moments.json.gz").read_bytes())); FR, cache = R._env()
    keys = sorted({(r["cfg"], *r["cell"], *r["heading"], r["rot"]) for r in rows}); kid = {k: i for i, k in enumerate(keys)}
    px = (np.clip(np.stack([R.render_frame(cache, k[0], k[1:3], k[3:5], k[5]) for k in keys]), 0, 1) * 255).astype(np.uint8); tw = E1.load_twin()
    raw = np.concatenate([E1.twin_features(tw, MP.monitor(px[i:i + E1.BATCH])) for i in range(0, len(px), E1.BATCH)]).astype(np.float32)
    fitk = np.array([k[0] in R.FIT_CFG for k in keys]); pca = PCA(144, random_state=0).fit(raw[fitk])
    fi = np.array([kid[(r["cfg"], *r["cell"], *r["heading"], r["rot"])] for r in rows]); np.save(OUT / "cortex_full.npy", raw[fi])
    ev = dict(n_neurons=int(raw.shape[1]), n_frames=int(len(keys)), pca144_variance_kept_fit=float(pca.explained_variance_ratio_.sum()),
              pca144_variance_kept_test=float(1 - ((raw[~fitk] - pca.inverse_transform(pca.transform(raw[~fitk]))) ** 2).sum() / ((raw[~fitk] - raw[fitk].mean(0)) ** 2).sum()))
    (OUT / "cortex_full_stats.json").write_text(json.dumps(ev, indent=1) + "\n"); print(ev)


def job_list_full():
    J = [("P", f"{x}-cortexfull", None, k) for x in ("image", "text", "cortex") for k in range(RSEEDS)]
    for n in nets():
        J += [("P", f"cortexfull-{y}", n, k) for y in ("act", "hid") for k in range(RSEEDS)]
    return J


def job_list():
    J = [("P", f"{x}-{y}", None, k) for x, y in (("image", "text"), ("image", "cortex"), ("text", "cortex")) for k in range(RSEEDS)]
    for n in nets():
        for x, y in (("image", "act"), ("text", "act"), ("cortex", "act"), ("image", "hid"), ("text", "hid"), ("cortex", "hid"), ("hid", "act")):
            J += [("P", f"{x}-{y}", n, k) for k in range(RSEEDS)]
    for a in ARMS:
        for s in SEEDS: J += [("B", br, f"{a}/{s}", k) for br in ("image", "cortex") for k in range(RSEEDS)]
    for s in SEEDS:
        prs = [(f"human_bc/{s}", f"{b}/{s}") for b in ("rat_bc", "human_legal_uniform", "cold_ppo")] + [(f"human_bc/{s}", f"human_bc/{SEEDS[(SEEDS.index(s) + 1) % 3]}")]
        prs += [(f"human_bc/{s}", f"ppo@human_bc/{s}"), (f"human_bc/{s}", f"s2_dfa@human_bc/{s}"), (f"ppo@human_bc/{s}", f"s2_dfa@human_bc/{s}")]
        J += [("N", f"{x}|{y}", None, k) for x, y in prs for k in range(RSEEDS)]
    return J


def run_job(job):
    import torch
    sys.path.insert(0, str(R.ROSETTA))
    from unpaired_rosetta import WassersteinProcrustes
    from unpaired_rosetta.geometry import geometry_scores
    from unpaired_rosetta.evaluation import disjoint_split, sample_pairs, retrieval_metrics
    from unpaired_rosetta.randomness import Randomness, SEEDS as RS
    from unpaired_rosetta.linalg import center_and_normalize as CN
    kind, pair, net, k = job; torch.set_num_threads(2)
    f = OUT / "jobs" / (hashlib.md5(json.dumps(job).encode()).hexdigest() + ".json")
    if f.exists(): return json.loads(f.read_text())
    rows = json.loads(gzip.decompress((OUT / "moments.json.gz").read_bytes())); F = np.load(OUT / "features.npz")
    def get(m):
        if m == "cortexfull": return torch.as_tensor(np.load(OUT / "cortex_full.npy"), dtype=torch.float32)
        return torch.as_tensor(F[{"image": "img", "text": "txt", "cortex": "cortex"}[m]] if m in ("image", "text", "cortex") else F[f"{m}:{net}"], dtype=torch.float32)
    fit = np.isin([r["cfg"] for r in rows], R.FIT_CFG); Rn = Randomness(RS[k]); fi = torch.as_tensor(np.where(fit)[0]); ti = torch.as_tensor(np.where(~fit)[0])
    ti = ti[Rn.permutation(len(ti))[:2000]]; ix, iy = disjoint_split(len(fi), Rn); out = dict(job=job, rosetta_seed=int(RS[k]))
    def wp(X, Y, npairs=0):
        m = WassersteinProcrustes(num_workers=1, num_threads=2)
        if npairs: pi = fi[sample_pairs(len(fi), npairs, Rn)]; m.fit(X[fi[ix]], Y[fi[iy]], X[pi], Y[pi], randomness=Rn)
        else: m.fit(X[fi[ix]], Y[fi[iy]], randomness=Rn)
        return m
    if kind in ("P", "N"):
        if kind == "P": X, Y = (get(m) for m in pair.split("-"))
        else: a, b = pair.split("|"); X, Y = torch.as_tensor(F[f"hid:{a}"], dtype=torch.float32), torch.as_tensor(F[f"hid:{b}"], dtype=torch.float32)
        ng = 1000; g = geometry_scores(X[ti[:ng]], Y[ti[:ng]], torch.Generator().manual_seed(RS[k]))
        gs = geometry_scores(X[ti[:ng]], Y[ti[:ng]][torch.randperm(ng, generator=torch.Generator().manual_seed(1))], torch.Generator().manual_seed(RS[k]))
        out.update(geometry=g, geometry_shuffled=gs)
        for npairs in (0, 16): out[f"pairs{npairs}"] = retrieval_metrics(wp(X, Y, npairs), X[ti], Y[ti], Rn)
    else:                                                                    # B: bridge through text, unpaired
        X, T, A = get(pair), get("text"), get("act"); m1, m2, md = wp(X, T), wp(T, A), wp(X, A)
        class Comp:                                                            # m1 lands in text's centred-normalised coords, m2's input space
            def similarity(self, q, keys): return (m1.transform(q) @ m2.weight) @ CN(keys.float(), m2.mean_y).T
        for nm, mm in (("composed", Comp()), ("direct", md)):
            try: out[nm] = retrieval_metrics(mm, X[ti], A[ti], Rn)
            except Exception as e: out[nm] = dict(error=repr(e))
    f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(out, default=float) + "\n"); return out


def geom_pass():
    """Schedule-only deviation (2026-11-03, disclosed): the registered geometry scores of every P/N job, computed first and
    exactly as in run_job (same rows, seeds, 1000 test moments, shuffled control); WP retrieval stays in run_job."""
    import torch
    sys.path.insert(0, str(R.ROSETTA))
    from unpaired_rosetta.geometry import geometry_scores
    from unpaired_rosetta.randomness import Randomness, SEEDS as RS
    rows = json.loads(gzip.decompress((OUT / "moments.json.gz").read_bytes())); F = np.load(OUT / "features.npz")
    fit = np.isin([r["cfg"] for r in rows], R.FIT_CFG); res = {}
    for kind, pair, net, k in [j for j in job_list() + job_list_full() if j[0] in ("P", "N")]:
        def get(m):
            if m == "cortexfull": return torch.as_tensor(np.load(OUT / "cortex_full.npy"), dtype=torch.float32)
            return torch.as_tensor(F[{"image": "img", "text": "txt", "cortex": "cortex"}[m]] if m in ("image", "text", "cortex") else F[f"{m}:{net}"], dtype=torch.float32)
        Rn = Randomness(RS[k]); ti = torch.as_tensor(np.where(~fit)[0]); ti = ti[Rn.permutation(len(ti))[:2000]]
        if kind == "P": X, Y = (get(m) for m in pair.split("-"))
        else: a, b = pair.split("|"); X, Y = torch.as_tensor(F[f"hid:{a}"], dtype=torch.float32), torch.as_tensor(F[f"hid:{b}"], dtype=torch.float32)
        ng = 1000; g = geometry_scores(X[ti[:ng]], Y[ti[:ng]], torch.Generator().manual_seed(RS[k]))
        gs = geometry_scores(X[ti[:ng]], Y[ti[:ng]][torch.randperm(ng, generator=torch.Generator().manual_seed(1))], torch.Generator().manual_seed(RS[k]))
        res[json.dumps([kind, pair, net, k])] = dict(geometry=g, geometry_shuffled=gs)
    (OUT / "geometry.json").write_text(json.dumps(res, default=float) + "\n"); print("geometry", len(res))


def jobs_rev(workers):
    """Second pool walking the same job list from the end (results are per-job files; identical either way)."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex: list(ex.map(run_job, (job_list_full() + job_list())[::-1]))
    print("jobs_rev done")


def jobs(workers, full=False):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex: list(ex.map(run_job, job_list_full() if full else job_list()))
    print("jobs done", len(job_list()))


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), outcomes_seen="matched-study stages A-E, lexfree, strata; none of this study",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), design=__doc__, n_jobs=len(job_list()), reading="descriptive, secondary"), indent=1) + "\n")
    print("registered", len(job_list()), "jobs")


if __name__ == "__main__":
    c = sys.argv[1]
    {"prereg": prereg, "collect": collect, "features": features, "cortex_full": cortex_full, "jobs": lambda: jobs(int(sys.argv[2])),
     "jobs_full": lambda: jobs(int(sys.argv[2]), True), "geom": geom_pass, "jobs_rev": lambda: jobs_rev(int(sys.argv[2]))}[c]()
