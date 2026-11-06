"""Matched-content image-text-action study (reviewer 6g1M; camera-ready supplement; author-approved 2026-10-31).
The reviewed Rosetta analysis paired images (current view) with text GENERATED FROM the action plan, so text-action
geometry was circular and modalities carried unequal information. Here every modality describes the SAME moment:
  image   DINOv2-base CLS of the rendered first-person frame (64x36, upsampled)
  text    a caption of what is VISIBLE in that frame, never of the action: exact template from geometry (primary);
          Qwen2.5-VL-3B caption of the frame (secondary)  -> MPNet sentence embedding
  action  the frozen agent's own action probabilities (and its last hidden layer)
Moments: decisions of frozen agents (human_bc, human_legal_uniform, rat_bc, cold_ppo; seeds 1000-1004) in the 16 fresh
confirmation rooms, re-rendered from the registered records (no new probes). Rooms 100-107 = fit, 108-115 = test.
Tests (per arm x seed):
  A  linear CKA image-text, text-action, image-action; 100 row shuffles; untrained-encoder controls
  B  room-held-out ridge probes: predict the agent's action from image and from text (accuracy, chance, majority)
  C  composition on disjoint data: image->text ridge map fit on fit-rooms moments of seeds 1000-1002, text->action
     probe fit on fit-rooms of seeds 1003-1004; composed image->text->action evaluated on test rooms
  D  closed loop in test rooms (familiar goal, 30 trials/room): the composed map drives the agent; vs its own head
     and a uniform policy
  python scripts/v4n_rosetta_matched_claude_20261031.py check | prereg | moments | features | analyse | closed W N
"""
import gzip, hashlib, json, os, re, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]; BASE = ROOT / "data/audit/v4n_rosetta_matched_20261031"; PROTOCOL = BASE / "PROTOCOL.json"
LEXFREE = os.environ.get("V4N_CAPTION") == "lexfree"          # AMENDMENT3: captions share no word with the action names
OUT = BASE / "lexfree" if LEXFREE else BASE
ACTION_WORDS = ("down", "up", "right", "left", "forward", "backward", "back", "north", "south", "east", "west", "turn", "move", "go")
ARMS = ("human_bc", "human_legal_uniform", "rat_bc", "cold_ppo"); SEEDS = (1000, 1001, 1002, 1003, 1004)
FIT_CFG, TEST_CFG = tuple(range(100, 108)), tuple(range(108, 116)); NAMES = ("down", "up", "right", "left")   # NB order: (1,0),(-1,0),(0,1),(0,-1)
NORTH = (-1, 0)                                                                                                  # black curtain on the row-0 edge


def rel(h, d):
    """Where direction d lies relative to heading h (row/col grid; left of h = (-h_col, h_row))."""
    left, right, back = (-h[1], h[0]), (h[1], -h[0]), (-h[0], -h[1])
    return {tuple(h): "ahead", left: "to the left", right: "to the right", back: "behind"}[tuple(d)]


LEX = {"ahead": "in front", "to the left": "on the anticlockwise side", "to the right": "on the clockwise side"}


def caption(walk, cell, h, lexfree=None):
    """What is visible: the adjacent module ahead, left and right (fog hides the rest) and the curtain's bearing.
    lexfree: same content, worded without any action name (left/right -> anticlockwise/clockwise side)."""
    lexfree = LEXFREE if lexfree is None else lexfree
    if lexfree:
        def side_(d):
            q = (cell[0] + d[0], cell[1] + d[1]); return "open floor" if 0 <= q[0] < 10 and 0 <= q[1] < 10 and walk[q] else "a drop with no floor"
        l_, r_ = (-h[1], h[0]), (h[1], -h[0]); cur = rel(h, NORTH)
        cur_txt = "The black curtain is behind you, out of view." if cur == "behind" else f"The black curtain is {LEX[cur]}."
        return f"In front there is {side_(h)}. On the anticlockwise side there is {side_(l_)}. On the clockwise side there is {side_(r_)}. {cur_txt}"
    def side(d):
        q = (cell[0] + d[0], cell[1] + d[1]); return "open floor" if 0 <= q[0] < 10 and 0 <= q[1] < 10 and walk[q] else "a drop with no floor"
    left, right = (-h[1], h[0]), (h[1], -h[0]); cur = rel(h, NORTH)
    cur_txt = "The black curtain is behind you, out of view." if cur == "behind" else f"The black curtain is {cur}."
    return f"Ahead there is {side(h)}. To the left there is {side(left)}. To the right there is {side(right)}. {cur_txt}"


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), outcomes_seen=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), design=__doc__,
        primary=dict(P1="B: text->action held-out accuracy minus chance (per arm, paired over 5 seeds)",
                     P2="C: composed image->text->action agreement with the agent's own action vs a shuffled-text-map control",
                     P3="D: closed-loop success of the composed map vs uniform (13.3%) and vs the agent's own head"),
        secondary="A CKA with shuffle/untrained controls; VLM-caption replication; arm comparisons (human_bc vs others) descriptive",
        reading="matched-content linkage is supported if P1-P3 exceed controls; otherwise reported as not supported"), indent=1) + "\n")
    print("registered")


def check():
    w = np.ones((10, 10), bool); w[4, 6] = False
    c = caption(w, (5, 5), (-1, 0))                                      # facing north at (5,5): ahead (4,5) open, left (5,4), right (5,6)
    assert "Ahead there is open floor" in c and "black curtain is ahead" in c, c
    c = caption(w, (4, 5), (0, 1))                                       # facing east: ahead (4,6) removed; curtain (north) to the left
    assert "Ahead there is a drop" in c and "curtain is to the left" in c, c
    assert rel((1, 0), NORTH) == "behind" and rel((0, -1), NORTH) == "to the right"
    import itertools
    for wk in (w, ~w):
        for cell, h in itertools.product([(r, c) for r in range(10) for c in range(10)], [(1, 0), (-1, 0), (0, 1), (0, -1)]):
            words = set(re.findall(r"[a-z]+", caption(wk, cell, h, lexfree=True).lower()))
            assert not words & set(ACTION_WORDS), (cell, h, words & set(ACTION_WORDS))
    print("lexfree ok |", caption(w, (4, 5), (0, 1), lexfree=True))
    print("check ok |", c)


# ---------------------------------------------------------------- implementation
def _env():
    import fresh_rooms_confirm_claude as FR
    _, _, source = FR.worlds.fresh.load(); cache = FR.worlds.RoomCache(source, json.loads(FR.MAN.read_text()))
    return FR, cache


def moments(per=2000):
    """Decision moments from the registered familiar-goal records (no new probes)."""
    from arc_fly.unimaze import NB
    FR, cache = _env(); rows = []
    for a in ARMS:
        for sd in SEEDS:
            ms = []
            for f in sorted((FR.D / "cohort/main" / a / f"seed{sd}").glob("pair*.full.json.gz")):
                for e in json.loads(gzip.decompress(f.read_bytes()))["episodes"]:
                    if "spawn" not in e: continue
                    h = NB[FR.worlds.fresh.source.inward_heading(tuple(e["spawn"]))]
                    for t, act in enumerate(e["actions"]):
                        if t: h = NB[e["actions"][t - 1]]
                        ms.append(dict(arm=a, seed=sd, cfg=e["configuration"], cell=list(e["path"][t]), heading=list(h), rot=e["rotation_index"],
                                       action=int(act), probs=[float(x) for x in e["probabilities"][t]]))
            idx = np.random.default_rng([20261031, sd, ARMS.index(a)]).permutation(len(ms))[:per]; rows += [ms[i] for i in sorted(idx)]
    for r in rows: r["caption"] = caption(cache.worlds[r["cfg"]], tuple(r["cell"]), tuple(r["heading"]))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "moments.json.gz").write_bytes(gzip.compress(json.dumps(rows).encode()))
    print(len(rows), "moments;", len({r["caption"] for r in rows}), "distinct captions")


def render_frame(cache, cfg, cell, heading, rot):
    import fresh_rooms_confirm_claude as FR
    return FR.worlds.fresh.render(cache.view, cache.worlds[cfg], tuple(cell), tuple(heading), cache.rotations[rot])[0]


class Encoders:
    def __init__(self, dev="cuda:0"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from transformers import AutoTokenizer
        self.torch, self.dev = torch, dev
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2"); self.mp = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(dev).eval()
        self.proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base"); self.dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()

    def image(self, frames):
        with self.torch.no_grad():
            x = self.proc(images=[np.repeat((np.clip(f, 0, 1) * 255).astype(np.uint8)[..., None], 3, -1) for f in frames], return_tensors="pt").to(self.dev)
            return self.dino(**x).last_hidden_state[:, 0].float().cpu().numpy()

    def text(self, caps):
        """all-mpnet-base-v2 sentence embedding: attention-masked mean pooling, L2-normalised (its standard pooling)."""
        out = []
        with self.torch.no_grad():
            for i in range(0, len(caps), 256):
                b = self.tok(caps[i:i + 256], padding=True, truncation=True, return_tensors="pt").to(self.dev); h = self.mp(**b).last_hidden_state
                m = b["attention_mask"][..., None].float(); e = (h * m).sum(1) / m.sum(1); out.append(self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out)


def features():
    FR, cache = _env(); rows = json.loads(gzip.decompress((OUT / "moments.json.gz").read_bytes())); E = Encoders()
    keys = sorted({(r["cfg"], *r["cell"], *r["heading"], r["rot"]) for r in rows}); kid = {k: i for i, k in enumerate(keys)}
    img = np.concatenate([E.image([render_frame(cache, k[0], k[1:3], k[3:5], k[5]) for k in keys[i:i + 128]]) for i in range(0, len(keys), 128)])
    caps = sorted({r["caption"] for r in rows}); cid = {c: i for i, c in enumerate(caps)}; txt = E.text(caps)
    np.savez_compressed(OUT / "features.npz", img=img, txt=txt, frame_index=np.array([kid[(r["cfg"], *r["cell"], *r["heading"], r["rot"])] for r in rows]),
                        caption_index=np.array([cid[r["caption"]] for r in rows]))
    print("frames", len(keys), "captions", len(caps))


def _cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0); hx, hy = X @ X.T, Y @ Y.T
    return float((hx * hy).sum() / np.sqrt((hx * hx).sum() * (hy * hy).sum()))


def _probe(Xtr, ytr, Xte, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr); return m, sc, m.predict_proba(sc.transform(Xte))


def analyse():
    from sklearn.linear_model import Ridge
    rows = json.loads(gzip.decompress((OUT / "moments.json.gz").read_bytes())); F = np.load(OUT / "features.npz")
    I, T = F["img"][F["frame_index"]], F["txt"][F["caption_index"]]; P = np.array([r["probs"] for r in rows]); y = np.array([r["action"] for r in rows])
    arm = np.array([r["arm"] for r in rows]); seed = np.array([r["seed"] for r in rows]); fit = np.isin([r["cfg"] for r in rows], FIT_CFG)
    rng = np.random.default_rng(0); out = dict(A={}, B={}, C={})
    for a in ARMS:
        for s in SEEDS:
            m = (arm == a) & (seed == s); k = f"{a}/{s}"; sub = rng.choice(np.where(m)[0], min(1500, m.sum()), replace=False)
            ck = dict(img_txt=_cka(I[sub], T[sub]), txt_act=_cka(T[sub], P[sub]), img_act=_cka(I[sub], P[sub]))
            sh = [_cka(T[sub], P[rng.permutation(sub)]) for _ in range(20)]; ck["txt_act_shuffle_p95"] = float(np.percentile(sh, 95))
            out["A"][k] = ck
            tr, te = m & fit, m & ~fit
            b = {}
            for nm, X in (("image", I), ("text", T)):
                _, _, pr = _probe(X[tr], y[tr], X[te]); b[nm] = float((pr.argmax(1) == y[te]).mean())
            b["chance"] = .25; b["majority"] = float(np.bincount(y[te], minlength=4).max() / te.sum()); b["own_head_argmax"] = float((P[te].argmax(1) == y[te]).mean())
            out["B"][k] = b
        # C: composition on disjoint seeds
        A1 = (arm == a) & np.isin(seed, SEEDS[:3]) & fit; A2 = (arm == a) & np.isin(seed, SEEDS[3:]) & fit; te = (arm == a) & ~fit
        mp = Ridge(alpha=10.).fit(I[A1], T[A1]); pm, sc, _ = _probe(T[A2], y[A2], T[A2][:1])
        comp = pm.predict_proba(sc.transform(mp.predict(I[te])))
        ys = rng.permutation(y[A2]); pm0, sc0, _ = _probe(T[A2], ys, T[A2][:1]); ctrl = pm0.predict_proba(sc0.transform(mp.predict(I[te])))
        out["C"][a] = dict(composed=float((comp.argmax(1) == y[te]).mean()), shuffled_control=float((ctrl.argmax(1) == y[te]).mean()),
                           text_map_r2=float(mp.score(I[te], T[te])), n_test=int(te.sum()))
        import pickle; pickle.dump(dict(map=mp, probe=pm, scaler=sc), open(OUT / f"composed_{a}.pkl", "wb"))
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    for a in ARMS:
        B = [out["B"][f"{a}/{s}"] for s in SEEDS]; A_ = [out["A"][f"{a}/{s}"] for s in SEEDS]
        print(a, "B img %.3f txt %.3f maj %.3f" % tuple(np.mean([b[k] for b in B]) for k in ("image", "text", "majority")),
              "| A it %.2f ta %.2f ia %.2f" % tuple(np.mean([x[k] for x in A_]) for k in ("img_txt", "txt_act", "img_act")), "| C", out["C"][a])


def closed(w, n):
    """D: the composed image->text->action map drives the agent in the test rooms (familiar goal, sampled actions)."""
    import pickle, torch
    from arc_fly.unimaze import NB
    FR, cache = _env(); E = Encoders(); W = FR.worlds
    jobs = [(a, s, cfg) for a in ARMS for s in SEEDS for cfg in TEST_CFG][w::n]
    starts = json.loads(FR.MAN.read_text())["starts"]
    for a, s, cfg in jobs:
        f = OUT / "closed" / a / f"seed{s}" / f"cfg{cfg}.json"
        if f.exists(): continue
        C = pickle.load(open(OUT / f"composed_{a}.pkl", "rb")); gen = np.random.default_rng([s, cfg]); succ = 0; lens = []
        for rot in range(3):
            for sp in starts:
                cell, h = tuple(sp), NB[W.fresh.source.inward_heading(tuple(sp))]; walk = cache.worlds[cfg]
                for t in range(45):
                    z = C["map"].predict(E.image([render_frame(cache, cfg, cell, h, rot)])); p = C["probe"].predict_proba(C["scaler"].transform(z))[0]
                    act = int(gen.choice(4, p=p / p.sum())); q = (cell[0] + NB[act][0], cell[1] + NB[act][1]); h = NB[act]
                    if 0 <= q[0] < 10 and 0 <= q[1] < 10 and walk[q]: cell = q
                    if cell == (3, 3): succ += 1; lens.append(t + 1); break
        f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(dict(arm=a, seed=s, cfg=cfg, goals=succ, trials=30, lengths=lens)) + "\n")
    print(json.dumps(dict(worker=w, done=len(jobs))), flush=True)


ROSETTA = ROOT / "data/audit/sota_reproduction_20261017/sources/unpaired-rosetta"      # upstream commit fdfad84, unmodified


def rosetta_one(job):
    """Their protocol, their code: geometry_scores (unbiased CKA, mutual kNN k=10, TSI, QSI) on held-out rows; unpaired
    WassersteinProcrustes on disjoint halves of fit-room rows (0 pairs, and 16 known pairs); FOSCTTM on test-room pairs."""
    import torch
    sys.path.insert(0, str(ROSETTA))
    from unpaired_rosetta import WassersteinProcrustes
    from unpaired_rosetta.geometry import geometry_scores
    from unpaired_rosetta.evaluation import disjoint_split, sample_pairs, retrieval_metrics
    from unpaired_rosetta.randomness import Randomness, SEEDS as RS
    pair, arm, seed, k = job; torch.set_num_threads(2)
    cache = OUT / "rosetta_jobs" / f"{pair}_{arm}_{seed}_{k}.json"                   # per-job result: a crash loses nothing
    if cache.exists(): return json.loads(cache.read_text())
    rows = json.loads(gzip.decompress((OUT / "moments.json.gz").read_bytes())); F = np.load(OUT / "features.npz")
    mod = dict(image=F["img"][F["frame_index"]], text=F["txt"][F["caption_index"]], action=np.array([r["probs"] for r in rows]))
    sel = np.array([(arm is None or r["arm"] == arm) and (seed is None or r["seed"] == seed) for r in rows])
    fit = np.isin([r["cfg"] for r in rows], FIT_CFG) & sel; te = ~np.isin([r["cfg"] for r in rows], FIT_CFG) & sel
    X, Y = (torch.as_tensor(mod[m], dtype=torch.float32) for m in pair.split("-"))
    R = Randomness(RS[k]); fi = torch.as_tensor(np.where(fit)[0]); ti = torch.as_tensor(np.where(te)[0])
    ti = ti[R.permutation(len(ti))[:2000]]; ng = min(1000, len(ti))         # AMENDMENT4: some arm/seed cells have <1000 test rows
    g = geometry_scores(X[ti[:ng]], Y[ti[:ng]], torch.Generator().manual_seed(RS[k]))
    gs = geometry_scores(X[ti[:ng]], Y[ti[:ng]][torch.randperm(ng, generator=torch.Generator().manual_seed(1))], torch.Generator().manual_seed(RS[k]))
    ix, iy = disjoint_split(len(fi), R); out = dict(pair=pair, arm=arm, seed=seed, rosetta_seed=int(RS[k]), n_geometry=ng, geometry=g, geometry_shuffled=gs)
    for npairs in (0, 16):
        wp = WassersteinProcrustes(num_workers=1, num_threads=2)
        if npairs:
            pi = fi[sample_pairs(len(fi), npairs, R)]; wp.fit(X[fi[ix]], Y[fi[iy]], X[pi], Y[pi], randomness=R)
        else: wp.fit(X[fi[ix]], Y[fi[iy]], randomness=R)
        out[f"pairs{npairs}"] = retrieval_metrics(wp, X[ti], Y[ti], R)
    cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(out, default=float) + "\n")
    return out


def rosetta(workers=4):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [("image-text", None, None, k) for k in range(5)] + [(p, a, s, i) for p in ("text-action", "image-action") for a in ARMS for i, s in enumerate(SEEDS)]
    with ProcessPoolExecutor(workers) as ex: res = list(ex.map(rosetta_one, jobs))
    (OUT / "rosetta_protocol.json").write_text(json.dumps(res, indent=1) + "\n")
    for p in ("image-text", "text-action", "image-action"):
        for a in ((None,) if p == "image-text" else ARMS):
            rr = [r for r in res if r["pair"] == p and r["arm"] == a]
            print(p, a, "CKA %.2f (shuf %.2f) mkNN %.2f | FOSCTTM 0p %.3f 16p %.3f" % (np.mean([r["geometry"]["cka"] for r in rr]), np.mean([r["geometry_shuffled"]["cka"] for r in rr]),
                  np.mean([r["geometry"]["mutual_knn"] for r in rr]), np.mean([r["pairs0"]["foscttm"] for r in rr]), np.mean([r["pairs16"]["foscttm"] for r in rr])))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "closed": closed(int(sys.argv[2]), int(sys.argv[3]))
    else: {"check": check, "prereg": prereg, "moments": moments, "features": features, "analyse": analyse, "rosetta": lambda: rosetta(int(os.environ.get("V4N_WORKERS", 4)))}[c]()
