"""Where does the view determine the choice? Stratified Rosetta analysis (author 2026-11-03: "measure with the goal if
it's within view; don't forget text-action geometry"). Descriptive, registered as AMENDMENT6 before computing.
Same 40,000 moments, features and fit/test rooms as v4n_rosetta_matched (original template captions; the goal is never
rendered, fog hides everything beyond the adjacent module). Each moment is labelled from the room geometry:
  forced    exactly one open move other than going back      junction  two or more      dead end  none
  goal_near the goal is the agent's cell's open neighbour ahead / left / right (inside the visible range, not drawn)
  directed  the agent's move starts a shortest path to the goal
Per stratum (test rooms, all arms pooled and per arm): action accuracy of the room-held-out probes from image and from
text (fit on all fit-room moments, as stage B), majority-in-stratum, legal-uniform (1 / open moves incl. back), and the
Rosetta authors' geometry_scores (unbiased CKA, mutual kNN) image-action and text-action vs row-shuffled.
Plus a goal-bearing caption: the template caption + where the goal lies relative to the heading (ahead/left/right/behind
by the larger axis, and the grid distance): what a view WOULD say if the goal were visible.
  python scripts/v4n_rosetta_strata_claude_20261103.py
"""
import gzip, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_rosetta_matched_claude_20261031 as R

OUT = R.BASE / "strata"


def labels(rows, cache):
    from arc_fly.unimaze import bfs, NB
    dist = {}; lab = []
    for r in rows:
        w = cache.worlds[r["cfg"]]; g = tuple(cache.goals[r["cfg"]]); c, h = tuple(r["cell"]), tuple(r["heading"])
        if r["cfg"] not in dist: dist[r["cfg"]] = bfs(np.asarray(w, bool), g)
        D = dist[r["cfg"]]; ok = lambda q: 0 <= q[0] < 10 and 0 <= q[1] < 10 and w[q]
        back = (-h[0], -h[1]); opens = [d for d in map(tuple, NB) if ok((c[0] + d[0], c[1] + d[1]))]
        fwd = [d for d in opens if d != back]; vis = [h, (-h[1], h[0]), (h[1], -h[0])]
        near = any((c[0] + d[0], c[1] + d[1]) == g for d in vis)
        nd = {d: D[c[0] + d[0], c[1] + d[1]] for d in opens if D[c[0] + d[0], c[1] + d[1]] >= 0}
        best = {d for d, x in nd.items() if x == min(nd.values())} if nd else set()
        act = tuple(NB[r["action"]]); gv = (g[0] - c[0], g[1] - c[1])
        ax = (int(np.sign(gv[0])), 0) if abs(gv[0]) >= abs(gv[1]) else (0, int(np.sign(gv[1])))
        lab.append(dict(kind="dead end" if not fwd else ("forced" if len(fwd) == 1 else "junction"), goal_near=near, directed=act in best,
                        n_legal=len(opens), bearing=R.rel(h, ax) if ax != (0, 0) else "here", gdist=int(abs(gv[0]) + abs(gv[1]))))
    return lab


def main():
    import torch
    sys.path.insert(0, str(R.ROSETTA)); from unpaired_rosetta.geometry import geometry_scores
    rows = json.loads(gzip.decompress((R.BASE / "moments.json.gz").read_bytes())); F = np.load(R.BASE / "features.npz"); _, cache = R._env()
    L = labels(rows, cache); I, T = F["img"][F["frame_index"]], F["txt"][F["caption_index"]]
    enc = R.Encoders("cuda" if torch.cuda.is_available() else "cpu")                   # the study's own MPNet pooling
    gcap = [f"{r['caption']} The goal is {l['bearing']}, {l['gdist']} modules away." for r, l in zip(rows, L)]
    uniq = sorted(set(gcap)); E = enc.text(uniq); G = E[[uniq.index(c) for c in gcap]]
    P = np.array([r["probs"] for r in rows]); y = np.array([r["action"] for r in rows]); arm = np.array([r["arm"] for r in rows])
    fit = np.isin([r["cfg"] for r in rows], R.FIT_CFG); te = ~fit
    strata = {"all": np.ones(len(rows), bool), "forced": np.array([l["kind"] == "forced" for l in L]), "junction": np.array([l["kind"] == "junction" for l in L]),
              "junction, goal near": np.array([l["kind"] == "junction" and l["goal_near"] for l in L]),
              "junction, goal not near": np.array([l["kind"] == "junction" and not l["goal_near"] for l in L])}
    res = dict(share={k: float(v.mean()) for k, v in strata.items()}, directed_share={k: float(np.mean([L[i]["directed"] for i in np.where(v)[0]])) for k, v in strata.items()}, arms={})
    for a in R.ARMS + ("pooled",):
        m = (arm == a) if a != "pooled" else np.ones(len(rows), bool); out = {}
        preds = {}
        for nm, X in (("image", I), ("text", T), ("text+goal bearing", G)):
            _, _, pr = R._probe(X[m & fit], y[m & fit], X[m & te]); preds[nm] = pr.argmax(1)
        yt = y[m & te]
        for k, s in strata.items():
            st = s[m & te]
            if st.sum() < 50: continue
            o = {nm: float((p[st] == yt[st]).mean()) for nm, p in preds.items()}
            o["majority"] = float(np.bincount(yt[st], minlength=4).max() / st.sum()); o["legal_uniform"] = float(np.mean([1 / L[i]["n_legal"] for i in np.where(m & te)[0][st]]))
            o["n"] = int(st.sum())
            if a == "pooled" or True:
                idx = np.where(m & te)[0][st]; idx = np.random.default_rng(0).permutation(idx)[:1000]
                for pair, X in (("image-action", I), ("text-action", T), ("text+goal-action", G)):
                    Xt, Yt = torch.as_tensor(X[idx], dtype=torch.float32), torch.as_tensor(P[idx], dtype=torch.float32)
                    g = geometry_scores(Xt, Yt, torch.Generator().manual_seed(0)); gs = geometry_scores(Xt, Yt[torch.randperm(len(idx), generator=torch.Generator().manual_seed(1))], torch.Generator().manual_seed(0))
                    o[pair] = dict(cka=float(g["cka"]), mknn=float(g["mutual_knn"]), cka_shuf=float(gs["cka"]), mknn_shuf=float(gs["mutual_knn"]))
            out[k] = o
        res["arms"][a] = out
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "strata.json").write_text(json.dumps(res, indent=1) + "\n")
    print("share", {k: round(v, 3) for k, v in res["share"].items()}); print("directed", {k: round(v, 2) for k, v in res["directed_share"].items()})
    for a in ("pooled",) + R.ARMS:
        for k, o in res["arms"][a].items():
            print(f"{a:20s} {k:24s} n={o['n']:5d} img {o['image']:.2f} txt {o['text']:.2f} txt+goal {o['text+goal bearing']:.2f} maj {o['majority']:.2f} legal {o['legal_uniform']:.2f} | "
                  + " ".join(f"{p} CKA {o[p]['cka']:.2f}({o[p]['cka_shuf']:.2f}) kNN {o[p]['mknn']:.3f}({o[p]['mknn_shuf']:.3f})" for p in ("image-action", "text-action", "text+goal-action")))


if __name__ == "__main__":
    main()
