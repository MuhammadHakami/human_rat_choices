"""Brain geometry in the animals' own mazes (camera-ready addition, author 2026-11-07: "add both one rat and one human ... where
our already trained tartarus maze rats can actually navigate well and we do our ai/human/rat brain geometry and analysis there
while the paper stays the same simply adding more rigorous brain geometry to ai behaviors given exact same setting").
Registered before any agent is run in these mazes. No training: the paper's frozen and learned networks.

Mazes (built in our 10 x 10 arena, same renderer, same four moves; non-track cells are removed modules, as in Tartarus):
  W-track  (rat)   Shin & Jadhav, DANDI 000978 (CC BY 4.0): 8 rats, dorsal CA1 + PFC single units, W-track alternation learned in
                   one day. Arms 40 cm apart, 80-90 cm long: at 0.2 m modules three arms two cells apart (cols 2, 4, 6, rows 2-5)
                   joined by the bottom corridor (row 6, cols 2-6), wells at the arm tops. Each rat's positions are mapped by its own
                   arm centres and track ends (camera frames differ), then snapped to the nearest track cell.
  Towns    (human) Seger, Lega & Ekstrom 2023, Zenodo 8058006 (CC BY 4.0): 11 patients, hippocampal iEEG, store-to-store
                   navigation in two virtual towns (6 stores each). Streets rasterized at 14-15 units per cell (registered
                   rasterizer below; overlap with the visited street area 0.66 / 0.55: thin blocks merge), padded to 10 x 10.
Same states: each recorded trial (cell sequence; consecutive duplicate cells collapsed, gaps filled by the shortest track path)
is replayed through the registered environment with the recorded moves, so every network receives the observation the animal's
or patient's position and heading produce (episode = trial; split every 45 moves). Model features per step: last hidden layer
(64), action probabilities; per state: image (DINOv2 CLS of the rendered view) and text (MPNet of the template caption).
Neural features per step (the time the subject occupied that cell): rat unit firing rates (CA1, PFC, all); human per-electrode
log band power (2-4, 4-8, 8-12, 12-30, 30-70, 70-110 Hz; Hilbert envelope, z-scored per electrode).
Conditions: (cell, heading) with >= 4 visits in each half (odd/even trials). RDMs: 1 - Pearson r between condition means.
Outcomes (descriptive): per subject Spearman rho(model RDM, neural RDM); partial rho controlling for the space RDM (track distance
+ heading change) and the pixel RDM (pooled view); mean over subjects with t CI; split-half noise ceiling; label permutation
(1,000). Behaviour on the same states: probability each network gives the subject's next move.
Networks: frozen human_bc, rat_bc, human_legal_uniform, cold_ppo; learned ppo, bl3_none, bl7_none from human_bc; seeds 1000-1029.
  check | prereg | collect | analyse
"""
import json, os, sys, time, hashlib, glob, collections
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_shared_mazes_claude_20261107 as SM
B = SM.B; G = SM.G; M = SM.M; F = SM.F; worlds = SM.worlds
ROOT = M.ROOT; OUT = ROOT / "data/audit/v4n_brain_mazes_20261107"; PROTOCOL = OUT / "PROTOCOL.json"
RAT = ROOT / "data/real_sources/shin_jadhav_dandi000978"; HUM = ROOT / "data/real_sources/seger2023_ieeg/unz/files"
SEEDS = tuple(range(1000, 1030)); FROZEN = ("human_bc", "rat_bc", "human_legal_uniform", "cold_ppo"); LEARNED = ("ppo", "bl3", "bl7")
MINK = 8; BANDS = ((2, 4), (4, 8), (8, 12), (12, 30), (30, 70), (70, 110)); MINV = 4; NB = ((1, 0), (-1, 0), (0, 1), (0, -1))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------- mazes ----------------
def wtrack():
    w = np.zeros((10, 10), bool); w[2:6, [2, 4, 6]] = True; w[6, 2:7] = True
    return dict(configuration=300, name="W-track", walkable=w, wells={1: (2, 4), 2: (2, 2), 3: (2, 6)})


def town_layout(env):
    """Registered rasterizer: square cells s in [10, 16], offsets 1-unit steps, cell walkable if > 25% of its area was visited;
    the (s, offset) with the largest overlap (IoU) with the 1-unit visited mask, grid <= 10 x 10; padded to 10 x 10 at the top."""
    import scipy.io as sio
    L = []; S = {}
    for f in sorted(glob.glob(str(HUM / "electrode_*.mat"))):
        e = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["electrode"]
        if e.info.env_version != env: continue
        for ti, p in zip(e.data.nav_and_sim_trials.trial_info, e.data.nav_and_sim_trials.nav_position):
            p = np.asarray(p, float); p = p[~np.isnan(p).any(1)]
            if len(p): L.append(p); S.setdefault(ti.start_store, []).append(p[0]); S.setdefault(ti.end_store, []).append(p[-1])
    P = np.concatenate(L); lo = P.min(0); W = int(np.ceil(P[:, 0].max() - lo[0])) + 1; H = int(np.ceil(P[:, 1].max() - lo[1])) + 1
    Mk = np.zeros((H, W), bool); ij = (P - lo).astype(int); Mk[ij[:, 1], ij[:, 0]] = True; yy, xx = np.nonzero(Mk); best = None
    for s in np.arange(10, 16.01, .25):
        for ox in np.arange(0, s, 1.0):
            for oy in np.arange(0, s, 1.0):
                nx, ny = int(np.ceil((W + ox) / s)), int(np.ceil((H + oy) / s))
                if nx > 10 or ny > 10: continue
                Gd = np.zeros((ny, nx)); np.add.at(Gd, (((yy + oy) // s).astype(int), ((xx + ox) // s).astype(int)), 1); g = Gd / (s * s) > .25
                up = g[((np.arange(H) + oy) // s).astype(int)][:, ((np.arange(W) + ox) // s).astype(int)]; iou = (up & Mk).sum() / (up | Mk).sum()
                if best is None or iou > best[0]: best = (iou, s, ox, oy, g)
    iou, s, ox, oy, g = best; pad = 10 - g.shape[0]; w = np.zeros((10, 10), bool); w[pad:, :g.shape[1]] = g[::-1]   # north (larger y) at row 0
    def cell(xy): r = int((xy[1] - lo[1] + oy) // s); c = int((xy[0] - lo[0] + ox) // s); return (pad + g.shape[0] - 1 - r, c)
    return dict(configuration=301 if env == "env1" else 302, name=f"town {env}", walkable=w, iou=float(iou), cell_units=float(s),
                stores={k: cell(np.median(v, 0)) for k, v in S.items()}, cell=cell)


def snap(w, rc):
    """nearest walkable cell to a fractional (row, col)"""
    cand = np.argwhere(w); d = ((cand - np.asarray(rc)) ** 2).sum(1); return tuple(map(int, cand[np.argmin(d)]))


def track_path(w, a, b):
    """shortest 4-neighbour walkable path a -> b (BFS), excluding a"""
    prev = {a: None}; q = collections.deque([a])
    while q:
        x = q.popleft()
        if x == b: break
        for dr, dc in NB:
            y = (x[0] + dr, x[1] + dc)
            if 0 <= y[0] < 10 and 0 <= y[1] < 10 and w[y] and y not in prev: prev[y] = x; q.append(y)
    out = []; x = b
    while x != a: out.append(x); x = prev[x]
    return out[::-1]


def to_steps(w, cells, times):
    """collapse repeats, fill gaps; returns [(cell, t_enter, t_exit)]"""
    seq = []
    for c, t in zip(cells, times):
        if not seq: seq.append([c, t, t]); continue
        if c == seq[-1][0]: seq[-1][2] = t; continue
        for k, y in enumerate(track_path(w, seq[-1][0], c)):
            seq.append([y, t, t])
    return [tuple(x) for x in seq]


# ---------------- subjects ----------------
def rats():
    W = wtrack(); out = []
    for f in sorted(RAT.glob("*.npz")):
        d = np.load(f); P, t = d["pos"][:, :2], d["pos_t"]; m = np.zeros(len(t), bool)
        for a, b in zip(d["trials__start_time"], d["trials__stop_time"]): m |= (t >= a) & (t <= b)
        x, y = P[m, 0], P[m, 1]; top = y > np.percentile(y, 60)
        from sklearn.cluster import KMeans
        cx = np.sort(KMeans(3, n_init=10, random_state=0).fit(x[top, None]).cluster_centers_.ravel())
        y0, y1 = np.percentile(y, 2), np.percentile(y, 98)                                       # bottom corridor, wells
        col = lambda xx: 2 + 4 * (xx - cx[0]) / (cx[2] - cx[0]); row = lambda yy: 6 - 4 * (yy - y0) / (y1 - y0)
        spk = [d["spike_times"][(d["spike_index"][i - 1] if i else 0):d["spike_index"][i]] for i in range(len(d["spike_index"]))]
        region = d["electrode_location"][d["unit_electrodes"]]; trials = []
        for i, (a, b) in enumerate(zip(d["trials__start_time"], d["trials__stop_time"])):
            s = (t >= a) & (t <= b) & ~np.isnan(P).any(1)
            if s.sum() < 5: continue
            cells = [snap(W["walkable"], (row(yy), col(xx))) for xx, yy in P[s]]
            trials.append(dict(steps=to_steps(W["walkable"], cells, t[s]), correct=float(d["trials__correct"][i])))
        out.append(dict(species="rat", id=f.stem.split("_")[0], maze=W, trials=trials, spikes=spk, region=region, arms_x=cx.tolist()))
    return out


def humans():
    import scipy.io as sio
    from scipy.signal import butter, sosfiltfilt, hilbert
    T = {e: town_layout(e) for e in ("env1", "env2")}; by = collections.defaultdict(list)
    for f in sorted(glob.glob(str(HUM / "electrode_*.mat"))):
        e = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["electrode"]; by[(e.info.subject_ID, e.info.env_version)].append((e.info.anat_loc, e.data.nav_and_sim_trials))
    out = []
    for (sid, env), els in sorted(by.items()):
        Tw = T[env]; d0 = els[0][1]; trials = []; feats = []                                     # trials are shared by the subject's electrodes
        for i, (ti, p) in enumerate(zip(d0.trial_info, d0.nav_position)):
            p = np.asarray(p, float); ok = ~np.isnan(p).any(1); idx = np.flatnonzero(ok)
            if len(idx) < 5 or ti.catch: continue
            cells = [snap(Tw["walkable"], Tw["cell"](xy)) for xy in p[idx]]; trials.append(dict(steps=to_steps(Tw["walkable"], cells, idx), trial=i))
        for loc, d in els:                                                                       # per electrode: z-scored log band power
            Pw = []
            for x in d.nav_eeg:
                x = np.asarray(x, float); Pw.append(np.stack([np.log(np.abs(hilbert(sosfiltfilt(butter(4, b, "bandpass", fs=250, output="sos"), x))) ** 2 + 1e-12) for b in BANDS], 1))
            allp = np.concatenate(Pw); mu, sd = allp.mean(0), allp.std(0) + 1e-9; feats.append([(q - mu) / sd for q in Pw])
        out.append(dict(species="human", id=f"S{sid}", maze=Tw, trials=trials, power=feats, locations=[l for l, _ in els]))
    return out


# ---------------- replay: the environment's own sensing on the recorded moves ----------------
def manifest(mazes):
    rows = [dict(configuration=m["configuration"], pair=0, stage="A", walkable=m["walkable"].astype(int).tolist(), A_goal=[0, 0], B_goal=[0, 0]) for m in mazes]
    return dict(starts=[[0, 0]], rows=rows)


class Replayer:
    """VisualRooms/ContinuousRooms sensing (frame, contact, reward, last move, LogBelief place filter) without goal termination."""
    def __init__(self, mazes):
        _, _, self.source = worlds.fresh.load(); self.cache = worlds.RoomCache(self.source, manifest(mazes))
        self.atlas = self.source.raw.features[0, :, :, :, 0, :64].reshape(100, 4, 64)

    def episode(self, cfg, cells, rot=0):
        """cells: consecutive 4-neighbour cells; returns obs (len(cells), 252), headings, actions taken from each cell (-1 last)."""
        filt = worlds.fresh.LogBelief(self.atlas, .03); X, H, A = [], [], []
        heading = NB[worlds.fresh.source.inward_heading(cells[0])]; last = None
        for k, c in enumerate(cells):
            if k:
                d = (c[0] - cells[k - 1][0], c[1] - cells[k - 1][1]); last = NB.index(d); heading = d
            act = np.zeros(4, np.float32)
            if last is not None: act[last] = 1
            x = np.concatenate((self.cache.frame(cfg, c, heading, rot), np.array([0., 0.], np.float32), act))
            p = filt.observe(x[:64], last, False).astype(np.float32)
            X.append(np.concatenate((x, p, p @ worlds.fresh.source.parent.COORD))); H.append(heading)
            A.append(NB.index((cells[k + 1][0] - c[0], cells[k + 1][1] - c[1])) if k + 1 < len(cells) else -1)
        return np.asarray(X, np.float32), H, A


def chunks(steps, n=45):
    """split a trial into episodes of at most 45 moves (46 cells)"""
    return [steps[i:i + n + 1] for i in range(0, max(len(steps) - 1, 1), n)]


# ---------------- networks ----------------
def networks():
    out = [("frozen", p, s) for p in FROZEN for s in SEEDS]
    return out + [(k, "human_bc", s) for k in LEARNED for s in SEEDS]


@torch.no_grad()
def net_features(kind, prior, seed, episodes):
    """episodes: list of obs arrays; returns per step (hidden, action probabilities). Probes act at tau = 1."""
    m = SM.agent(kind, prior, seed); net = m.net; dev = next(net.parameters()).device; Hs, Ps = [], []
    G.CTX.update(c=0, mode="G0")
    for X in episodes:
        xt = torch.as_tensor(X, device=dev)
        if kind == "bl7":                                                    # the goal map's episode state: one step at a time
            hh, pp = [], []
            for k in range(len(X)):
                h = net.body(xt[k:k + 1]); hh.append(net.pi.code(h).cpu().numpy()); pp.append(torch.softmax(net.pi(h), -1).cpu().numpy())
            Hs.append(np.concatenate(hh)); Ps.append(np.concatenate(pp))
        else:
            h = net.body(xt); Ps.append(torch.softmax(net.pi(h), -1).cpu().numpy())
            Hs.append((net.pi.code(h) if kind == "bl3" else h).cpu().numpy())
    return Hs, Ps


# ---------------- per-subject states and neural vectors ----------------
def subject_states(subj, rep):
    """episodes of obs plus per-step (condition key, trial index, neural vector or None)."""
    cfg = subj["maze"]["configuration"]; eps, meta = [], []
    for ti, tr in enumerate(subj["trials"]):
        for ch in chunks(tr["steps"]):
            cells = [c for c, _, _ in ch]
            if len(cells) < 2: continue
            X, H, A = rep.episode(cfg, cells); eps.append(X)
            for k, (c, t0, t1) in enumerate(ch):
                hk = H[k] if subj["species"] == "rat" else (0, 0)             # humans: heading pooled (registered: ~39 trials)
                meta.append(dict(key=(c[0], c[1], hk[0], hk[1]), view=(c[0], c[1], H[k][0], H[k][1]), trial=ti, next=A[k], neural=neural(subj, ti, t0, t1)))
    return eps, meta


def neural(subj, ti, t0, t1):
    if subj["species"] == "rat":
        dt = t1 - t0
        if dt < .05: return None
        return np.array([((s >= t0) & (s < t1)).sum() / dt for s in subj["spikes"]], np.float32)
    tr = subj["trials"][ti]["trial"]; a, b = int(t0), int(t1) + 1
    return np.concatenate([e[tr][a:b].mean(0) for e in subj["power"]]).astype(np.float32)


def condition_means(meta, values, half=None):
    acc = collections.defaultdict(list)
    for m, v in zip(meta, values):
        if v is None or (half is not None and m["trial"] % 2 != half): continue
        acc[m["key"]].append(v)
    return {k: np.mean(v, 0) for k, v in acc.items()}


def rdm(D, keys, metric="corr"):
    X = np.stack([D[k] for k in keys]).astype(np.float64)
    if metric == "corr":
        X = X - X.mean(1, keepdims=True); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12; R = 1 - X @ X.T
    else: R = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    return R[np.triu_indices(len(keys), 1)]


def rank(v): from scipy.stats import rankdata; return rankdata(v)


def partial_rho(a, b, covs):
    ra, rb = rank(a), rank(b)
    if covs:
        C = np.column_stack([np.ones(len(ra))] + [rank(c) for c in covs])
        ra = ra - C @ np.linalg.lstsq(C, ra, rcond=None)[0]; rb = rb - C @ np.linalg.lstsq(C, rb, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def space_rdms(w, keys):
    n = len(keys); D = np.zeros((n, n)); Hd = np.zeros((n, n))
    dist = {}
    for i, a in enumerate(keys):
        if a[:2] not in dist: dist[a[:2]] = {c: len(track_path(w, a[:2], c)) for c in {k[:2] for k in keys}}
        for j, b in enumerate(keys): D[i, j] = dist[a[:2]][b[:2]]; Hd[i, j] = (a[2] - b[2]) ** 2 + (a[3] - b[3]) ** 2
    iu = np.triu_indices(n, 1); return D[iu], Hd[iu]


def subjects():
    return rats() + humans()


def check():
    S = subjects(); W = wtrack()
    print("W-track walkable", int(W["walkable"].sum()))
    for e in ("env1", "env2"):
        T = town_layout(e); print(T["name"], "IoU", round(T["iou"], 3), "cell", T["cell_units"], "stores", T["stores"], "walkable", int(T["walkable"].sum()))
        print("\n".join(" " + "".join("." if v else "#" for v in row) for row in T["walkable"]))
    rep = Replayer([W, town_layout("env1"), town_layout("env2")])
    for subj in S:
        eps, meta = subject_states(subj, rep); n = condition_means(meta, [m["neural"] for m in meta])
        keys = [k for k in n if all(sum(1 for m in meta if m["key"] == k and m["neural"] is not None and m["trial"] % 2 == h) >= MINV for h in (0, 1))]
        print(subj["species"], subj["id"], "trials", len(subj["trials"]), "steps", len(meta), "conditions", len(keys),
              "neural dim", len(next(iter(n.values()))) if n else 0,
              "belief on true cell", round(float(np.mean(np.concatenate([X[:, 150:250].argmax(1) for X in eps]) == np.array([m["key"][0] * 10 + m["key"][1] for m in meta]))), 3))
    print("subjects", len(S))



def keys_of(meta):
    cnt = collections.Counter((m["key"], m["trial"] % 2) for m in meta if m["neural"] is not None)
    k = sorted({k for k, _ in cnt if all(cnt[(k, h)] >= MINV for h in (0, 1))})
    return k if len(k) >= MINK else []


def neural_rdms(subj, meta, keys):
    out = {}
    groups = {"all": None} if subj["species"] == "human" else {"all": None, "CA1": subj["region"] == "CA1", "PFC": subj["region"] == "PFC"}
    for g, sel in groups.items():
        vals = [None if m["neural"] is None else (m["neural"] if sel is None else m["neural"][sel]) for m in meta]
        if sel is not None and sel.sum() < 5: continue
        if subj["species"] == "rat":                                       # units active on the track
            mu = np.mean([v for v in vals if v is not None], 0); keep = mu > .1; vals = [None if v is None else v[keep] for v in vals]
        out[g] = {h: rdm(condition_means(meta, vals, hh), keys) for h, hh in (("full", None), ("odd", 1), ("even", 0))}
    return out


def collect():
    import v4n_rosetta_matched_claude_20261031 as R
    assert PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    S = subjects(); mazes = [wtrack(), town_layout("env1"), town_layout("env2")]; rep = Replayer(mazes); walk = {m["configuration"]: m["walkable"] for m in mazes}
    E = R.Encoders("cuda:0"); subj_out = []; store = {}
    for si, subj in enumerate(S):
        eps, meta = subject_states(subj, rep); keys = keys_of(meta); cfg = subj["maze"]["configuration"]
        if not keys: print(subj["species"], subj["id"], "excluded (< %d conditions)" % MINK, flush=True); continue
        views = sorted({m["view"] for m in meta}); vi = {v: i for i, v in enumerate(views)}
        iv = np.concatenate([E.image([R.render_frame(rep.cache, cfg, v[:2], v[2:], 0) for v in views[i:i + 64]]) for i in range(0, len(views), 64)])
        tv = E.text([R.caption(walk[cfg], v[:2], v[2:], lexfree=False) for v in views])
        img = condition_means(meta, [iv[vi[m["view"]]] for m in meta]); tx = condition_means(meta, [tv[vi[m["view"]]] for m in meta])
        pix = condition_means(meta, list(np.concatenate(eps)[:, :144])); D, Hd = space_rdms(walk[cfg], keys)
        subj_out.append(dict(species=subj["species"], id=subj["id"], n_keys=len(keys), n_steps=len(meta), neural=neural_rdms(subj, meta, keys),
                             image=rdm(img, keys), text=rdm(tx, keys), pixels=rdm(pix, keys), space=D, heading=Hd))
        store[len(subj_out) - 1] = (eps, meta, keys); print(subj["species"], subj["id"], "conditions", len(keys), flush=True)
    res = {}
    for kind, prior, seed in networks():
        name = f"{kind}/{prior}/{seed}"; res[name] = []
        for si in range(len(subj_out)):
            eps, meta, keys = store[si]; Hs, Ps = net_features(kind, prior, seed, eps); Hc, Pc = np.concatenate(Hs), np.concatenate(Ps)
            nx = np.array([m["next"] for m in meta]); ok = nx >= 0
            res[name].append(dict(hidden=rdm(condition_means(meta, list(Hc)), keys), action=rdm(condition_means(meta, list(Pc)), keys, "euclid"),
                                  next_move=float(Pc[ok, nx[ok]].mean())))
        print(name, flush=True)
    import pickle
    (OUT / "collected.pkl").write_bytes(pickle.dumps(dict(subjects=subj_out, networks=res, written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))))


def ci(v):
    from scipy import stats
    v = np.asarray(v, float); return [round(float(v.mean()), 3), round(float(stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))), 3) if len(v) > 1 else None, len(v)]


def analyse(nperm=1000):
    import pickle
    C = pickle.loads((OUT / "collected.pkl").read_bytes()); S = C["subjects"]; rng = np.random.default_rng(20261107); out = dict(ceiling={}, families={}, controls={})
    fam = collections.defaultdict(list)
    for name in C["networks"]: k, p, s = name.split("/"); fam[k if k != "frozen" else f"frozen/{p}"].append(name)
    regions = sorted({g for s_ in S for g in s_["neural"]})
    for g in regions:
        for sp in ("rat", "human"):
            idx = [i for i, s_ in enumerate(S) if s_["species"] == sp and g in s_["neural"]]
            if not idx: continue
            tag = f"{sp}/{g}"
            ceil = []
            for i in idx:
                r = partial_rho(S[i]["neural"][g]["odd"], S[i]["neural"][g]["even"], []); ceil.append(2 * r / (1 + r))
            ids = [S[i]["id"] for i in idx]; out["ceiling"][tag] = ci([np.mean([c for c, j in zip(ceil, ids) if j == q]) for q in sorted(set(ids))])
            for c in ("image", "text", "pixels", "space"):
                out["controls"][f"{tag}/{c}"] = ci([partial_rho(S[i][c], S[i]["neural"][g]["full"], []) for i in idx])
            for f, names in sorted(fam.items()):
                for rep_ in ("hidden", "action"):
                    raw = np.array([[partial_rho(C["networks"][n][i][rep_], S[i]["neural"][g]["full"], []) for i in idx] for n in names])
                    part = np.array([[partial_rho(C["networks"][n][i][rep_], S[i]["neural"][g]["full"], [S[i]["space"], S[i]["heading"], S[i]["pixels"]]) for i in idx] for n in names])
                    ids = [S[i]["id"] for i in idx]; u = sorted(set(ids))
                    by = lambda v: np.array([np.mean([x for x, j in zip(v, ids) if j == q]) for q in u])
                    subj_raw, subj_part = by(raw.mean(0)), by(part.mean(0))           # seed-averaged, sessions averaged within subject
                    null = np.array([np.mean([partial_rho(C["networks"][names[0]][i][rep_], rng.permutation(S[i]["neural"][g]["full"]), []) for i in idx]) for _ in range(nperm // 10)])
                    out["families"][f"{tag}/{f}/{rep_}"] = dict(rho=ci(subj_raw), partial=ci(subj_part), perm_p=float((np.sum(null >= subj_raw.mean()) + 1) / (len(null) + 1)))
                if g == "all":
                    out["families"][f"{tag}/{f}/next_move"] = dict(prob=ci([np.mean([C["networks"][n][i]["next_move"] for n in names]) for i in idx]))
    out["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)[:6000])


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    W = wtrack(); T1, T2 = town_layout("env1"), town_layout("env2")
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=sha(__file__), shared_mazes_script_sha256=sha(SM.__file__),
        sources=dict(rat="DANDI 000978 (Shin & Jadhav), streamed: position, trials, units (spike times, CA1/PFC); 9 sessions / 8 rats",
                     human="Zenodo 8058006 (Seger, Lega & Ekstrom 2023), md5 1af121e6618477bceea09354cedc4f8e; 11 patients, 87 hippocampal-area electrodes"),
        mazes=dict(wtrack=W["walkable"].astype(int).tolist(), town_env1=T1["walkable"].astype(int).tolist(), town_env2=T2["walkable"].astype(int).tolist(),
                   town_iou=[T1["iou"], T2["iou"]], town_cell_units=[T1["cell_units"], T2["cell_units"]]),
        networks=f"frozen {FROZEN} and learned {LEARNED} (from human_bc), seeds {SEEDS[0]}-{SEEDS[-1]}; no training",
        replay="recorded trials replayed with the environment's own sensing (frame, flags, last move, LogBelief); no goal termination; 45-move episodes",
        conditions=f"rats (cell, heading), humans cell (heading pooled; ~39 trials), each with >= {MINV} visits with neural data in each of odd / even trials; sessions with < {MINK} conditions excluded; sessions averaged within subject",
        neural=dict(rat="unit rates in each cell visit (>= 50 ms), units with mean > 0.1 Hz; groups all / CA1 / PFC", human=f"log Hilbert power in {BANDS} Hz, z per electrode; all electrodes"),
        rdm="1 - Pearson (hidden, image, text, pixels, neural); Euclidean (action probabilities); space = track distance, heading change",
        outcomes="descriptive: per subject Spearman rho(model, neural), partial (space, heading, pixels); mean over subjects, t CI; split-half ceiling "
                 "(Spearman-Brown); label permutation; next-move probability on the same states; families = frozen priors, ppo, bl3, bl7",
        role="addition to the camera-ready; the paper's registered claims are unchanged",
        outcomes_seen="none; check printed trial / condition counts and the belief-on-true-cell rate only",
        deviations_allowed="none; any fix after this file goes to an ADDENDUM"), indent=1) + "\n")
    print(PROTOCOL.read_text()[:500])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check": check()
    elif cmd == "prereg": prereg()
    elif cmd == "collect": collect()
    elif cmd == "analyse": analyse()
