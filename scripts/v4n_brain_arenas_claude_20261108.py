"""Brain geometry in open arenas, human + rat (camera-ready addition, author 2026-11-08: "can't you find similar 2D maze for human
from DANDI or other sources that is open source"; "continue"). Same method as v4n_brain_mazes_claude_20261107 (registered there;
functions reused unchanged); registered here before any agent is run in these arenas. No training.
Arenas (our 10 x 10 arena, same renderer and moves; cells outside the arena are removed modules):
  human  OpenNeuro ds005522 (Herrema & Kahana, CC0), task YC1: 58 patients, iEEG (bipolar EDF), open 64.8 x 36 arena; test
         trials = self-navigation to a remembered (invisible) object location. Grid 10 x 6 (cells 6.48 x 6.0 units), rows 2-7.
  rat    DANDI 000405 (Gonzalez & Giocomo 2022, CC BY 4.0): 5 rats, MEC / PrS / PaS units, open-field foraging, ~1.25 x 1.28 m.
         Grid 6 x 6 (~0.21 m cells, our 0.2 m module), rows 2-7, cols 2-7.
Same states: recorded paths (human test trials; rat sessions cut into 45-move segments) replayed through every network.
Neural per step: human log Hilbert power (bands as brain_mazes) of bipolar channels in the medial temporal lobe (both contacts
labelled parahippocampal / entorhinal, or unlabelled depth contacts inside the MNI box |x| 18-38, y -42..-8, z -28..-2 =
hippocampus), resampled to 250 Hz, z per channel; rat unit rates (50 Hz bins) averaged over the step.
Conditions: cell (heading pooled), >= 4 visits per odd / even trial-or-segment, >= 8 conditions per session; sessions
averaged within subject. Outcomes, RDMs, partial controls, ceilings, permutation, next-move probability: as brain_mazes.
  check | prereg | collect | analyse
"""
import json, os, sys, time, glob, csv, collections
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_brain_mazes_claude_20261107 as BM
import edf_min
ROOT = BM.ROOT; OUT = ROOT / "data/audit/v4n_brain_arenas_20261108"; PROTOCOL = OUT / "PROTOCOL.json"
HU = ROOT / "data/real_sources/kahana_ds005522"; RA = ROOT / "data/real_sources/gonzalez2022_dandi000405"
HBOX = ((18, 38), (-42, -8), (-28, -2)); MTL = ("parahippocampal", "entorhinal")


def arena(cfg, rows, cols):
    w = np.zeros((10, 10), bool); w[rows[0]:rows[1], cols[0]:cols[1]] = True; return dict(configuration=cfg, walkable=w, name=f"arena {cfg}")


HUMAN_ARENA, RAT_ARENA = arena(310, (2, 8), (0, 10)), arena(311, (2, 8), (2, 8))
to_cell_h = lambda x, y: (2 + int(np.clip((18.0 - y) // 6.0, 0, 5)), int(np.clip((x + 32.4) // 6.48, 0, 9)))   # north (+y) at row 2


def mtl_channels(sdir):
    """bipolar channels whose both contacts are MTL (labelled) or in the hippocampal MNI box (unlabelled depth)"""
    el = {}
    for f in glob.glob(str(sdir / "ieeg/*electrodes.tsv")):
        for r in csv.DictReader(open(f), delimiter="\t"):
            reg = (r.get("ind.region") or "").lower(); xyz = [float(r[k]) if r[k] not in ("n/a", "") else np.nan for k in ("x", "y", "z")]
            box = all(lo <= abs(v) <= hi if i == 0 else lo <= v <= hi for i, (v, (lo, hi)) in enumerate(zip(xyz, HBOX))) if not np.isnan(xyz).any() else False
            el[r["name"]] = "mtl" if reg in MTL else ("hip" if reg in ("n/a", "") and box else "other")
    ch = []
    for f in glob.glob(str(sdir / "ieeg/*bipolar_channels.tsv")):
        for r in csv.DictReader(open(f), delimiter="\t"):
            a, _, b = r["name"].partition("-")
            if el.get(a, "other") != "other" and el.get(b, "other") != "other": ch.append(r["name"])
    return ch


def human_sessions():
    from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly
    out = []
    for sdir in sorted(HU.glob("sub-*/ses-*")):
        edf = glob.glob(str(sdir / "ieeg/*acq-bipolar_ieeg.edf")); ev = glob.glob(str(sdir / "beh/*_beh.tsv"))
        if not edf or not ev: continue
        chans = mtl_channels(sdir)
        if not chans: continue
        H = edf_min.header(edf[0]); idx = [H["label"].index(c) for c in chans if c in H["label"]]
        if os.path.getsize(edf[0]) < H["hbytes"] + 2 * H["nrec"] * sum(H["nsamp"]): print("incomplete EDF", edf[0], flush=True); continue
        if not idx: continue
        fs = H["fs"][idx[0]]; up, down = (5, 32) if abs(fs - 1600) < 1 else (250, int(round(fs)))
        P = []
        for i in idx:
            x = resample_poly(edf_min.signal(edf[0], i, H), up, down)
            p = np.stack([np.log(np.abs(hilbert(sosfiltfilt(butter(4, b, "bandpass", fs=250, output="sos"), x))) ** 2 + 1e-12) for b in BM.BANDS], 1)
            P.append((p - p.mean(0)) / (p.std(0) + 1e-9))
        P = np.concatenate(P, 1); rows = list(csv.DictReader(open(ev[0]), delimiter="\t")); trials = collections.defaultdict(list)
        for r in rows:
            if r["trial_type"] in ("TRAVEL_TEST", "TRAVEL_LEARN") and r["x"] not in ("n/a", ""):   # registered: learning (guided) + test (self)
                s = int(int(r["sample"]) * 250 / fs); trials[(r["trial"], r["block"], r["trial_type"])].append((to_cell_h(float(r["x"]), float(r["y"])), s))
        T = [dict(steps=BM.to_steps(HUMAN_ARENA["walkable"], [c for c, _ in v], [s for _, s in v]), self=k[2] == "TRAVEL_TEST") for k, v in trials.items() if len(v) >= 2]
        out.append(dict(species="human", id=sdir.parent.name, session=sdir.name, maze=HUMAN_ARENA, trials=T, power=P, channels=chans))
    return out


def rat_sessions(seg=45):
    import h5py
    out = []
    for f in sorted(RA.glob("sub-*/*_ses-*-OF-*.nwb")):
        h = h5py.File(f, "r"); tr = h["acquisition/track_data/data"][:]; fr = h["acquisition/firing_rates/data"][:]
        n = min(len(tr), len(fr)); tr, fr = tr[:n], fr[:n]; ok = ~np.isnan(tr[:, :2]).any(1)
        x, y = tr[:, 0], tr[:, 1]; x0, x1 = np.nanpercentile(x, [1, 99]); y0, y1 = np.nanpercentile(y, [1, 99])
        cell = lambda a, b: (7 - int(np.clip((b - y0) / (y1 - y0) * 6, 0, 5)), 2 + int(np.clip((a - x0) / (x1 - x0) * 6, 0, 5)))
        cells = [cell(a, b) for a, b in zip(x[ok], y[ok])]; st = BM.to_steps(RAT_ARENA["walkable"], cells, np.flatnonzero(ok))
        T = [dict(steps=st[i:i + seg + 1]) for i in range(0, len(st) - 1, seg)]
        out.append(dict(species="rat", id=f.parent.name, session=f.stem, maze=RAT_ARENA, trials=T, rates=fr, n_units=fr.shape[1]))
    return out


def neural(subj, ti, t0, t1):
    a, b = int(t0), int(t1) + 1
    if subj["species"] == "human": return subj["power"][a:b].mean(0).astype(np.float32) if b > a else None
    return subj["rates"][a:b].mean(0).astype(np.float32)


BM.neural = neural                                                          # the arena neural accessor


def subject_states(subj, rep):
    """BM.subject_states with heading pooled for both species (open arenas)"""
    cfg = subj["maze"]["configuration"]; eps, meta = [], []
    for ti, tr in enumerate(subj["trials"]):
        for ch in BM.chunks(tr["steps"]):
            cells = [c for c, _, _ in ch]
            if len(cells) < 2: continue
            X, H, A = rep.episode(cfg, cells); eps.append(X)
            for k, (c, t0, t1) in enumerate(ch):
                meta.append(dict(key=(c[0], c[1], 0, 0), view=(c[0], c[1], H[k][0], H[k][1]), trial=ti, next=A[k] if tr.get("self", True) else -1,
                                 neural=neural(subj, ti, t0, t1)))                         # next move only where the subject chose it
    return eps, meta


def neural_rdms(subj, meta, keys):
    vals = [m["neural"] for m in meta]
    if subj["species"] == "rat":
        mu = np.mean([v for v in vals if v is not None], 0); keep = mu > .1; vals = [None if v is None else v[keep] for v in vals]
    return {"all": {h: BM.rdm(BM.condition_means(meta, vals, hh), keys) for h, hh in (("full", None), ("odd", 1), ("even", 0))}}


BM.subject_states, BM.neural_rdms = subject_states, neural_rdms


def check():
    R = rat_sessions(); print("rat OF sessions", len(R), "rats", len({r["id"] for r in R}), "units", sum(r["n_units"] for r in R))
    rep = BM.Replayer([HUMAN_ARENA, RAT_ARENA])
    for s in R[:3]:
        eps, meta = subject_states(s, rep); print(" rat", s["session"], "steps", len(meta), "conditions", len(BM.keys_of(meta)))
    H = human_sessions(); print("human sessions with MTL channels", len(H), "patients", len({h["id"] for h in H}), "channels", [len(h["channels"]) for h in H])
    for s in H[:3]:
        eps, meta = subject_states(s, rep); print(" human", s["id"], s["session"], "trials", len(s["trials"]), "steps", len(meta), "conditions", len(BM.keys_of(meta)),
              "belief on true cell", round(float(np.mean(np.concatenate([X[:, 150:250].argmax(1) for X in eps]) == np.array([m["key"][0] * 10 + m["key"][1] for m in meta]))), 3))



def collect():
    """BM.collect for the arenas (same features, same networks); output OUT/collected.pkl in the BM format"""
    import pickle, v4n_rosetta_matched_claude_20261031 as R
    assert PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    S = rat_sessions() + human_sessions(); mazes = [HUMAN_ARENA, RAT_ARENA]; rep = BM.Replayer(mazes); walk = {m["configuration"]: m["walkable"] for m in mazes}
    E = R.Encoders("cuda:0"); subj_out, store = [], {}
    for subj in S:
        eps, meta = subject_states(subj, rep); keys = BM.keys_of(meta); cfg = subj["maze"]["configuration"]
        if not keys: print(subj["species"], subj["id"], subj["session"], "excluded", flush=True); continue
        views = sorted({m["view"] for m in meta}); vi = {v: i for i, v in enumerate(views)}
        iv = np.concatenate([E.image([R.render_frame(rep.cache, cfg, v[:2], v[2:], 0) for v in views[i:i + 64]]) for i in range(0, len(views), 64)])
        tv = E.text([R.caption(walk[cfg], v[:2], v[2:], lexfree=False) for v in views])
        img = BM.condition_means(meta, [iv[vi[m["view"]]] for m in meta]); tx = BM.condition_means(meta, [tv[vi[m["view"]]] for m in meta])
        pix = BM.condition_means(meta, list(np.concatenate(eps)[:, :144])); D, Hd = BM.space_rdms(walk[cfg], keys)
        subj_out.append(dict(species=subj["species"], id=subj["id"], session=subj["session"], n_keys=len(keys), n_steps=len(meta), neural=neural_rdms(subj, meta, keys),
                             image=BM.rdm(img, keys), text=BM.rdm(tx, keys), pixels=BM.rdm(pix, keys), space=D, heading=Hd))
        store[len(subj_out) - 1] = (eps, meta, keys); print(subj["species"], subj["id"], subj["session"], "conditions", len(keys), flush=True)
    res = {}
    for kind, prior, seed in BM.networks():
        name = f"{kind}/{prior}/{seed}"; res[name] = []
        for si in range(len(subj_out)):
            eps, meta, keys = store[si]; Hs, Ps = BM.net_features(kind, prior, seed, eps); Hc, Pc = np.concatenate(Hs), np.concatenate(Ps)
            nx = np.array([m["next"] for m in meta]); ok = nx >= 0
            res[name].append(dict(hidden=BM.rdm(BM.condition_means(meta, list(Hc)), keys), action=BM.rdm(BM.condition_means(meta, list(Pc)), keys, "euclid"),
                                  next_move=float(Pc[ok, nx[ok]].mean())))
        print(name, flush=True)
    (OUT / "collected.pkl").write_bytes(pickle.dumps(dict(subjects=subj_out, networks=res, written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))))


def analyse():
    """ADDENDUM1: rat sessions with < 5 units above 0.1 Hz have a constant correlation-distance RDM (undefined rho) and are
    dropped, the minimum BM already applies to region groups; applied by filtering collected.pkl into analysis_input.pkl."""
    import pickle
    C = pickle.loads((OUT / "collected.pkl").read_bytes()); keep = [i for i, s in enumerate(C["subjects"]) if np.ptp(s["neural"]["all"]["full"]) > 0
                                                                     and all(np.ptp(s["neural"]["all"][h]) > 0 for h in ("odd", "even"))]
    C2 = dict(C, subjects=[C["subjects"][i] for i in keep], networks={k: [v[i] for i in keep] for k, v in C["networks"].items()}, dropped=len(C["subjects"]) - len(keep))
    tmp = OUT / "filtered"; tmp.mkdir(exist_ok=True); (tmp / "collected.pkl").write_bytes(pickle.dumps(C2)); print("dropped", C2["dropped"], "sessions")
    BM.OUT = tmp; BM.analyse(); (OUT / "analysis.json").write_text((tmp / "analysis.json").read_text())


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=BM.sha(__file__),
        brain_mazes_script_sha256=BM.sha(BM.__file__), edf_reader_sha256=BM.sha(edf_min.__file__),
        sources=dict(human="OpenNeuro ds005522 v1.0.0 (CC0), task YC1, bipolar EDF; learning (guided) + test (self-navigation) travel; next-move metric on test only (decided from condition counts: test-only gave 0-24 conditions per session)",
                     rat="DANDI 000405 draft (CC BY 4.0), open-field sessions (133), track_data + firing_rates (50 Hz)"),
        arenas=dict(human=HUMAN_ARENA["walkable"].astype(int).tolist(), rat=RAT_ARENA["walkable"].astype(int).tolist()),
        channels=f"human bipolar pairs with both contacts labelled {MTL} or unlabelled inside MNI box {HBOX} (hippocampus)",
        method="as v4n_brain_mazes_claude_20261107 (replay, features, RDMs, partial controls, ceilings, permutation, next move); conditions = cell, heading pooled",
        networks="frozen human_bc / rat_bc / human_legal_uniform / cold_ppo; learned ppo / bl3 / bl7 from human_bc; seeds 1000-1029; no training",
        role="addition to the camera-ready supplement; paper claims unchanged",
        outcomes_seen="rat check: 6 sessions' condition counts (34-36) and belief accuracy (1.0); human: condition counts under 4 candidate rules (25 sessions) and belief accuracy (1.0) on 4 sessions; no neural or model RSA",
        deviations_allowed="none; fixes go to an ADDENDUM"), indent=1) + "\n")
    print(PROTOCOL.read_text()[:400])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check": check()
    elif cmd == "prereg": prereg()
    elif cmd == "collect": collect()
    elif cmd == "analyse": analyse()
