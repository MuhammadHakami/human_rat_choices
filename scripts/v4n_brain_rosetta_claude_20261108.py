"""Rosetta modalities against recorded brains on the same states (camera-ready page, author 2026-11-08: "update the page with
brain geometry to action-image-text also given rosetta setup"). Descriptive; no new runs, no retraining.
Input: the registered collections of the brain-maze (W-track rat CA1/PFC, town human hippocampus) and open-arena (rat MEC,
human MTL; ADDENDUM1-filtered) studies. Each holds, per subject, condition RDMs on the replayed states for
  image   DINOv2-base CLS of the rendered view (the Rosetta encoder)     text    MPNet of the Rosetta template caption
  brain   recorded neural RDM                                              and, per network, hidden (last layer) and action (probabilities).
Outcome: for every setting x network family, the 5 x 5 matrix of Spearman rho between the condition RDMs of
image, text, action, hidden and brain, seed-averaged, then sessions averaged within subject, mean +- t CI over subjects; the
same matrix partialling out space (track distance + heading change). Brain ceiling = split-half Spearman-Brown.
RSA on condition RDMs replaces CKA / the unpaired solver: 19-56 conditions per subject are too few for Wasserstein-Procrustes.
Disclosed: brain-image, brain-text, brain-hidden and brain-action rho (raw; partial with pixels) were already reported by those
studies; the network-modality pairs on these states (image/text-action, image/text-hidden, hidden-action) are new.
  prereg | analyse
"""
import collections, json, pickle, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_brain_mazes_claude_20261107 as BM

ROOT = BM.ROOT; OUT = ROOT / "data/audit/v4n_brain_rosetta_20261108"; PROTOCOL = OUT / "PROTOCOL.json"
SOURCES = {"maze": ROOT / "data/audit/v4n_brain_mazes_20261107/collected.pkl",
           "arena": ROOT / "data/audit/v4n_brain_arenas_20261108/filtered/collected.pkl"}
SETTINGS = (("wtrack_rat_CA1", "maze", "rat", "CA1"), ("wtrack_rat_PFC", "maze", "rat", "PFC"), ("towns_human", "maze", "human", "all"),
            ("arena_rat", "arena", "rat", "all"), ("arena_human", "arena", "human", "all"))
MODS = ("image", "text", "action", "hidden", "brain")


def family(name):
    k, p, _ = name.split("/"); return f"frozen/{p}" if k == "frozen" else k


def subject_rdms(S, N, names, i, g):
    """modality -> list of RDMs (one per seed for network modalities, one otherwise)"""
    s = S[i]; one = dict(image=[s["image"]], text=[s["text"]], brain=[s["neural"][g]["full"]])
    return {**one, "action": [N[n][i]["action"] for n in names], "hidden": [N[n][i]["hidden"] for n in names]}


def matrix(R, covs):
    M = np.eye(len(MODS))
    for a in range(len(MODS)):
        for b in range(a + 1, len(MODS)):
            A, B = R[MODS[a]], R[MODS[b]]
            v = [BM.partial_rho(x, y, covs) for x, y in (zip(A, B) if len(A) == len(B) else ((x, y) for x in A for y in B))]
            M[a, b] = M[b, a] = np.mean(v)
    return M


def analyse():
    assert PROTOCOL.exists(); out = dict(mods=MODS, settings={})
    for tag, src, sp, g in SETTINGS:
        C = pickle.loads(SOURCES[src].read_bytes()); S, N = C["subjects"], C["networks"]
        fam = collections.defaultdict(list)
        for n in N: fam[family(n)].append(n)
        idx = [i for i, s in enumerate(S) if s["species"] == sp and g in s["neural"]]; ids = [S[i]["id"] for i in idx]; u = sorted(set(ids))
        ceil = []
        for i in idx: r = BM.partial_rho(S[i]["neural"][g]["odd"], S[i]["neural"][g]["even"], []); ceil.append(2 * r / (1 + r))
        res = dict(n_subjects=len(u), n_sessions=len(idx), ceiling=BM.ci([np.mean([c for c, j in zip(ceil, ids) if j == q]) for q in u]), families={})
        for f, names in sorted(fam.items()):
            per = {m: [] for m in ("raw", "partial")}
            for i in idx:
                R = subject_rdms(S, N, names, i, g)
                per["raw"].append(matrix(R, [])); per["partial"].append(matrix(R, [S[i]["space"], S[i]["heading"]]))
            res["families"][f] = {}
            for m, Ms in per.items():
                Ms = np.array(Ms); bysub = np.array([Ms[[j == q for j in ids]].mean(0) for q in u])
                res["families"][f][m] = [[BM.ci(bysub[:, a, b]) for b in range(len(MODS))] for a in range(len(MODS))]
        out["settings"][tag] = res; print(tag, "subjects", len(u), "sessions", len(idx), flush=True)
    out["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1))
    b = MODS.index("brain")
    for tag, r in out["settings"].items():
        print(tag, "ceiling", r["ceiling"][0], " ".join(f"{f.split('/')[-1]}:" + ",".join(f"{MODS[a][0]}{r['families'][f]['partial'][b][a][0]:+.3f}" for a in range(4)) for f in r["families"]))


def prereg():
    OUT.mkdir(parents=True, exist_ok=True); assert not PROTOCOL.exists()
    src = Path(__file__).read_text(); import hashlib
    PROTOCOL.write_text(json.dumps(dict(registered_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), design=__doc__, settings=SETTINGS, modalities=MODS,
                                        inputs={k: str(v.relative_to(ROOT)) for k, v in SOURCES.items()}, script_sha256=hashlib.sha256(src.encode()).hexdigest()), indent=1))
    print("registered", PROTOCOL)


if __name__ == "__main__":
    {"prereg": prereg, "analyse": analyse}[sys.argv[1]]()
