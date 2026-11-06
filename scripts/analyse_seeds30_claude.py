"""Pre-registered analysis of the 30-seed native-prior replication (data/wacv/seeds30/preregistration.json).
Refuses to read any score until every arm x seed is complete (no peeking at partial cohorts).
  python scripts/analyse_seeds30_claude.py            -> data/wacv/seeds30/analysis.json
  python scripts/analyse_seeds30_claude.py selftest   (synthetic numbers only)
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_seed_power_claude import contrast

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/wacv/seeds30"
SECONDARY = [("rat_bc", "cold_ppo"), ("rat_bc", "rat_sham"), ("rat_legal_uniform", "cold_ppo"), ("human_bc", "human_legal_uniform")]


def scores():
    P = json.loads((D / "preregistration.json").read_text()); S = {a: {} for a in P["arms"]}
    for a in P["arms"]:
        for s in P["seeds"]:
            r = json.loads((D / "cohort" / a / f"seed{s}" / "result.json").read_text())
            assert r["status"] in ("complete", "failed"), (a, s, r["status"])
            if r["status"] == "complete": S[a][s] = r["adaptive_successes"]
    return P, S


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.
    for k, i in enumerate(order): run = max(run, min(1., (len(ps) - k) * ps[i])); adj[i] = run
    return adj.tolist()


def analyse(S, solved=40):
    out = dict(primary=contrast(S, "rat_bc", "rat_legal_uniform"), secondary={}, solved_seeds={}, per_seed=S)
    p = out["primary"]
    p["reading"] = ("equivalent" if p["tost_p"] < .05 else
                    "rat-choice advantage" if p["t_p"] < .05 and p["signflip_p"] < .05 and p["mean"] > 0 else
                    "legal-uniform advantage" if p["t_p"] < .05 and p["signflip_p"] < .05 else "inconclusive")
    sec = {f"{a}-{b}": contrast(S, a, b) for a, b in SECONDARY}
    for k, adj in zip(sec, holm([v["t_p"] for v in sec.values()])): sec[k]["holm_t_p"] = adj
    out["secondary"] = sec
    ref = S["rat_bc"]
    for a, v in S.items():
        k = sum(x >= solved for x in v.values()); out["solved_seeds"][a] = dict(k=k, n=len(v))
        if a != "rat_bc":
            kr = sum(x >= solved for x in ref.values())
            out["solved_seeds"][a]["fisher_p_vs_rat_bc"] = float(stats.fisher_exact([[kr, len(ref) - kr], [k, len(v) - k]])[1])
    return out


def selftest():
    rng = np.random.default_rng(1); seeds = range(30)
    S = {a: {s: int(rng.integers(0, 61)) for s in seeds} for a in ("rat_bc", "rat_legal_uniform", "cold_ppo", "rat_sham", "human_bc", "human_legal_uniform")}
    S["rat_sham"] = {s: max(0, v - 30) for s, v in S["rat_bc"].items()}             # planted large effect
    o = analyse(S); assert o["secondary"]["rat_bc-rat_sham"]["holm_t_p"] < 1e-6
    S["rat_legal_uniform"] = {s: v for s, v in S["rat_bc"].items()}; S["rat_legal_uniform"][0] += 1   # near-identical arms
    assert analyse(S)["primary"]["reading"] == "equivalent"
    assert holm([.01, .04, .03]) == [.03, .06, .06]; print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]: selftest()
    else:
        P, S = scores(); o = analyse(S); o["preregistration"] = str(D / "preregistration.json")
        (D / "analysis.json").write_text(json.dumps(o, indent=1) + "\n"); print(json.dumps({k: o[k] for k in ("primary", "solved_seeds")}, indent=1))
