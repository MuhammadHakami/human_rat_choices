"""Seed-count planning for the native-prior comparisons (Claude; descriptive, reads audited per-seed results only).
Paired per-seed differences (same seed = same init/PPO RNG) from the audited 7-arm cohort and 4-arm controls; exact
paired-t power via the noncentral t for n seeds, assuming the observed 3-seed SD (an unstable estimate with n=3,
so a 2x-SD row is also given).  python scripts/prior_seed_power_claude.py -> data/wacv/prior_seed_power.json
"""
import glob, json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "data/audit/wacv_native_behavior_prior_20261012"
PAIRS = [("rat_bc", "cold_ppo"), ("rat_bc", "rat_sham"), ("rat_bc", "rat_legal_uniform"), ("rat_bc", "rat_surrogate"),
         ("rat_legal_uniform", "cold_ppo"), ("human_bc", "human_legal_uniform"), ("human_expert", "cold_ppo")]


def scores():
    assert json.loads((B / "cohort/root_audit.json").read_text())["ok"]
    assert json.loads((B / "controls_cohort/controls_audit.json").read_text())["ok"]
    S = {}
    for f in glob.glob(str(B / "cohort*/*/seed*/result.json")) + glob.glob(str(B / "controls_cohort/*/seed*/result.json")):
        r = json.loads(Path(f).read_text()); assert r["status"] == "complete" and not r["dry"]
        S.setdefault(r["arm"], {})[r["seed"]] = r["adaptive_successes"]
    return S


def power(delta, sd, n, alpha=.05):
    df = n - 1; nc = delta / (sd / np.sqrt(n)); t = stats.t.ppf(1 - alpha / 2, df)
    p = stats.nct.sf(t, df, nc) + stats.nct.cdf(-t, df, nc)
    return 1. if np.isnan(p) and nc > 10 else float(p)   # ponytail: scipy nct gives nan at very large nc; power is ~1 there


def seeds_needed(delta, sd, target=.8):
    return next((n for n in range(3, 201) if power(delta, sd, n) >= target), None)


def main():
    S = scores(); out = dict(note=__doc__.split("\n")[1], arms={k: dict(sorted(v.items())) for k, v in sorted(S.items())}, pairs={})
    for a, b in PAIRS:
        d = np.array([S[a][s] - S[b][s] for s in sorted(S[a])], float); sd = d.std(ddof=1)
        row = dict(diffs=d.tolist(), mean=float(d.mean()), sd=float(sd))
        for eff in (10, 20):                       # successes /60 per seed
            row[f"seeds_80pct_delta{eff}"] = seeds_needed(eff, sd); row[f"seeds_80pct_delta{eff}_2xsd"] = seeds_needed(eff, 2 * sd)
        out["pairs"][f"{a}-{b}"] = row
        print(f"{a:18s}-{b:20s} diffs={d.astype(int).tolist()} mean={d.mean():6.1f} sd={sd:5.1f} n(d=10)={row['seeds_80pct_delta10']} n(d=20)={row['seeds_80pct_delta20']}")
    solved = {k: sum(v >= 40 for v in S[k].values()) for k in S}  # bimodal per-seed outcomes: >=40/60 vs <=31/60 observed
    out["seeds_at_or_above_40_of_60"] = solved; print(solved)
    (ROOT / "data/wacv/prior_seed_power.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    assert seeds_needed(10, 10) in range(9, 13)    # textbook: d=1 paired, 80% power -> ~10
    main()


def contrast(S, a, b, bound=10.):
    """Pre-registered paired analysis (written before any new-seed outcome): seeds present in both arms."""
    seeds = sorted(set(S[a]) & set(S[b])); d = np.array([S[a][s] - S[b][s] for s in seeds], float); n = len(d)
    signs = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T if n <= 16 else \
        np.random.default_rng(0).choice([-1, 1], (100000, n))           # exact sign-flip up to 16 seeds, else 1e5 draws
    perm_p = float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))
    se = d.std(ddof=1) / np.sqrt(n); df = n - 1
    tost_p = float(max(stats.t.sf((d.mean() + bound) / se, df), stats.t.cdf((d.mean() - bound) / se, df)))
    return dict(n=n, mean=float(d.mean()), ci95=[float(d.mean() + q * se) for q in stats.t.ppf([.025, .975], df)],
                t_p=float(stats.ttest_1samp(d, 0).pvalue), signflip_p=perm_p, tost_bound=bound, tost_p=tost_p)


if __name__ == "__main__" and __import__("sys").argv[1:] == ["contrast"]:
    S = scores()
    for a, b in PAIRS: print(f"{a}-{b}", json.dumps(contrast(S, a, b)))
