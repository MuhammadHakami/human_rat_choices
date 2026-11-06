"""Pre-registered follow-up (data/wacv/seeds30_followup/preregistration.json): is the human-choice OOD gain explained by a
competent expert or the fitted goal-bias/persistence surrogate? F1: human_bc - human_surrogate and human_bc - human_expert
on zero-shot success (seed mean over 16 rooms), paired n=30, Holm over 2. F2 descriptive: rat arms and in-distribution success.
  python scripts/analyse_ood_followup_claude.py -> data/audit/wacv_ood_followup_claude/analysis_followup.json
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_seed_power_claude import contrast
from analyse_seeds30_claude import holm

ROOT = Path(__file__).resolve().parents[1]
MAIN, FU = ROOT / "data/audit/wacv_ood_zero_shot_claude", ROOT / "data/audit/wacv_ood_followup_claude"


def ood(base, arm, seeds):
    return {s: float(np.mean([json.loads((base / "cohort" / arm / f"seed{s}" / f"pair{p}_{st}.json").read_text())["goals"]
                              for p in range(8) for st in ("A", "B")])) for s in seeds}


def main():
    seeds = list(range(1000, 1030)); S = {a: ood(MAIN, a, seeds) for a in ("human_bc", "rat_bc", "cold_ppo")}
    S.update({a: ood(FU, a, seeds) for a in ("human_expert", "human_surrogate", "rat_expert", "rat_surrogate")})
    f1 = {k: contrast(S, "human_bc", k, 3.) for k in ("human_surrogate", "human_expert")}
    for k, h in zip(f1, holm([v["t_p"] for v in f1.values()])): f1[k]["holm_t_p"] = h
    ind = {a: float(np.mean([json.loads((ROOT / "data/wacv/seeds30_followup/cohort" / a / f"seed{s}" / "result.json").read_text())["adaptive_successes"]
                             for s in seeds])) for a in ("human_expert", "human_surrogate", "rat_expert", "rat_surrogate")}
    out = dict(ood_mean_per30={a: float(np.mean(list(v.values()))) for a, v in S.items()}, F1=f1,
               F2=dict(rat_bc_minus_rat_surrogate=contrast(S, "rat_bc", "rat_surrogate", 3.), rat_bc_minus_rat_expert=contrast(S, "rat_bc", "rat_expert", 3.),
                       in_distribution_mean_per60=ind))
    (FU / "analysis_followup.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
