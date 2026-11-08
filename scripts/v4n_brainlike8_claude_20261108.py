"""Brain-constrained circuit, stage 8: inhibition of return at every visited place (camera-ready, author 2026-11-08: "investigate why
sometimes in new scenarios our biologically plausible system keeps repeating ... finding the root cause and resolving for
biologically plausible way end to end"). GPU 1 only. Registered before any outcome of this stage's learner.

Root cause (descriptive diagnosis with the project page's step-exact JavaScript port of the stage-7 agents, seeds 1000-1004 x 3
priors, the 30-trial probe in each seed's trained and never-seen fresh room, both goals; outcomes seen, disclosed below):
  * Memoryless agents (frozen priors, PPO, the stage-3 circuit) loop by pushing into a wall their place-driven habit is confident
    about: at trapped steps the habit's best action is optimal 2-25% of the time with p = 0.74-0.95, the belief is correct (100%),
    and 10-14 of the last 16 moves are wall bumps. No within-episode state can change this (Lean: trapped_forever).
  * The stage-7 goal map still fails in never-seen rooms with the familiar goal (62%) and 90% of those failures are loops in which
    the DRIVE itself asks for the bump: at trapped steps the chosen link is already depressed (99%), yet D[a] = 0 (best action) with
    p_drive = 0.78-1.00 and 15 of 16 moves are bumps. Mechanism: q_a for a depressed link equals v[c] ("staying"), and the blurred
    goal field (sigma) makes the agent's own cell a local maximum of v next to a wall: v[c] = G_e[c] >= gamma_h * v(any open
    neighbour), so staying (bumping) is the drive's best action and short-term depression, which only closes links, cannot change
    it. Inhibition of return never fires there because stage 7 inhibits only places with G_e[c] > thr * max G_e (thr = 0.7 was the
    searched winner), i.e. only at the remembered goal cell itself, never on the blur around it. A second, rarer case (trained room,
    moved goal: 19 of 450 probe trials): the field peak sits in an unreachable cell (a write at a mis-estimated place; the max write
    never forgets), which can never be visited and therefore never inhibited.
  * Fix tested at probe time with the stage-7 weights (same 15 agents): thr = 0, i.e. the dopamine dip inhibits EVERY visited place
    in proportion to the expectation it carried (G_e <- G_e (1 - K_sigma(place)) whenever G_e[place] > 0): never-seen familiar
    61.8 -> 100.0, never-seen moved 81.8 -> 84.4, trained moved 93.3 -> 93.3, trained familiar 100 -> 100; trapped failures 0% in
    every condition; bumps per episode 14.7 -> 2.5 (never-seen familiar). Alternatives tried there: a known collision worth 0
    instead of "staying" (61.8 -> 91.1; adds nothing on top of thr = 0); zeroing the field at the cell behind a bump (harmful,
    never-seen moved 81.8 -> 29.6). Oracle pre-closing of the three visible links changed little (61.8 -> 69.6): the loop is not a
    perception failure.
Why thr = 0 is the biologically principled form: the reward prediction error at a place with expectation G_e[c] and no reward is
-G_e[c]; a dopamine dip proportional to it (Schultz 1997) is what the stage-7 rule already computes, and K_sigma(c)[c] = 1, so a
visit without reward removes the expectation at that cell exactly. Thresholding the dip at 70% of the maximum expectation was a
searched convenience with no biological reading. Corollary of the Lean theorem wave_ascent (proofs_ml/ProofsML/Wavefront.lean):
after a visit without reward the visited cell is no longer a source, so (as long as a remembered place is reachable) some open
neighbour has strictly higher wavefront value: the drive can never prefer staying at a visited cell. Loops of the kind diagnosed
are therefore impossible with thr = 0.

Mechanism: stage 7 exactly (v4n_brainlike7_claude_20261107: goal memory, lattice with bump depression, wavefront value, steering),
with thr = 0 in both stage-7 winning configurations (BEST.json: none, kappa 50; inferred, kappa 10). Everything else, including the
learning schedule, the mushroom-body parameters and the probes, is unchanged. The whole learner is re-run (learning uses the drive
too), so this is an end-to-end test, not a probe-time patch. Arms: bl8_none, bl8_inferred. No search.
Check (before registering): with thr = 0.7 this script's run path reproduces the stage-7 test record of cold_ppo seed 1000
(bl7_none) exactly (same code, same device).
Predictions, paired over (prior, seed) with the stage-7 test records on the same seeds, priors, rooms and probes (30 seeds x 3
priors; contrast as stages 4-7, 95% CI; TOST margin 10 pp):
  P1 familiar goal, never-seen room: stage 8 - stage 7 >= +20 pp (probe-time estimate +38 on 5 seeds).
  P2 moved goal, never-seen room: within +-5 pp (estimate +2.6).  P3 both trained-room probes: within +-5 pp.
  P4 (descriptive, page port after the test): trapped failures in never-seen rooms near 0%.
Disclosure: the diagnosis used seeds 1000-1004 (dev seeds 1000-1002 plus test seeds 1003-1004) in the fresh test rooms with the
tested stage-7 weights, so the rule was chosen after seeing probe-time outcomes on 5 of the 30 test seeds; the analysis reports all
30 seeds and the 25 seeds not used in the diagnosis. Test once; no further parameter changes after this file is hashed.
  check | prereg | test W N | analyse
"""
import json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_brainlike7_claude_20261107 as B7
M = B7.M; P4 = B7.P4

OUT = M.ROOT / "data/audit/v4n_brainlike8_20261108"; PROTOCOL = OUT / "PROTOCOL.json"
ARMS = ("none", "inferred")
EXCLUDED = (1000, 1001, 1002, 1003, 1004)          # seeds used in the probe-time diagnosis


def cfgs():
    best = json.loads((B7.OUT / "BEST.json").read_text())["configs"]
    return {g: dict(best[g], thr=0.0) for g in ARMS}


def run8(args):
    """stage-7 run path with this stage's protocol hash in the record."""
    B7.PROTOCOL = PROTOCOL
    return B7.run7(args)


def test(w, n):
    assert PROTOCOL.exists() and os.environ.get("CUDA_VISIBLE_DEVICES") == "1"; pick = cfgs()
    for g, cfg, p, s in [(g, cfg, p, s) for s in M.FR.S30 for p in M.PRIORS for g, cfg in sorted(pick.items())][w::n]:
        run8((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / f"bl8_{g}" / p / f"seed{s}.json")))
    print(json.dumps(dict(worker=w, done=True)), flush=True)


def analyse():
    from prior_seed_power_claude import contrast
    own = lambda d: {(r["prior"], r["seed"]): r for r in (json.loads(f.read_text()) for f in d.glob("*/seed*.json"))}
    res = {}
    for g in ARMS:
        S = own(OUT / "test" / f"bl8_{g}"); O = own(B7.OUT / "test" / f"bl7_{g}")
        if not S: continue
        R = dict(n=len(S), means={q: float(np.mean([r[q] for r in S.values()]) * 100 / 30) for q in P4}, diverged=sum(r["diverged"] is not None for r in S.values()),
                 stage7_means={q: float(np.mean([O[k][q] for k in S if k in O]) * 100 / 30) for q in P4})
        for tag, keys in (("all", sorted(set(S) & set(O))), ("excluding_diagnosis_seeds", sorted(k for k in set(S) & set(O) if k[1] not in EXCLUDED))):
            for q in P4:
                R[f"{q}/bl7/{tag}"] = contrast({"a": {k: S[k][q] * 100 / 30 for k in keys}, "b": {k: O[k][q] * 100 / 30 for k in keys}}, "a", "b", 10.)
        res[g] = R; print(g, R["n"], {q[:14]: round(v, 1) for q, v in R["means"].items()}, "stage 7", {q[:14]: round(v, 1) for q, v in R["stage7_means"].items()}, "diverged", R["diverged"])
        for q, v in R.items():
            if "/" in q: print("  ", q, round(v["mean"], 1), [round(x, 1) for x in v["ci95"]], "p=%.2g tost=%.2g n=%d" % (v["t_p"], v["tost_p"], v.get("n", 0)))
    pred = {g: dict(P1=res[g]["familiar_heldout_room/bl7/all"]["ci95"][0] >= 20. or res[g]["familiar_heldout_room/bl7/all"]["mean"] >= 20.,
                    P2=abs(res[g]["moved_heldout_room/bl7/all"]["mean"]) <= 5., P3=all(abs(res[g][f"{q}/bl7/all"]["mean"]) <= 5. for q in ("moved_trained_room", "familiar_trained_room"))) for g in res}
    (OUT / "analysis.json").write_text(json.dumps(dict(res, written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), predictions_met=pred), indent=1, default=float) + "\n")
    print("predictions", pred)


def prereg():
    assert not PROTOCOL.exists() and (OUT / "CHECK.json").exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script_sha256=M.sha(__file__), stage7_script_sha256=M.sha(B7.__file__),
        design=__doc__, configs=cfgs(), lean=dict(file="proofs_ml/ProofsML/Wavefront.lean", sha256=M.sha(B7.LEAN)),
        outcomes_seen="stages 1-7 dev and test; the probe-time diagnosis in the design text (page port, seeds 1000-1004, stage-7 weights, test rooms); the CHECK.json reproduction run",
        diagnosis_seeds=list(EXCLUDED), dev_seeds=list(M.DEV_SEEDS), test_seeds=list(M.FR.S30), priors=M.PRIORS), indent=1) + "\n"); print("registered")


def check():
    """thr = 0.7 (stage 7's value) through this script reproduces the shipped stage-7 test record exactly."""
    import tempfile
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    ref = json.loads((B7.OUT / "test/bl7_none/cold_ppo/seed1000.json").read_text()); cfg = json.loads((B7.OUT / "BEST.json").read_text())["configs"]["none"]
    tmp = Path(tempfile.mkdtemp()); global PROTOCOL; P0, PROTOCOL = PROTOCOL, tmp / "P.json"; PROTOCOL.write_text("{}")
    rec = run8((cfg, "cold_ppo", 1000, str(M.FR.MAN), str(tmp / "rep.json"))); PROTOCOL = P0
    assert all(rec[q] == ref[q] for q in P4), ([rec[q] for q in P4], [ref[q] for q in P4])
    assert rec["goal_memory"] == ref["goal_memory"], "goal memory differs"
    OUT.mkdir(parents=True, exist_ok=True)
    rep = dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), reproduced=dict(prior="cold_ppo", seed=1000, arm="none", values=[rec[q] for q in P4]), gpu=rec["gpu_name"], wall_s=rec["wall_s"])
    (OUT / "CHECK.json").write_text(json.dumps(rep, indent=1) + "\n"); print("check ok", rep)


if __name__ == "__main__":
    {"check": check, "prereg": prereg, "analyse": analyse, "test": lambda: test(int(sys.argv[2]), int(sys.argv[3]))}[sys.argv[1]]()
