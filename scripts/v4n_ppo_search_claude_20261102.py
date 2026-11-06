"""Matched PPO search: the fairness reference for v4n_local_search2 (camera-ready, 2026-11-02). GPU 1.
Written after 47 dev trials of the local search had reached the dev ceiling (30/30): the registered PPO was never tuned,
so a tuned local rule against an untuned PPO would not be a fair comparison. Same dev rooms, seeds, priors, budget,
objective, trial count (160) and expansion option; the space matches wherever PPO has the knob (rollout T, epochs,
minibatches, clip, GAE lambda, entropy, value weight, Adam rate, linear decay, expansion K / k-WTA). Backprop through
all layers (PPO's own credit), registered gamma 0.99, advantage normalisation and grad-norm clip 0.5 as registered.
Best trial tested once on 30 seeds x 3 priors in the fresh rooms, then the three frozen KPI probes.
  check | prereg | search W TRIALS | best | test W N | kpi W N
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4n_moved_goal_learning_claude_20261031 as M
import v4n_local_search2_claude_20261102 as S2

OUT = M.ROOT / "data/audit/v4n_ppo_search_20261102"; PROTOCOL = OUT / "PROTOCOL.json"; DB = f"sqlite:///{OUT / 'optuna.db'}"; NAME = "ppo_tuned"


def configure(m, cfg, seed):
    if cfg["expand"]: S2.expand(m.net, cfg["expand"], cfg["kwta"], seed)
    m.cfg.update(n_steps=cfg["T"], epochs=cfg["epochs"], mb=cfg["minibatches"], clip=cfg["clip"], lam=cfg["gae"], ent=cfg["ent"], vf=cfg["vf"])
    m.opt = torch.optim.Adam(m.net.parameters(), lr=cfg["rate"], eps=1e-5)


def run_cfg(args):
    cfg, prior, seed, manifest, out = args
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1"); torch.set_num_threads(1); manifest, out = Path(manifest), Path(out)
    if out.exists(): return json.loads(out.read_text())
    pair, stage = M.room_of(seed); gm = M.goal_map(manifest); m = M.build(prior, seed); configure(m, cfg, seed); t0 = time.perf_counter()
    env = M.env_for(manifest, gm, pair, M.N_ENV, seed, "clean", M.BUDGET); obs = env.reset()
    curve = {0: M.probe_now(m, manifest, gm, pair, stage, seed)}
    wrapped = M.F.LoggingRooms(env, obs); state = (torch.zeros(1, M.N_ENV, 0), torch.zeros(1, M.N_ENV, 0)); dprev = np.zeros(M.N_ENV, np.float32); diverged = None
    while m.count["frames"] < M.BUDGET:
        if cfg["decay"]:
            for g in m.opt.param_groups: g["lr"] = cfg["rate"] * (1 - m.count["frames"] / M.BUDGET)
        B, s0, obs, state, dprev = m.rollout(wrapped, obs, state, dprev); m.update(B, s0)
        if not all(torch.isfinite(q).all() for q in m.net.parameters()): diverged = m.count["frames"]; break
        if m.count["frames"] in M.CURVE: curve[m.count["frames"]] = M.probe_now(m, manifest, gm, pair, stage, seed)
    for x in M.CURVE: curve.setdefault(x, dict(goals=0, episodes=[]))
    rec = dict(learner=NAME, cfg=cfg, prior=prior, seed=seed, pair=pair, room=stage, goal=list(gm[100 + 2 * pair]), budget=M.BUDGET,
               curve={k: v["goals"] for k, v in curve.items()}, diverged_at=diverged, final_episodes=curve[M.BUDGET]["episodes"],
               wall_s=time.perf_counter() - t0, protocol_sha256=M.sha(PROTOCOL), gpu_name=torch.cuda.get_device_name(0))
    out.parent.mkdir(parents=True, exist_ok=True); torch.save({k: v.cpu() for k, v in m.net.state_dict().items()}, out.with_suffix(".final.pt"))
    out.write_text(json.dumps(rec) + "\n"); return rec


def objective(trial):
    s = trial.suggest_categorical
    cfg = dict(expand=s("expand", [0, 512, 2048]), kwta=trial.suggest_float("kwta", .02, .3, log=True), T=s("T", [8, 32, 128]),
               epochs=s("epochs", [1, 2, 4, 8]), minibatches=s("minibatches", [1, 4]), clip=s("clip", [.1, .2, .3]),
               gae=trial.suggest_float("gae", .5, .99), ent=trial.suggest_float("ent", 0., .1), vf=trial.suggest_float("vf", .15, 1.5, log=True),
               rate=trial.suggest_float("rate", 1e-5, 1e-2, log=True), decay=s("decay", [True, False]))
    jobs = [(cfg, p, sd, str(M.FR.EXPOSED), str(OUT / "search" / f"trial{trial.number}" / p / f"seed{sd}.json")) for p in M.PRIORS for sd in M.DEV_SEEDS]
    with ProcessPoolExecutor(9) as ex: recs = list(ex.map(run_cfg, jobs))
    cur = lambda r, x: r["curve"][str(x)] if str(x) in r["curve"] else r["curve"][x]
    xs = np.array(M.CURVE, float); final = float(np.mean([cur(r, M.BUDGET) for r in recs]))
    auc = float(np.mean([np.trapezoid([cur(r, int(x)) for x in xs], xs) / (30 * M.BUDGET) for r in recs]))
    trial.set_user_attr("final", final); trial.set_user_attr("auc", auc); return final + auc


def search(w, trials):
    import optuna
    st = optuna.create_study(study_name="v4n_ppo", storage=DB, direction="maximize", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=20261102 + w, n_startup_trials=12, multivariate=True, group=True))
    st.optimize(objective, n_trials=trials)


def best():
    import optuna
    st = optuna.load_study(study_name="v4n_ppo", storage=DB); done = [t for t in st.trials if t.value is not None]; t = max(done, key=lambda t: t.value)
    (OUT / "BEST.json").write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), trials=len(done), config=t.params,
        trial_number=t.number, dev_score=t.value, dev_final=t.user_attrs["final"]), indent=1) + "\n"); print(t.number, t.value, t.params)


def test(w, n):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"; cfg = json.loads((OUT / "BEST.json").read_text())["config"]
    for p, s in [(p, s) for s in M.FR.S30 for p in M.PRIORS][w::n]: run_cfg((cfg, p, s, str(M.FR.MAN), str(OUT / "test" / NAME / p / f"seed{s}.json")))
    print(json.dumps(dict(worker=w, done=True)), flush=True)


def kpi(w, n):
    import v4n_kpi_claude_20261031 as K
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    for f in sorted((OUT / "test").glob("*/*/seed*.json"))[w::n]:
        r = json.loads(f.read_text()); out = K.OUT / "kpi" / NAME / r["prior"] / f.name
        if out.exists(): continue
        m = M.build(r["prior"], r["seed"])
        if r["cfg"]["expand"]: S2.expand(m.net, r["cfg"]["expand"], r["cfg"]["kwta"], r["seed"])
        m.net.load_state_dict({k: v.to(next(m.net.parameters()).device) for k, v in torch.load(f.with_suffix(".final.pt")).items()})
        pair = r["seed"] % 8; mg = M.goal_map(M.FR.MAN); fam = {c: K.FAM for c in mg}
        rec = dict(learner=NAME, prior=r["prior"], seed=r["seed"], source=str(f), familiar_trained_room=K.probe(m, fam, pair, "A", r["seed"]),
                   moved_heldout_room=K.probe(m, mg, pair, "B", r["seed"]), familiar_heldout_room=K.probe(m, fam, pair, "B", r["seed"]))
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec) + "\n")
    print(json.dumps(dict(stage="kpi", worker=w, done=True)), flush=True)


def prereg():
    assert not PROTOCOL.exists(); OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(dict(written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        outcomes_seen="47 dev trials of v4n_local_search2 (dev ceiling reached); no test outcome of either search",
        script_sha256=M.sha(__file__), design=__doc__, trials=160, objective="as v4n_local_search2", selection="best trial",
        test="30 seeds x 3 priors, fresh rooms; three frozen KPI probes",
        reading="secondary/descriptive: s2 learners vs tuned PPO, final success and AUC paired by prior x seed; per-prior contrasts as v4n_moved_goal_learning"), indent=1) + "\n")
    print("registered")


def check():
    m = M.build("human_bc", 1000); x = torch.randn(5, 252, device=next(m.net.parameters()).device)
    with torch.no_grad(): p0 = m.net.heads(m.net.body(x))[0].probs
    configure(m, dict(expand=512, kwta=.05, T=8, epochs=2, minibatches=4, clip=.1, gae=.9, ent=.01, vf=.5, rate=1e-3, decay=True), 1000)
    with torch.no_grad(): assert torch.allclose(p0, m.net.heads(m.net.body(x))[0].probs)
    assert any(q is m.net.pi.R.weight for g in m.opt.param_groups for q in g["params"]) and m.cfg["n_steps"] == 8
    print("check ok")


if __name__ == "__main__":
    c = sys.argv[1]; torch.set_num_threads(1)
    {"check": check, "prereg": prereg, "best": best, "search": lambda: search(int(sys.argv[2]), int(sys.argv[3])),
     "test": lambda: test(int(sys.argv[2]), int(sys.argv[3])), "kpi": lambda: kpi(int(sys.argv[2]), int(sys.argv[3]))}[c]()
