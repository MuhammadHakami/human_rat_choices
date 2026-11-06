"""Source-only protection ablations, GPU1 only, before fresh target tests.

Same continuous source stream and checked credit as the acquisition pilot.
Rates are selected from that audited source pilot; a common-rate BP reference
is also retained. Uniform attenuation and stationary-goal controls distinguish
selective protection from lower learning rate and background training drift.
"""
import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import wacv_embodied_long_control as parent
import wacv_native_behavior_prior_20261012 as source
from local_synaptic_protection_20261012 import ProtectedAC
from arc_fly.unimaze import NB
DeviceFFAgent = source.DeviceFFAgent  # trusted earlier __main__ checkpoint class

ROOT = parent.ROOT
BASE = ROOT / "data/audit/wacv_synaptic_protection_source_20261012"
PROTOCOL = ROOT / "docs/WACV_SYNAPTIC_PROTECTION_SOURCE_PROTOCOL_20261012.json"
ACQUISITION = ROOT / "data/audit/wacv_local_ac_source_pilot_20261012"
SEEDS = (60, 61, 62)
FRAMES = 65536
N = 16


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding():
    return dict(protocol=sha(PROTOCOL), source=source.hashes(),
                runner=sha(__file__), credit=sha(ROOT / "scripts/local_continual_actor_critic.py"),
                protection=sha(ROOT / "scripts/local_synaptic_protection_20261012.py"),
                acquisition_report=sha(ACQUISITION / "source_report.json"),
                acquisition_audit=sha(ACQUISITION / "audit.json"))


def prepare():
    assert not PROTOCOL.exists()
    report=json.loads((ACQUISITION / "source_report.json").read_text())
    audit=json.loads((ACQUISITION / "audit.json").read_text())
    assert report["status"]=="complete" and audit["ok"] and audit["cases"]==27
    assert report["audit_sha256"]==sha(ACQUISITION / "audit.json")
    selected={m["rule"]:m["selected"] for m in report["methods"]}
    assert all(selected[k]["competent_A"] and selected[k]["competent_B"] for k in ["local","bp"])
    local_rate=selected["local"]["lr"];bp_rate=selected["bp"]["lr"]
    treatments=[("local_selected","local",local_rate),("bp_selected","bp",bp_rate)]
    if bp_rate != local_rate:treatments.append(("bp_common","bp",local_rate))
    configs=[]
    for label,rule,lr in treatments:
        for protection in ["meta","slow","both","uniform_meta"]:
            configs.append(dict(id=label+"_"+protection+"_switch",rule=rule,rate=lr,protection=protection,shift="switch",B_goal=[6,6]))
        for protection in ["none","both"]:
            configs.append(dict(id=label+"_"+protection+"_stationary",rule=rule,rate=lr,protection=protection,shift="stationary",B_goal=[3,3]))
    source_protocol=json.loads((ROOT / "docs/WACV_LOCAL_AC_SOURCE_PILOT_PROTOCOL_20261012.json").read_text())
    p=dict(registered_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),status="source-only protection development",
       configurations=configs,source_models=source_protocol["source_models"],source_seeds=SEEDS,fits=len(configs)*3,
       frames_per_phase=FRAMES,environments=N,source_configuration=0,goals=dict(A=[3,3]),
       settings=dict(slow_tau=4096,meta_tau=2048,meta_max=25,consolidation=.001),
       fixed_learning=source_protocol["learning"],
       acquisition_selection={k:v["lr"] for k,v in selected.items()},
       reference_none_switch="Reused acquisition-pilot runs at the source-selected rates; common-rate BP uses its original rate2 runs",
       primary_development_metrics=["B acquisition success and success-weighted shortest-path efficiency","A efficiency change after B","Joint A/B competence; all seeds retained"],
       diagnostics=["Uniform_meta uses the same tensor-mean gate but removes synapse selectivity","Stationary reward goal controls continued-learning drift","Mean/min/max availability and slow-reference displacement at A/B endpoints","Visits to the alternative goal distinguish target retrieval from goal-agnostic tours"],
       interpretation="Exploratory source development. Inherited source A competence and frozen hidden-goal recall do not establish a task-inferred memory mechanism or lifetime no-forgetting guarantee.",
       no_task_id=True,no_phase_reset=True,target_maps_accessed=False,device="GPU1 RTX3090, one worker",
       readout="Scores read only after full cohort and independent audit",code=dict(runner=sha(__file__),credit=sha(ROOT / "scripts/local_continual_actor_critic.py"),protection=sha(ROOT / "scripts/local_synaptic_protection_20261012.py")))
    PROTOCOL.write_text(json.dumps(p,indent=2)+"\n");BASE.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(protocol=str(PROTOCOL),sha256=sha(PROTOCOL),fits=p["fits"])))


def cache_for(goal, frames):
    original, _, _ = parent.load()
    c = copy.copy(original)
    c.raw = copy.copy(original.raw)
    c.goal = c.raw.goal = tuple(goal)
    c.protocol = copy.deepcopy(original.protocol)
    c.protocol["source"].update(training_frames=frames, curriculum=[dict(
        until_total_frames=frames, configurations=[0], native_starts=True)])
    _, starts = c.geometry(0)
    assert tuple(goal) not in starts and c.geometry(0)[0][tuple(goal)]
    return c


def probe(agent, goal, seed, dry=False):
    cache = cache_for(goal, 30 * 45)
    _, starts = cache.geometry(0)
    schedule = [dict(configuration=0, spawn=s, rotation_index=rot)
                for rot in range(3) for s in starts]
    if dry:
        schedule = schedule[:2]
    m = copy.deepcopy(agent)
    m.tg = torch.Generator(device="cuda:0").manual_seed(230017 + seed)
    initial_rng = m.tg.get_state().cpu().numpy().copy()
    env = parent.LongBatch(cache, n=1, frames=45 * len(schedule), schedule=schedule)
    obs = env.reset()
    before = {k: p.clone() for k, p in m.params.items()}
    probs, values = [], []
    while not env.exhausted and env.frames < env.limit:
        a, p, v = m.sample(torch.as_tensor(obs, device="cuda:0"))
        probs.append(p[0].cpu().tolist()); values.append(float(v[0]))
        obs, _, _ = env.step(a.cpu().numpy())
    assert all(torch.equal(p, before[k]) for k, p in m.params.items())
    eps = parent.base.episodes(env, probs, values)
    alternative = (6, 6) if tuple(goal) == (3, 3) else (3, 3)
    decision_efficiency = []
    for ep in eps:
        optimal = sum(abs(a-b) for a,b in zip(ep["spawn"], goal))
        decision_efficiency.append(float(ep["success"]) * optimal / max(optimal, ep["decisions"], 1))
    return dict(goals=sum(e["success"] for e in eps), trials=len(eps), frames=env.frames,
                mean_spl=float(np.mean([e["spl"] for e in eps])),
                mean_decision_efficiency=float(np.mean(decision_efficiency)),
                alternative_goal_visit_fraction=float(np.mean([list(alternative) in e["path"] for e in eps])),
                weights_frozen_verified=True, episodes=eps,
                rng_initial=initial_rng.tolist())


def fit(configuration, seed, dry=False):
    p=json.loads(PROTOCOL.read_text())
    assert p["code"]==dict(runner=sha(__file__),credit=sha(ROOT / "scripts/local_continual_actor_critic.py"),protection=sha(ROOT / "scripts/local_synaptic_protection_20261012.py"))
    case=next(c for c in p["configurations"] if c["id"]==configuration)
    rule=case["rule"];rate=case["rate"]
    assert seed in SEEDS
    if not dry:
        go=json.loads((BASE / "GO.json").read_text());assert go["status"]=="GO" and go["hashes"]==binding()
    d=BASE / ("dry" if dry else "cohort") / configuration / f"seed{seed}"
    d.mkdir(parents=True,exist_ok=True)
    with (d / "started.json").open("x") as f:json.dump(dict(hashes=binding(),dry=dry),f)
    rec=dict(configuration=configuration,rule=rule,rate=rate,protection=case["protection"],shift=case["shift"],B_goal=case["B_goal"],seed=seed,dry=dry,hashes=binding(),status="running")
    start = time.perf_counter()
    try:
        item = next(r for r in p["source_models"] if r["seed"] == seed)
        assert sha(ROOT / item["path"]) == item["sha256"]
        # Serialized CPU action-generator state must remain a ByteTensor on
        # CPU. Copy only the initialized network weights into our GPU learner.
        loaded = torch.load(ROOT / item["path"], map_location="cpu", weights_only=False)
        m = ProtectedAC(n_envs=N, rule=rule, lr=rate, protection=case["protection"], seed=27000 + seed,
                             device="cuda:0", source=loaded.net)
        del loaded
        before = probe(m, p["goals"]["A"], seed, dry)
        phase_frames = N * 8 if dry else FRAMES
        cache = cache_for(p["goals"]["A"], phase_frames * 2)
        env = parent.LongBatch(cache, n=N, seed=82000 + seed, frames=phase_frames * 2)
        obs = env.reset()
        log = {k: [] for k in ("actions", "probabilities", "values", "rewards", "done", "before_position", "after_position")}
        action_rng = m.tg.get_state().cpu().numpy().copy()
        switch_check = None
        acquired = None
        for tick in range(phase_frames * 2 // N):
            if env.frames == phase_frames:
                # Internal engine operation only. The agent's current input and
                # all learner state survive; no observation or reset is emitted.
                acquired = probe(m, p["goals"]["A"], seed, dry)
                b_before = probe(m, case["B_goal"], seed, dry)
                snapshot_A=m.plasticity_snapshot()
                torch.save(m, d / "checkpoint_afterA.pt")
                trace_before = {k: t.clone() for k, t in m.actor_trace.items()}
                worlds_before = copy.deepcopy(env.worlds)
                old_obs = obs.copy()
                env.env.cache.goal = tuple(case["B_goal"])
                switch_check = dict(observation_unchanged=np.array_equal(old_obs, obs),
                                    world_state_unchanged=worlds_before == env.worlds,
                                    eligibility_unchanged=all(torch.equal(trace_before[k], t) for k, t in m.actor_trace.items()),
                                    no_learning_state_reset=True)
                assert all(switch_check.values())
            x = torch.as_tensor(obs, device="cuda:0")
            a, prob, value = m.sample(x)
            positions = [w["position"] for w in env.worlds]
            obs2, rewards, done = env.step(a.cpu().numpy())
            # Reconstruct physical end positions before auto-respawn using the
            # logged actions; all source0 cells are free, except arena bounds.
            after_pos = []
            for pos, action in zip(positions, a.cpu().tolist()):
                q = tuple(u + v for u, v in zip(pos, NB[action]))
                after_pos.append(q if all(0 <= z < 10 for z in q) else pos)
            m.learn(x, a, torch.as_tensor(rewards, device="cuda:0"),
                    torch.as_tensor(done, device="cuda:0"), torch.as_tensor(obs2, device="cuda:0"))
            for k, v in [("actions", a.cpu().numpy()), ("probabilities", prob.cpu().numpy()),
                         ("values", value.cpu().numpy()), ("rewards", rewards), ("done", done),
                         ("before_position", positions), ("after_position", after_pos)]:
                log[k].append(np.asarray(v))
            obs = obs2
        assert m.count["frames"] == env.frames == phase_frames * 2
        after_a = probe(m, p["goals"]["A"], seed, dry)
        after_b = probe(m, case["B_goal"], seed, dry)
        snapshot_B=m.plasticity_snapshot()
        torch.save(m, d / "checkpoint_afterB.pt")
        np.savez_compressed(d / "training.npz", **{k: np.asarray(v) for k, v in log.items()},
                            action_rng_initial=action_rng)
        phase_data = dict(A_initial=before, A_acquired=acquired, B_before=b_before,
                          A_afterB=after_a, B_after=after_b)
        parent.gz_write(d / "probes.json.gz", phase_data)
        rec.update(status="complete", phase_frames=phase_frames,
                   probes={k: {a: b for a, b in v.items() if a not in ("episodes", "rng_initial")} for k, v in phase_data.items()},
                   switch_check=switch_check, frames=env.frames, counters=m.count,
                   plasticity=dict(afterA=snapshot_A,afterB=snapshot_B),
                   resources=m.memory_bytes(), gpu_name=torch.cuda.get_device_name(0),
                   files={name: sha(d / name) for name in ["checkpoint_afterA.pt", "checkpoint_afterB.pt", "training.npz", "probes.json.gz"]})
    except Exception as exc:
        rec.update(status="invalid", error=repr(exc), traceback=traceback.format_exc())
        (d / "result.json").write_text(json.dumps(rec, indent=2) + "\n")
        raise
    rec["wall_s"] = time.perf_counter() - start
    (d / "result.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in ("rule", "rate", "seed", "dry", "status", "wall_s")}), flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=["prepare","dry","fit"])
    ap.add_argument("configuration",nargs="?");ap.add_argument("seed",nargs="?",type=int);args=ap.parse_args()
    torch.set_num_threads(1)
    if args.command=="prepare":prepare()
    else:
        assert os.environ.get("CUDA_DEVICE_ORDER")=="PCI_BUS_ID" and os.environ.get("CUDA_VISIBLE_DEVICES")=="1"
        assert "3090" in torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        fit(args.configuration,args.seed,args.command=="dry")
