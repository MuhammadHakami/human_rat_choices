"""Strong PPO reference on the same continuous source goal-shift stream.

Source-only follow-up: the earlier18 PPO fits failed B competence. This is
standard PPO tuning, not a proposed algorithm. No target policy results have
been inspected. Shorter rollouts and entropy are selected only on the source.
This is standard PPO, not a proposed algorithm. CPU sampling and GPU1 affine
layers reuse the already reviewed collector and objective. Original registered
local/protection programs remain untouched. Scores are gated on the full audit.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import wacv_synaptic_protection_source_20261012 as controls
import wacv_native_behavior_prior_20261012 as source
from ppo_feedforward_reference import FeedforwardPPO

DeviceFFAgent = source.DeviceFFAgent
ROOT = controls.ROOT
BASE = ROOT / "data/audit/wacv_continual_ppo_tuning_source_20261012"
PROTOCOL = ROOT / "docs/WACV_CONTINUAL_PPO_TUNING_SOURCE_PROTOCOL_20261012.json"
SEEDS = (60, 61, 62)
N, FRAMES = 16, 65536
RATES = (.0001, .0003, .001)


def binding():
    return dict(protocol=controls.sha(PROTOCOL), runner=controls.sha(__file__),
                ppo=controls.sha(ROOT / "scripts/recurrent_ppo.py"),
                feedforward=controls.sha(ROOT / "scripts/ppo_feedforward_reference.py"),
                device_module=controls.sha(ROOT / "scripts/wacv_native_behavior_prior_20261012.py"),
                environment=controls.binding())


def prepare():
    assert not PROTOCOL.exists()
    previous=json.loads(controls.PROTOCOL.read_text())
    p=dict(registered_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
           configurations=[dict(id=f"rate{i}_steps{steps}_entropy{ei}",rate=lr,shift="switch", B_goal=[6,6],n_steps=steps,ent=entropy)
                           for i,lr in enumerate(RATES) for steps in (16,32) for ei,entropy in enumerate((.01,.05,.1))],
           seeds=SEEDS,fits=54,source_models=previous["source_models"],
           frames_per_phase=FRAMES,environments=N,
           ppo=dict(n_steps=128,gamma=.99,lam=.95,clip=.2,ent=.01,vf=.5,
                    epochs=4,minibatches=2,gradient_clip=.5,adam_eps=1e-5),
           selection="First require B_after>=15/30 in at least2/3 seeds; select maximum pooled B acquisition, then higher mean B SPL, then lower entropy, lower rate, longer rollout. If no configuration passes, do not claim a competent strong PPO reference.",
           metrics=["B acquisition success/SPL", "Immediate A-probe success/SPL", "Alternative-goal visits", "Stationary drift", "Actual persistent and rollout storage"],
           interpretation="Standard strong backprop reference; source development, not selective hidden-task retrieval or pure-local whole-pipeline evidence.",
           source_initialization="Same BP-trained rat_bc source checkpoints as local/protection studies; fresh optimizer at start only.",
           protocol_continuity="Engine goal changes only. Observation, world, optimizer, sampler and rollout state carry. Switch aligned with a normal PPO rollout boundary; no extra update or reset.",
           device="Physical GPU1 only; one worker",sampling="Reviewed CPU categorical generator, distinct from local CUDA sampling; same seed identities, not identical sampled actions.",
           readout="Full cohort and independent audit before scores. Full seed61 re-execution for each of the eighteen configurations.",
           code={name:controls.sha(ROOT / name) for name in ["scripts/wacv_continual_ppo_tuning_source_20261012.py", "scripts/recurrent_ppo.py", "scripts/ppo_feedforward_reference.py"]})
    PROTOCOL.write_text(json.dumps(p,indent=2)+"\n")
    BASE.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(protocol=str(PROTOCOL),fits=54,sha256=controls.sha(PROTOCOL))))


def build(seed, rate, n_steps=128, ent=.01):
    p=json.loads(PROTOCOL.read_text())
    row=next(x for x in p["source_models"] if x["seed"]==seed)
    assert controls.sha(ROOT / row["path"])==row["sha256"]
    initial=torch.load(ROOT / row["path"],map_location="cpu",weights_only=False)
    settings={k:v for k,v in p["ppo"].items() if k not in ("gradient_clip","adam_eps")}
    settings.update(n_steps=n_steps,ent=ent)
    m=FeedforwardPPO(252,4,n_envs=N,lr=rate,seed=27000+seed,**settings)
    m.net=DeviceFFAgent(252,4).to("cuda:0")
    m.net.load_state_dict(initial.net.state_dict())
    m.opt=torch.optim.Adam(m.net.parameters(),lr=rate,eps=1e-5)
    assert not m.opt.state and m.count["frames"]==0
    return m


@torch.no_grad()
def probe(agent, goal, seed, dry=False):
    cache=controls.cache_for(goal,30*45)
    _,starts=cache.geometry(0)
    schedule=[dict(configuration=0,spawn=s,rotation_index=rot) for rot in range(3) for s in starts]
    if dry:schedule=schedule[:2]
    m=copy.deepcopy(agent)
    m.tg=torch.Generator().manual_seed(230017+seed)
    rng=m.tg.get_state().numpy().copy()
    env=controls.parent.LongBatch(cache,n=1,frames=45*len(schedule),schedule=schedule)
    obs=env.reset();params={k:v.detach().clone() for k,v in m.net.named_parameters()}
    state=(torch.zeros(1,1,0),torch.zeros(1,1,0))
    probs,values=[],[]
    while not env.exhausted and env.frames<env.limit:
        o,_=m.net.unroll(torch.as_tensor(obs)[None],state,torch.zeros(1,1))
        dist,v=m.net.heads(o[0]);a=torch.multinomial(dist.probs,1,generator=m.tg).squeeze(-1)
        probs.append(dist.probs[0].tolist());values.append(float(v[0]))
        obs,_,_=env.step(a.numpy())
    assert all(torch.equal(params[k],v) for k,v in m.net.named_parameters())
    eps=controls.parent.base.episodes(env,probs,values)
    alternative=(6,6) if tuple(goal)==(3,3) else (3,3)
    de=[]
    for ep in eps:
        shortest=sum(abs(a-b) for a,b in zip(ep["spawn"],goal))
        de.append(float(ep["success"])*shortest/max(shortest,ep["decisions"],1))
    return dict(goals=sum(e["success"] for e in eps),trials=len(eps),frames=env.frames,
                mean_spl=float(np.mean([e["spl"] for e in eps])),mean_decision_efficiency=float(np.mean(de)),
                alternative_goal_visit_fraction=float(np.mean([list(alternative) in e["path"] for e in eps])),
                weights_frozen_verified=True,episodes=eps,rng_initial=rng.tolist())


class LoggedEnvironment:
    def __init__(self, env):
        self.env=env;self.log={k:[] for k in ("before_position","after_position","actions","rewards","done")}
    def step(self, actions):
        before=[w["position"] for w in self.env.worlds]
        after=[]
        for pos,a in zip(before,actions):
            q=tuple(u+v for u,v in zip(pos,controls.NB[int(a)]))
            after.append(q if all(0<=z<10 for z in q) else pos)
        obs,r,d=self.env.step(actions)
        for k,v in [("before_position",before),("after_position",after),("actions",actions),("rewards",r),("done",d)]:
            self.log[k].append(np.array(v,copy=True))
        return obs,r,d


def fit(configuration,seed,dry=False):
    p=json.loads(PROTOCOL.read_text());case=next(c for c in p["configurations"] if c["id"]==configuration)
    if not dry:
        go=json.loads((BASE / "GO.json").read_text());assert go["hashes"]==binding()
    assert seed in SEEDS
    d=BASE / ("dry" if dry else "cohort") / configuration / f"seed{seed}"
    d.mkdir(parents=True,exist_ok=True)
    with (d / "started.json").open("x") as f:json.dump(dict(hashes=binding()),f)
    start=time.perf_counter();m=build(seed,case["rate"],case["n_steps"],case["ent"])
    # Dry cases retain normal PPO rollout length and a genuine update on each
    # side, but do not use the full source budget or select any scientific rate.
    count=N*m.cfg["n_steps"] if dry else FRAMES
    env=controls.parent.LongBatch(controls.cache_for((3,3),2*count),n=N,seed=82000+seed,frames=2*count)
    obs=env.reset();wrapped=LoggedEnvironment(env)
    state=(torch.zeros(1,N,0),torch.zeros(1,N,0));dp=np.zeros(N,np.float32)
    action_rng=m.tg.get_state().numpy().copy()
    phase_data=dict(A_initial=probe(m,(3,3),seed,dry))
    probabilities,values=[],[]
    switch_check=None
    while m.count["frames"]<2*count:
        if m.count["frames"]==count:
            phase_data["A_acquired"]=probe(m,(3,3),seed,dry)
            phase_data["B_before"]=probe(m,case["B_goal"],seed,dry)
            torch.save(m,d / "checkpoint_afterA.pt")
            old_obs=obs.copy();old_world=copy.deepcopy(env.worlds)
            old_sampler=m.tg.get_state().clone();old_opt=copy.deepcopy(m.opt.state_dict())
            env.env.cache.goal=tuple(case["B_goal"])
            switch_check=dict(observation_unchanged=np.array_equal(obs,old_obs),world_state_unchanged=env.worlds==old_world,
                              sampler_unchanged=torch.equal(old_sampler,m.tg.get_state()),no_learning_state_reset=True)
            # Engine mutation cannot touch optimizer/state; exact tensors are
            # also compared by the independent full re-execution auditor.
            assert old_opt.keys()==m.opt.state_dict().keys() and all(switch_check.values())
        B,s0,obs,state,dp=m.rollout(wrapped,obs,state,dp)
        probabilities.extend(B["probs"].numpy());values.extend(B["val"].numpy())
        m.update(B,s0)
    assert m.count["frames"]==env.frames==2*count
    phase_data["A_afterB"]=probe(m,(3,3),seed,dry)
    phase_data["B_after"]=probe(m,case["B_goal"],seed,dry)
    torch.save(m,d / "checkpoint_afterB.pt")
    np.savez_compressed(d / "training.npz",**{k:np.asarray(v) for k,v in wrapped.log.items()},
                        probabilities=np.asarray(probabilities),values=np.asarray(values),action_rng_initial=action_rng)
    controls.parent.gz_write(d / "probes.json.gz",phase_data)
    r=dict(**case,seed=seed,dry=dry,status="complete",hashes=binding(),phase_frames=count,
           probes={k:{a:b for a,b in v.items() if a not in ("episodes","rng_initial")} for k,v in phase_data.items()},
           switch_check=switch_check,counters=m.count,resources=m.memory_bytes(),
           gpu_name=torch.cuda.get_device_name(0),wall_s=time.perf_counter()-start,
           files={f:controls.sha(d / f) for f in ("training.npz","probes.json.gz","checkpoint_afterA.pt","checkpoint_afterB.pt")})
    (d / "result.json").write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(dict(configuration=configuration,seed=seed,dry=dry,status="complete",wall_s=r["wall_s"])),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=["prepare","dry","fit"])
    ap.add_argument("configuration",nargs="?");ap.add_argument("seed",type=int,nargs="?");a=ap.parse_args()
    torch.set_num_threads(1)
    if a.command=="prepare":prepare()
    else:
        assert os.environ.get("CUDA_VISIBLE_DEVICES")=="1" and os.environ.get("CUDA_DEVICE_ORDER")=="PCI_BUS_ID"
        fit(a.configuration,a.seed,a.command=="dry")
