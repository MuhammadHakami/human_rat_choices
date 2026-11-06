"""Registered visual-context continual development on pair0, GPU1 only."""
import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch

import wacv_continual_ppo_source_20261012 as ppo_source
import wacv_continual_ppo_tuning_source_20261012 as ppo_tuning
import wacv_continual_geometry_env_20261012 as worlds
from local_synaptic_protection_20261012 import ProtectedAC
from ppo_replay_regularized_reference_20261012 import ReplayRegularizedPPO
from wacv_native_behavior_prior_20261012 import DeviceFFAgent
from ppo_feedforward_reference import FeedforwardPPO

ROOT=worlds.ROOT
BASE=ROOT / "data/audit/wacv_continual_geometry_development_20261012"
PROTOCOL=ROOT / "docs/WACV_CONTINUAL_GEOMETRY_DEVELOPMENT_PROTOCOL_20261012.json"
ADDENDUM=ROOT / "docs/WACV_CONTINUAL_GEOMETRY_DEVELOPMENT_ADDENDUM1_20261012.json"
ADDENDUM2=ROOT / "docs/WACV_CONTINUAL_GEOMETRY_DEVELOPMENT_ADDENDUM2_20261012.json"
MANIFEST=BASE / "manifest_v2.json"
CODE=["scripts/wacv_continual_geometry_development_20261012.py", "scripts/wacv_continual_geometry_env_20261012.py",
      "scripts/local_continual_actor_critic.py", "scripts/local_synaptic_protection_20261012.py",
      "scripts/recurrent_ppo.py", "scripts/ppo_feedforward_reference.py",
      "scripts/ppo_replay_regularized_reference_20261012.py"]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def binding():
    return dict(protocol=sha(PROTOCOL),addendum=sha(ADDENDUM),addendum2=sha(ADDENDUM2),code={s:sha(ROOT / s) for s in CODE},
                geometry=sha(worlds.fresh.BASE / "geometry_manifest.json"),
                source=ppo_source.controls.binding(),
                ppo_report=sha(ppo_source.BASE / "source_report.json"),
                ppo_audit=sha(ppo_source.BASE / "audit.json"),
                ppo_tuning_report=sha(ppo_tuning.BASE / "source_report.json"),
                ppo_tuning_audit=sha(ppo_tuning.BASE / "audit.json"))


def prepare():
    p=json.loads(PROTOCOL.read_text())
    status=json.loads((ppo_source.BASE / "status.json").read_text())
    report=json.loads((ppo_source.BASE / "source_report.json").read_text())
    audit=json.loads((ppo_source.BASE / "audit.json").read_text())
    assert status["status"]==report["status"]=="complete" and audit["ok"] and audit["cases"]==18
    assert report["audit_sha256"]==sha(ppo_source.BASE / "audit.json")
    tuned=json.loads((ppo_tuning.BASE / "source_report.json").read_text())
    tuned_audit=json.loads((ppo_tuning.BASE / "audit.json").read_text())
    assert tuned["status"]=="complete" and tuned["competent"] and tuned_audit["ok"] and tuned_audit["cases"]==54
    assert tuned["audit_sha256"]==sha(ppo_tuning.BASE / "audit.json")
    source_models=json.loads(ppo_source.PROTOCOL.read_text())["source_models"]
    methods=copy.deepcopy(p["methods"])
    for m in methods:
        if m["family"] in ("ppo","ppo_replay"):
            m["rate"]=tuned["selected"]["rate"]
            m["ppo_settings"]={k:tuned["selected"][k] for k in ("n_steps","ent")}
    BASE.mkdir(parents=True,exist_ok=True)
    with MANIFEST.open("x") as f:
        json.dump(dict(hashes=binding(),methods=methods,source_models=source_models,
                       registered_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                       withheld_pairs=p["withheld_confirmation_pairs"]),f,indent=2)
    print(json.dumps(dict(status="prepared",fits=p["fits"],target_results_seen=False)))


def source_model(seed):
    manifest=json.loads(MANIFEST.read_text());row=next(r for r in manifest["source_models"] if r["seed"]==seed)
    assert sha(ROOT / row["path"])==row["sha256"]
    return torch.load(ROOT / row["path"],map_location="cpu",weights_only=False)


def build(method,seed):
    initial=source_model(seed)
    if method["family"]=="eligibility":
        return ProtectedAC(n_envs=16,rule=method["credit"],protection=method["protection"],
                           lr=method["rate"],seed=27000+seed,device="cuda:0",source=initial.net)
    typ=ReplayRegularizedPPO if method["family"]=="ppo_replay" else FeedforwardPPO
    settings=json.loads(ppo_source.PROTOCOL.read_text())["ppo"]
    settings.update(method["ppo_settings"])
    m=typ(252,4,n_envs=16,lr=method["rate"],seed=27000+seed,
          **{k:v for k,v in settings.items() if k not in ("gradient_clip","adam_eps")})
    m.net=DeviceFFAgent(252,4).to("cuda:0");m.net.load_state_dict(initial.net.state_dict())
    m.opt=torch.optim.Adam(m.net.parameters(),lr=method["rate"],eps=1e-5)
    return m


def environment(n,seed,tier,phase_frames,schedule=None):
    _,_,source=worlds.fresh.load()
    geometry=json.loads((worlds.fresh.BASE / "geometry_manifest.json").read_text())
    cache=worlds.RoomCache(source,geometry)
    candidates=json.loads(ADDENDUM2.read_text())["goal_override"]["candidates"]
    for pair in range(8):
        a,b=100+2*pair,101+2*pair
        chosen=next(tuple(g) for g in candidates if tuple(g) not in cache.starts and tuple(g)!=cache.goals[b]
                    and tuple(g)!=(3,3) and cache.worlds[a][tuple(g)] and cache.worlds[b][tuple(g)])
        cache.goals[a]=chosen
    physical=worlds.ContinuousRooms(cache,n=n,seed=82000+seed,phase_frames=phase_frames,pair=0,schedule=schedule)
    return worlds.VisualRooms(physical,source,tier,noise_seed=[81013,seed,int(tier=="noisy")])


@torch.no_grad()
def sample(m,observations):
    if isinstance(m,ProtectedAC):return m.sample(torch.as_tensor(observations,device="cuda:0"))
    n=len(observations);dummy=(torch.zeros(1,n,0),torch.zeros(1,n,0))
    o,_=m.net.unroll(torch.as_tensor(observations)[None],dummy,torch.zeros(1,n))
    dist,v=m.net.heads(o[0]);a=torch.multinomial(dist.probs,1,generator=m.tg).squeeze(-1)
    return a,dist.probs,v


def tensor_parameters(m):return m.params if isinstance(m,ProtectedAC) else dict(m.net.named_parameters())


def probe(agent,stage,seed,tier,dry=False):
    cfg=100+int(stage=="B")
    geometry=json.loads((worlds.fresh.BASE / "geometry_manifest.json").read_text())
    schedule=[dict(configuration=cfg,spawn=s,rotation_index=rot) for rot in range(3) for s in geometry["starts"]]
    if dry:schedule=schedule[:2]
    m=copy.deepcopy(agent);device="cuda:0" if isinstance(m,ProtectedAC) else "cpu"
    m.tg=torch.Generator(device=device).manual_seed(230017+seed)
    initial_rng=m.tg.get_state().cpu().tolist()
    env=environment(1,seed,tier,45*len(schedule)//2,schedule)
    observations=env.reset();before={k:p.clone() for k,p in tensor_parameters(m).items()}
    probabilities,values=[],[]
    while not env.exhausted and env.frames<env.limit:
        action,prob,value=sample(m,observations)
        probabilities.append(prob[0].cpu().tolist());values.append(float(value[0]))
        observations,_,_=env.step(action.cpu().numpy())
    assert all(torch.equal(p,before[k]) for k,p in tensor_parameters(m).items())
    episodes=ppo_source.controls.parent.base.episodes(env,probabilities,values)
    alternative=env.cache.goals[101 if stage=="A" else 100]
    return dict(goals=sum(e["success"] for e in episodes),trials=len(episodes),frames=env.frames,
                mean_spl=float(np.mean([e["spl"] for e in episodes])),
                mean_decision_efficiency=float(np.mean([e["decision_efficiency"] for e in episodes])),
                alternative_goal_visit_fraction=float(np.mean([list(alternative) in e["path"] for e in episodes])),
                rng_initial=initial_rng,weights_frozen_verified=True,episodes=episodes,
                pose_audit=env.pose_audit)


class LoggingRooms:
    def __init__(self,env,observations):
        self.env=env;self.observations=observations
        self.log={k:[] for k in ("configurations","before_position","after_position","actions","rewards","done","obs_sha256","observations")}

    def step(self,actions):
        cfgs=[w["configuration"] for w in self.env.worlds];positions=[w["position"] for w in self.env.worlds]
        after=[]
        for cfg,pos,a in zip(cfgs,positions,actions):
            walk,_=self.env.cache.geometry(cfg);q=tuple(u+v for u,v in zip(pos,worlds.NB[int(a)]))
            after.append(q if all(0<=z<10 for z in q) and walk[q] else pos)
        digest=np.frombuffer(hashlib.sha256(self.observations.tobytes()).digest(),dtype=np.uint8).copy()
        obs,rewards,done=self.env.step(actions)
        for key,value in [("configurations",cfgs),("before_position",positions),("after_position",after),
                          ("actions",actions),("rewards",rewards),("done",done),("obs_sha256",digest),("observations",self.observations)]:
            self.log[key].append(np.array(value,copy=True))
        self.observations=obs;return obs,rewards,done


def resource_report(m,env):
    result=m.memory_bytes()
    if isinstance(m,ProtectedAC):
        result["active_components"]=dict(slow_reference=m.protection in ("slow","both"),
            metaplasticity=m.protection in ("meta","both","uniform_meta"),fixed_feedback=m.rule=="local")
    result["shared_sensory_scaffold"]=dict(source_atlas_allocated_bytes=sum(f.atlas.nbytes for f in env.filters),
                                          posterior_bytes=sum(f.logp.nbytes for f in env.filters),
                                          input_dimensions=252,teacher_pose_inputs=0)
    result["renderer_cache_bytes_engine_only"]=sum(x.nbytes for x in env.cache.memo.values())
    return result


def fit(method_id,seed,tier,dry=False):
    protocol=json.loads(PROTOCOL.read_text());manifest=json.loads(MANIFEST.read_text())
    assert manifest["hashes"]==binding() and seed in protocol["source_seeds"] and tier in protocol["sensory_tiers"]
    method=next(m for m in manifest["methods"] if m["id"]==method_id)
    if not dry:assert json.loads((BASE / "GO.json").read_text())["hashes"]==binding()
    d=BASE / ("dry" if dry else "cohort") / method_id / tier / f"seed{seed}"
    d.mkdir(parents=True,exist_ok=True)
    with (d / "started.json").open("x") as f:json.dump(dict(hashes=binding()),f)
    start=time.perf_counter();m=build(method,seed);count=2048 if dry else protocol["nominal_interactions_per_phase"]
    env=environment(16,seed,tier,count);observations=env.reset();wrapped=LoggingRooms(env,observations)
    initial_action_rng=m.tg.get_state().cpu().numpy().copy()
    phases=dict(A_initial=probe(m,"A",seed,tier,dry));probs,values=[],[]
    state=(torch.zeros(1,16,0),torch.zeros(1,16,0));done_previous=np.zeros(16,np.float32)
    while m.count["frames"]<2*count:
        if m.count["frames"]==count:
            phases["A_after_A_learning"]=probe(m,"A",seed,tier,dry)
            phases["B_before_learning"]=probe(m,"B",seed,tier,dry)
            torch.save(m,d / "checkpoint_afterA.pt")
        if isinstance(m,ProtectedAC):
            x=torch.as_tensor(observations,device="cuda:0");a,p,v=m.sample(x)
            next_observations,rewards,done=wrapped.step(a.cpu().numpy())
            m.learn(x,a,torch.as_tensor(rewards,device="cuda:0"),torch.as_tensor(done,device="cuda:0"),torch.as_tensor(next_observations,device="cuda:0"))
            probs.append(p.cpu().numpy());values.append(v.cpu().numpy());observations=next_observations
        else:
            B,s0,observations,state,done_previous=m.rollout(wrapped,observations,state,done_previous)
            probs.extend(B["probs"].numpy());values.extend(B["val"].numpy());m.update(B,s0)
    phases["A_after_B_learning"]=probe(m,"A",seed,tier,dry)
    phases["B_after_learning"]=probe(m,"B",seed,tier,dry)
    assert m.count["frames"]==env.frames==2*count
    torch.save(m,d / "checkpoint_afterB.pt")
    arrays={k:np.asarray(v) for k,v in wrapped.log.items()}
    np.savez_compressed(d / "training.npz",**arrays,probabilities=np.asarray(probs),values=np.asarray(values),
                        action_rng_initial=initial_action_rng,pose_audit=np.asarray(env.pose_audit))
    ppo_source.controls.parent.gz_write(d / "probes.json.gz",phases)
    realized={str(cfg):int((arrays["configurations"]==cfg).sum()) for cfg in (100,101)}
    assert sum(realized.values())==2*count and realized["100"]>=count
    rec=dict(method=method,seed=seed,tier=tier,dry=dry,status="complete",hashes=binding(),phase_frames=count,
             realized_interactions=realized,room_transitions=env.transitions,counters=m.count,
             probes={k:{a:b for a,b in v.items() if a not in ("episodes","rng_initial","pose_audit")} for k,v in phases.items()},
             resources=resource_report(m,env),wall_s=time.perf_counter()-start,gpu_name=torch.cuda.get_device_name(0),
             files={name:sha(d / name) for name in ("training.npz","probes.json.gz","checkpoint_afterA.pt","checkpoint_afterB.pt")})
    (d / "result.json").write_text(json.dumps(rec,indent=2)+"\n")
    print(json.dumps(dict(method=method_id,seed=seed,tier=tier,dry=dry,status="complete",wall_s=rec["wall_s"])),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=["prepare","dry","fit"])
    ap.add_argument("method",nargs="?");ap.add_argument("seed",nargs="?",type=int);ap.add_argument("tier",nargs="?")
    a=ap.parse_args();torch.set_num_threads(1)
    if a.command=="prepare":prepare()
    else:
        assert os.environ.get("CUDA_VISIBLE_DEVICES")=="1" and os.environ.get("CUDA_DEVICE_ORDER")=="PCI_BUS_ID"
        fit(a.method,a.seed,a.tier,a.command=="dry")
