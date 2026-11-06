"""Continuous fresh visual rooms with natural episode transitions.

Reuses the pre-outcome fresh geometry and source-only localization atlas. No
policy receives room, goal, phase or simulator position. Geometry is observable
only through rendered partial views/contact; some views can remain ambiguous.
Curriculum changes which room is sampled at the next natural episode, without
forcing a terminal or resetting learner state. Final budget termination is
recorded separately. This file produces no trained-policy outcome.
"""
import argparse
import hashlib
import json

import numpy as np

import wacv_native_prior_confirm_env as fresh
from arc_fly.unimaze import bfs, NB

ROOT=fresh.ROOT
BASE=ROOT / "data/audit/wacv_continual_geometry_preflight_20261012"


class RoomCache(fresh.RawCache):
    def __init__(self,source_cache,manifest):
        super().__init__(source_cache,manifest,(3,3))
        self.goals={r["configuration"]:tuple(r[r["stage"]+"_goal"]) for r in manifest["rows"]}
        self.geometry_digest=hashlib.sha256(json.dumps(manifest["rows"],sort_keys=True).encode()).hexdigest()
        self.memo={}

    def frame(self,cfg,cell,heading,rotation):
        # This instance has its own immutable geometry bank, avoiding the old
        # global memo's alias if a later experiment reuses configuration IDs.
        return super().frame(cfg,cell,heading,rotation)


class ContinuousRooms:
    def __init__(self,cache,n=16,seed=0,phase_frames=65536,pair=0,schedule=None):
        self.cache=cache;self.n=int(n);self.rng=np.random.default_rng(seed)
        self.phase_frames=int(phase_frames);self.limit=2*self.phase_frames
        assert self.limit%self.n==0 and self.phase_frames%self.n==0
        assert 0<=pair<8 and (schedule is None or self.n==1)
        self.pair=pair;self.schedule=schedule;self.frames=0;self.episode_counter=0
        self.worlds=[None]*self.n;self.completed=[];self.exhausted=False
        self.obs_dim=150;self.n_act=4;self.transitions=[];self._distances={}

    def _new_world(self):
        if self.schedule is not None:
            if self.episode_counter>=len(self.schedule):self.exhausted=True;return None
            row=self.schedule[self.episode_counter]
            cfg=int(row["configuration"]);spawn=tuple(row["spawn"]);rot=int(row["rotation_index"])
        else:
            cfg=100+2*self.pair+int(self.frames>=self.phase_frames)
            _,starts=self.cache.geometry(cfg)
            spawn=tuple(starts[int(self.rng.integers(len(starts)))])
            rot=int(self.rng.integers(len(self.cache.rotations)))
        w,_=self.cache.geometry(cfg);assert w[spawn] and spawn!=self.cache.goals[cfg]
        heading=NB[fresh.source.inward_heading(spawn)]
        out=dict(configuration=cfg,spawn=spawn,position=spawn,heading=heading,
                 rotation_index=rot,decisions=0,travelled=0,contact=False,last_action=None,
                 reward=0.,episode=self.episode_counter,path=[spawn],actions=[])
        self.episode_counter+=1;return out

    def _sense(self,w):
        if w is None:return np.zeros(150,np.float32)
        frame=self.cache.frame(w["configuration"],w["position"],w["heading"],w["rotation_index"])
        action=np.zeros(4,np.float32)
        if w["last_action"] is not None:action[w["last_action"]]=1
        return np.concatenate((frame,np.array([w["contact"],w["reward"]],np.float32),action))

    def reset(self):
        assert self.frames==0
        self.worlds=[self._new_world() for _ in range(self.n)]
        return np.stack([self._sense(w) for w in self.worlds])

    def step(self,actions):
        assert self.frames<self.limit and not self.exhausted
        actions=np.asarray(actions);assert actions.shape==(self.n,) and np.issubdtype(actions.dtype,np.integer)
        assert np.all((actions>=0)&(actions<4))
        self.frames+=self.n;final=self.frames==self.limit
        rewards=np.zeros(self.n,np.float32);done=np.zeros(self.n,bool);observations=[]
        for i,a in enumerate(actions):
            a=int(a);world=self.worlds[i];cfg=world["configuration"];goal=self.cache.goals[cfg]
            walk,_=self.cache.geometry(cfg);at=world["position"]
            q=tuple(x+y for x,y in zip(at,NB[a]));valid=all(0<=z<10 for z in q) and bool(walk[q])
            world["position"]=q if valid else at;world["contact"]=not valid
            world["heading"]=NB[a];world["last_action"]=a;world["decisions"]+=1
            world["travelled"]+=int(valid);world["path"].append(world["position"]);world["actions"].append(a)
            success=world["position"]==goal;world["reward"]=rewards[i]=float(success)
            done[i]=success or world["decisions"]==45 or final
            terminal=self._sense(world)
            if done[i]:
                key=cfg,world["spawn"]
                if key not in self._distances:self._distances[key]=int(bfs(walk,world["spawn"])[goal])
                shortest=self._distances[key]
                self.completed.append(dict(episode=world["episode"],slot=i,configuration=cfg,spawn=list(world["spawn"]),
                    rotation_index=world["rotation_index"],decisions=world["decisions"],travelled=world["travelled"],
                    success=bool(success),reward_sum=float(success),spl=float(success)*shortest/max(shortest,world["travelled"],1),
                    decision_efficiency=float(success)*shortest/max(shortest,world["decisions"],1),
                    forced_boundary=bool(final),path=[list(p) for p in world["path"]],actions=list(world["actions"])))
                if not final:
                    self.worlds[i]=self._new_world()
                    if self.worlds[i] is not None and self.worlds[i]["configuration"]!=cfg:
                        self.transitions.append(dict(slot=i,step=self.frames,from_configuration=cfg,
                                                     to_configuration=self.worlds[i]["configuration"],natural_terminal=True))
                    observations.append(self._sense(self.worlds[i]))
                else:observations.append(terminal)
            else:observations.append(terminal)
        return np.stack(observations).astype(np.float32),rewards,done


class VisualRooms:
    def __init__(self,physical,source_cache,tier="clean",noise_seed=0):
        assert tier in ("clean","noisy")
        self.env=physical;self.n=physical.n;self.tier=tier;self.rng=np.random.default_rng(noise_seed)
        atlas=source_cache.raw.features[0,:,:,:,0,:64].reshape(100,4,64)
        self.filters=[fresh.LogBelief(atlas,.03) for _ in range(self.n)]
        self.obs_dim=252;self.n_act=4;self.pose_audit=[]

    def __getattr__(self,key):return getattr(self.env,key)

    def augment(self,observations,actions=None,done=None):
        out=[]
        for i,(x,filt) in enumerate(zip(observations,self.filters)):
            x=x.copy()
            if self.tier=="noisy":x[:144]=np.clip(x[:144]+self.rng.normal(0,.1,144),-1,1)
            reset=actions is None or bool(done[i])
            if reset:filt.reset()
            if self.env.exhausted:p=np.zeros(100,np.float32)
            else:p=filt.observe(x[:64],None if reset else int(actions[i]),bool(x[144])).astype(np.float32)
            out.append(np.concatenate((x,p,p@fresh.source.parent.COORD)))
            # Separate diagnostics, computed after policy input; no true pose
            # or geometry is accessed by the localization filter itself.
            if not self.env.exhausted:
                cell=self.env.worlds[i]["position"];s=cell[0]*10+cell[1]
                self.pose_audit.append((self.env.frames,i,int(p.argmax()),s,float(p[s])))
        result=np.asarray(out,np.float32);assert result.shape==(self.n,252) and np.isfinite(result).all()
        return result

    def reset(self):return self.augment(self.env.reset())

    def step(self,actions):
        obs,rewards,done=self.env.step(actions)
        return self.augment(obs,actions,done),rewards,done


def check():
    _,_,source=fresh.load()
    manifest=json.loads((fresh.BASE / "geometry_manifest.json").read_text())
    rows=[]
    for pair in range(8):
        for tier in ("clean","noisy"):
            cache=RoomCache(source,manifest)
            physical=ContinuousRooms(cache,n=4,seed=9381+pair,phase_frames=256,pair=pair)
            env=VisualRooms(physical,source,tier,seed_for_noise:=[7311,pair])
            obs=env.reset();rng=np.random.default_rng([237,pair]);credited=0.
            while env.frames<env.limit:
                actions=rng.integers(4,size=env.n)
                before=[(w["configuration"],w["position"],w["decisions"]) for w in env.worlds]
                obs,rewards,done=env.step(actions);credited+=float(rewards.sum())
                for i,(cfg,at,t) in enumerate(before):
                    walk,_=cache.geometry(cfg);q=tuple(u+v for u,v in zip(at,NB[int(actions[i])]))
                    q=q if all(0<=z<10 for z in q) and walk[q] else at
                    reward=float(q==cache.goals[cfg])
                    assert rewards[i]==reward
                    assert bool(done[i])==(bool(reward) or t+1==45 or env.frames==env.limit)
                assert obs.shape==(4,252)
                if env.frames==env.phase_frames:
                    assert not any(e["forced_boundary"] for e in env.completed)
            assert env.frames==512 and sum(e["decisions"] for e in env.completed)==512
            assert sum(e["reward_sum"] for e in env.completed)==credited
            assert len(env.transitions)==env.n and all(t["natural_terminal"] for t in env.transitions)
            assert all(256<=t["step"]<=256+4*45 for t in env.transitions)
            pose=np.array(env.pose_audit)
            rows.append(dict(pair=pair,tier=tier,frames=512,natural_transitions=env.transitions,
                             pose_map_accuracy=float(np.mean(pose[:,2]==pose[:,3]))))
    BASE.mkdir(parents=True,exist_ok=True)
    out=dict(ok=True,code_sha256=fresh.source.parent.sha(__file__),geometry_sha256=fresh.source.parent.sha(fresh.BASE / "geometry_manifest.json"),
             rows=rows,total_random_interactions=16*512,trained_policy_cases=0,
             interpretation="Technical physics/sensor/state continuity only; random-policy performance is not a scientific result.")
    (BASE / "preflight.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="rows"}))


if __name__=="__main__":
    argparse.ArgumentParser().parse_args();check()
