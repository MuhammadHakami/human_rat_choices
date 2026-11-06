"""Fresh registered geometry and noisy causal observations; no learner fit.

Room/action priors and source atlas remain explicit. Policy inputs contain no
goal, layout, phase, true position or future action label. Pose audit metadata
is collected separately after the observation is constructed.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import wacv_native_behavior_prior_20261012 as source
from wacv_embodied_source_env import SourceBatch
from wacv_embodied_source_cache import pool_pixels
from tartarus_visual_env import View, render

ROOT=source.ROOT
PROTOCOL=ROOT/'docs/WACV_NATIVE_PRIOR_CONFIRMATION_PROTOCOL_20261012.json'
SHA='fcd1c7f51fed553710673cf0b4514a54bfda4e5c75f5bde961976c3a9563785d'
ADDENDUM=ROOT/'docs/WACV_NATIVE_PRIOR_CONFIRMATION_ADDENDUM1_20261012.json'
ADD_SHA='2240c6448261d04266906dfeaf41d2f7eccf34f46a8984b9c78716d1bfe46fde'
BASE=ROOT/'data/audit/wacv_native_prior_confirmation_20261012'
RENDER_MEMO={}


def load():
    assert source.parent.sha(PROTOCOL)==SHA and source.parent.sha(ADDENDUM)==ADD_SHA
    P=json.loads(PROTOCOL.read_text());A=json.loads(ADDENDUM.read_text())
    assert A['amends_sha256']==SHA
    _,cache,_=source.load()
    return P,A,cache


def generate():
    P,A,cache=load();rng=np.random.default_rng(P['geometry']['seed'])
    starts=list(map(tuple,cache.geometry(1)[1]));goal=tuple(A['goal_override']['A_goal'])
    rows=[]
    for pair,bg in enumerate(A['goal_override']['B_goals_by_pair']):
        bg=tuple(bg);reserved=set(starts)|{goal,bg}
        candidates=[s for s in range(100) if divmod(s,10) not in reserved]
        for attempt in range(1,10001):
            blocked=[sorted(map(int,rng.choice(candidates,18,replace=False))) for _ in range(2)]
            if len(set(blocked[0])^set(blocked[1]))<20:continue
            maps=[];ok=True
            for cells in blocked:
                w=np.ones(100,bool);w[cells]=False;w=w.reshape(10,10)
                for g in (goal,bg):
                    d=source.goal_distances(w,g)
                    ok &= len(d)==int(w.sum()) and all(s in d and d[s]<=30 for s in starts)
                maps.append(w)
            if ok:break
        else:raise RuntimeError('Registered geometry rejection budget exhausted before policy evaluation')
        for stage,w,cells in zip(('A','B'),maps,blocked):
            rows.append(dict(pair=pair,stage=stage,configuration=100+2*pair+int(stage=='B'),
                             blocked=cells,walkable=w.astype(int).tolist(),
                             A_goal=list(goal),B_goal=list(bg),attempts=attempt))
    BASE.mkdir(parents=True,exist_ok=True)
    manifest=dict(protocol_sha256=SHA,addendum_sha256=ADD_SHA,generator_sha256=source.parent.sha(__file__),
                  starts=list(map(list,starts)),rows=rows,source_models=P['source_models'],policy_results_seen=False)
    with (BASE/'geometry_manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(pairs=8,worlds=len(rows),policy_results_seen=False,
                         geometry_manifest_sha256=source.parent.sha(BASE/'geometry_manifest.json'))))


class LogBelief:
    """Same Bayes observation/motion model with stable log-domain arithmetic."""
    def __init__(self,atlas,sigma=.03):
        self.atlas=atlas.copy().astype(np.float64);self.sigma=sigma;self.reset()

    def reset(self):self.logp=np.full((10,10),-np.log(100.),np.float64)

    def observe(self,cue,action=None,contact=False):
        if action is not None and not contact:
            dr,dc=source.MOVES[int(action)]
            self.logp=np.roll(self.logp,(dr,dc),axis=(0,1))
            if dr==1:self.logp[0]=-np.inf
            if dr==-1:self.logp[-1]=-np.inf
            if dc==1:self.logp[:,0]=-np.inf
            if dc==-1:self.logp[:,-1]=-np.inf
        templates=self.atlas if action is None else self.atlas[:,int(action)]
        ll=-np.square(templates-np.asarray(cue,np.float64)).mean(-1)/(2*self.sigma**2)
        if action is None:ll=logsumexp(ll,axis=1)-np.log(4.)
        lp=self.logp+ll.reshape(10,10);norm=logsumexp(lp)
        if not np.isfinite(norm):raise FloatingPointError('Invalid observation/motion model support')
        self.logp=lp-norm
        return np.exp(self.logp).ravel()


class RawCache:
    """Only the environment renders known geometry; neither observer nor policy gets it."""
    def __init__(self,source_cache,manifest,goal):
        self.protocol=source_cache.raw.protocol
        self.features=source_cache.raw.features  # source400-view atlas only
        self.goal=tuple(goal);self.rotations=source_cache.rotations;self.obs_dim=150
        self.worlds={r['configuration']:np.asarray(r['walkable'],bool) for r in manifest['rows']}
        self.index={k:i for i,k in enumerate(self.worlds)}
        self.starts=list(map(tuple,manifest['starts']));self.memo=RENDER_MEMO
        self.view=View(**self.protocol['view'])

    def geometry(self,cfg):return self.worlds[cfg],self.starts

    def frame(self,cfg,cell,heading,rotation):
        key=cfg,tuple(cell),tuple(heading),int(rotation)
        if key not in self.memo:
            w,_=self.geometry(cfg)
            img,_,_=render(self.view,w,tuple(cell),tuple(heading),self.rotations[int(rotation)])
            _,x=pool_pixels(img,self.view,self.protocol['features']);self.memo[key]=x
        return self.memo[key].copy()


class PhysicalBatch(SourceBatch):
    """Unchanged source step/sense/reward semantics; fresh fixed-schedule worlds."""
    def _new_world(self):
        assert self.n==1 and self.schedule is not None
        if self.episode_counter>=len(self.schedule):self.exhausted=True;return None
        row=self.schedule[self.episode_counter];cfg=int(row['configuration']);spawn=tuple(row['spawn'])
        w,_=self.cache.geometry(cfg)
        assert w[spawn] and spawn!=self.cache.goal
        heading=source.MOVES[source.inward_heading(spawn)]
        state=dict(configuration=cfg,spawn=spawn,position=spawn,heading=heading,
                   rotation_index=int(row['rotation_index']),decisions=0,travelled=0,
                   contact=False,last_action=None,reward=0.,episode=self.episode_counter,path=[spawn],actions=[])
        self.episode_counter+=1
        return state


class Observations:
    def __init__(self,raw,source_cache,schedule,frames,tier,noise_seed):
        physical=copy.copy(raw);physical.protocol=source_cache.protocol
        self.env=PhysicalBatch(physical,n=1,seed=0,frames=frames,schedule=schedule)
        atlas=source_cache.raw.features[0,:,:,:,0,:64].reshape(100,4,64)
        self.filter=LogBelief(atlas,.03);self.obs_dim=252;self.tier=tier
        assert tier in ('clean','noisy')
        self.rng=np.random.default_rng(noise_seed);self.pose_audit=[]

    def __getattr__(self,key):return getattr(self.env,key)

    def augment(self,obs,action=None,done=None):
        x=obs[0].copy()
        if self.tier=='noisy':x[:144]=np.clip(x[:144]+self.rng.normal(0,.1,144),-1,1)
        reset=action is None or bool(done[0])
        if reset:self.filter.reset()
        if self.env.exhausted:p=np.zeros(100,np.float32)
        else:p=self.filter.observe(x[:64],None if reset else int(action[0]),bool(x[144])).astype(np.float32)
        out=np.concatenate((x,p,p@source.parent.COORD))[None]
        assert out.shape==(1,252) and np.isfinite(out).all()
        # Evaluation metadata only; never supplied to filter or policy.
        if not self.env.exhausted:
            cell=self.env.worlds[0]['position'];true=cell[0]*10+cell[1]
            self.pose_audit.append(dict(step=self.env.frames,MAP=int(p.argmax()),true=true,
                                       true_mass=float(p[true]),max_mass=float(p.max())))
        return out.astype(np.float32)

    def reset(self):return self.augment(self.env.reset())

    def step(self,actions):
        obs,r,d=self.env.step(actions);return self.augment(obs,actions,d),r,d


def check():
    P,A,cache=load();m=json.loads((BASE/'geometry_manifest.json').read_text())
    assert m['protocol_sha256']==SHA and m['addendum_sha256']==ADD_SHA and len(m['rows'])==16
    from wacv_embodied_belief_source import LandmarkBelief
    atlas=cache.raw.features[0,:,:,:,0,:64].reshape(100,4,64)
    old=LandmarkBelief(atlas,.03);new=LogBelief(atlas,.03);maxerr=0.
    env=source.parent.LongBatch(cache,n=1,seed=0,frames=128);obs=env.reset();rng=np.random.default_rng(771)
    for i in range(129):
        action=None if i==0 or done[0] else int(a[0])
        if action is None:old.reset();new.reset()
        old.observe(obs[0,:64],action,bool(obs[0,144]));p=new.observe(obs[0,:64],action,bool(obs[0,144]))
        maxerr=max(maxerr,float(np.max(np.abs(old.p.ravel()-p))))
        if i==128:break
        a=rng.integers(4,size=1);obs,_,done=env.step(a)
    assert maxerr<1e-10
    # Longer noisy physical streams test only observer numerics and terminal plumbing.
    # No scientific policy/controller result is produced by this technical check.
    trials=[]
    for tier in ('clean','noisy'):
        raw=RawCache(cache,m,A['goal_override']['B_goals_by_pair'][0]);cfg=101
        schedule=[dict(configuration=cfg,spawn=m['starts'][i%10],rotation_index=i%3) for i in range(512)]
        e=Observations(raw,cache,schedule,1024,tier,[771,1]);x=e.reset()
        for _ in range(1024):
            x,_,_=e.step(rng.integers(4,size=1));assert np.isfinite(e.filter.logp[np.isfinite(e.filter.logp)]).all()
        assert sum(v['decisions'] for v in e.completed)==1024
        trials.append(dict(tier=tier,technical_frames=1024,finite=True))
    out=dict(ok=True,old_log_filter_max_abs_error=maxerr,technical_streams=trials,
             protocol_sha256=SHA,addendum_sha256=ADD_SHA,code_sha256=source.parent.sha(__file__),
             geometry_manifest_sha256=source.parent.sha(BASE/'geometry_manifest.json'),scientific_policy_cases=0)
    (BASE/'preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['generate','check']);a=ap.parse_args()
    {'generate':generate,'check':check}[a.cmd]()
