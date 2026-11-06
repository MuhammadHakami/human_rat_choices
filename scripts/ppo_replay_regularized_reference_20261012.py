"""PPO plus fixed-budget reservoir policy/value cloning.

CLEAR-inspired reference, not exact CLEAR: the on-policy objective is the
reviewed PPO, with one additional supervised cloning step per rollout. There
is no off-policy RL update, supplied task label or boundary-dependent store.
Teachers are causal pre-update predictions from experienced observations.
Reservoir storage and extra optimizer steps are explicitly reported.
"""
import json

import numpy as np
import torch

from ppo_feedforward_reference import FeedforwardPPO


class ReplayRegularizedPPO(FeedforwardPPO):
    def __init__(self,*args,replay_capacity=620,replay_batch=128,
                 policy_cloning=1.,value_cloning=.5,**kwargs):
        super().__init__(*args,**kwargs)
        assert replay_capacity>0 and replay_batch>0
        self.replay_capacity=int(replay_capacity);self.replay_batch=int(replay_batch)
        self.policy_cloning=policy_cloning;self.value_cloning=value_cloning
        self.replay_obs=torch.zeros(self.replay_capacity,self.obs_dim)
        self.replay_policy=torch.zeros(self.replay_capacity,self.net.pi.out_features)
        self.replay_value=torch.zeros(self.replay_capacity)
        self.replay_rng=np.random.default_rng([902117,kwargs.get("seed",0)])
        self.replay_seen=0;self.replay_size=0
        self.count.update(replay_optimizer_steps=0,replay_presentations=0)

    @torch.no_grad()
    def remember(self,B):
        observations=B["obs"].detach().reshape(-1,self.obs_dim).cpu()
        policies=B["probs"].detach().reshape(-1,self.replay_policy.shape[-1]).cpu()
        values=B["val"].detach().reshape(-1).cpu()
        for x,p,v in zip(observations,policies,values):
            if self.replay_seen<self.replay_capacity:index=self.replay_seen
            else:index=int(self.replay_rng.integers(self.replay_seen+1))
            self.replay_seen+=1
            if index<self.replay_capacity:
                self.replay_obs[index].copy_(x);self.replay_policy[index].copy_(p);self.replay_value[index].copy_(v)
        self.replay_size=min(self.replay_seen,self.replay_capacity)

    def replay_loss(self,indices):
        x=self.replay_obs[indices][None];n=x.shape[1]
        dummy=(torch.zeros(1,n,0),torch.zeros(1,n,0))
        o,_=self.net.unroll(x,dummy,torch.zeros(1,n))
        dist,values=self.net.heads(o[0])
        teacher=self.replay_policy[indices].to(dist.probs.device)
        target_values=self.replay_value[indices].to(values.device)
        kl=(teacher*(teacher.clamp_min(1e-12).log()-dist.probs.clamp_min(1e-12).log())).sum(-1).mean()
        mse=.5*(values-target_values).square().mean()
        return self.policy_cloning*kl+self.value_cloning*mse

    def update(self,B,state0):
        super().update(B,state0)
        if self.replay_size:
            count=min(self.replay_batch,self.replay_size)
            indices=torch.tensor(self.replay_rng.choice(self.replay_size,count,replace=False))
            loss=self.replay_loss(indices)
            self.opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.net.parameters(),.5);self.opt.step()
            self.count["replay_optimizer_steps"]+=1;self.count["replay_presentations"]+=count
            self.count["weight_updates"]+=sum(p.numel() for p in self.net.parameters())
        # Teachers belong to the collector's pre-update policy. New observations
        # cannot enter the prior-memory gradient before their on-policy update.
        self.remember(B)

    def memory_bytes(self):
        result=super().memory_bytes()
        result["replay_buffer"]=sum(t.numel()*t.element_size() for t in (self.replay_obs,self.replay_policy,self.replay_value))
        result["replay_capacity"]=self.replay_capacity;result["replay_size"]=self.replay_size
        result["replay_seen"]=self.replay_seen
        return result


def check():
    from ppo_feedforward_reference import Bandit
    from local_continual_actor_critic import LocalContinualAC
    torch.set_num_threads(1)
    # Reservoir replacement uses instance RNG; perturbing global generators
    # does not affect any retained sample or the learned parameters.
    def run(disturb):
        p=ReplayRegularizedPPO(10,4,n_envs=4,n_steps=16,seed=77,replay_capacity=23,replay_batch=11)
        env=Bandit(4,seed=89);obs=env.reset();state=(torch.zeros(1,4,0),torch.zeros(1,4,0));done=np.zeros(4,np.float32)
        for _ in range(4):
            if disturb:np.random.rand(9);torch.rand(7)
            B,s0,obs,state,done=p.rollout(env,obs,state,done);p.update(B,s0)
        return p
    first=run(False);second=run(True)
    assert first.replay_seen==256 and first.replay_size==23
    assert first.count["replay_optimizer_steps"]==3 and first.count["replay_presentations"]==33
    assert torch.equal(first.replay_obs,second.replay_obs)
    assert torch.equal(first.replay_policy,second.replay_policy)
    assert torch.equal(first.replay_value,second.replay_value)
    assert all(torch.equal(a,b) for a,b in zip(first.net.parameters(),second.net.parameters()))
    assert not any(t.requires_grad or t.grad_fn is not None for t in (first.replay_obs,first.replay_policy,first.replay_value))
    # Independent central differences check the actual cloning loss, including
    # policy KL direction and the value coefficient.
    indices=torch.arange(11);first.opt.zero_grad();first.replay_loss(indices).backward()
    errors=[]
    for parameter,index in [(first.net.pi.weight,(0,0)),(first.net.v.weight,(0,0))]:
        gradient=float(parameter.grad[index]);eps=.001
        with torch.no_grad():
            parameter[index]+=eps;plus=float(first.replay_loss(indices))
            parameter[index]-=2*eps;minus=float(first.replay_loss(indices));parameter[index]+=eps
        errors.append(abs((plus-minus)/(2*eps)-gradient))
    assert max(errors)<2e-4
    # At the exact source configuration, the 620-entry replay PPO's actual
    # allocated training storage stays below full-layer eligibility AC. This
    # comparison excludes shared sensory scaffolds and temporary tensors.
    p=ReplayRegularizedPPO(252,4,n_envs=16,n_steps=128,seed=81)
    T,N=128,16
    B={"obs":torch.zeros(T,N,252),"act":torch.zeros(T,N,dtype=torch.int64),"probs":torch.ones(T,N,4)/4}
    for key in ("logp","val","rew","done_before","done_after","adv","ret"):B[key]=torch.zeros(T,N)
    p.last_buffer=(B,(torch.zeros(1,N,0),torch.zeros(1,N,0)))
    # Allocate Adam storage through a real replay-gradient optimizer step.
    p.replay_policy.fill_(.25);p.opt.zero_grad();p.replay_loss(torch.arange(16)).backward();p.opt.step()
    local=LocalContinualAC(n_envs=N,rule="bp",seed=81)
    local.learn(torch.zeros(N,252),torch.zeros(N,dtype=torch.int64),torch.zeros(N),torch.zeros(N,dtype=torch.bool),torch.zeros(N,252))
    pm=p.memory_bytes();lm=local.memory_bytes()
    replay_total=sum(pm[k] for k in ("weights","optimiser","persistent_state","experience_buffer","replay_buffer"))
    local_total=sum(lm[k] for k in ("weights","optimizer","eligibility","slow_reference","metaplasticity","fixed_feedback"))
    assert replay_total<=local_total,(replay_total,local_total)
    print(json.dumps(dict(ok=True,reservoir_rng_isolated=True,causal_detached_teachers=True,cloning_fd_error=max(errors),
                         source_replay_ppo_tensor_bytes=replay_total,source_bp_eligibility_tensor_bytes=local_total,
                         source_replay_capacity=620,scientific_navigation_cases=0),indent=2))


if __name__=="__main__":check()
