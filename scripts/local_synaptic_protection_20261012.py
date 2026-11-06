"""Source-development protection variants, sharing checked local/BP credit.

None/meta/slow/both separate the mechanisms. ``uniform_meta`` is a diagnostic:
its tensor-wise mean attenuation preserves the amount of gating but removes
synapse selectivity. It is deliberately not described as synapse-local gating.
"""
import copy
import json

import torch

from local_continual_actor_critic import LocalContinualAC


class ProtectedAC(LocalContinualAC):
    def __init__(self, *args, protection="both", **kwargs):
        assert protection in ("none", "meta", "slow", "both", "uniform_meta")
        super().__init__(*args, protection="none", **kwargs)
        self.protection = protection

    @torch.no_grad()
    def learn(self, x, actions, rewards, done, next_x):
        a, c, ent, activity, _, value = self.instantaneous(x, actions)
        nxt = self.forward(next_x)[3]
        delta = rewards + self.gamma * (~done).to(x.dtype) * nxt - value
        if not torch.isfinite(delta).all():
            raise FloatingPointError("non-finite TD signal")
        availability, before = {}, {}
        for k, p in self.plastic_params.items():
            self.actor_trace[k].mul_(self.gamma * self.lam).add_(a[k])
            self.critic_trace[k].mul_(self.gamma * self.lam).add_(c[k])
            shape = (len(x),) + (1,) * p.ndim
            g = (delta.view(shape) * (self.actor_trace[k] + self.vf * self.critic_trace[k])).mean(0)
            g.add_(ent[k].mean(0), alpha=self.entropy)
            g.clamp_(-.1, .1)
            if self.protection in ("meta", "both", "uniform_meta"):
                self.meta[k].lerp_(self.meta_max * activity[k], 1 / self.meta_tau)
                gate = torch.exp(-(self.meta[k] * p).abs())
                availability[k] = gate.mean() if self.protection == "uniform_meta" else gate
                before[k] = p.clone()
            p.grad = g
        self.opt.step()
        for k, p in self.plastic_params.items():
            if k in availability:
                p.copy_(before[k] + availability[k] * (p - before[k]))
            if self.protection in ("slow", "both"):
                gate = availability[k] if self.protection == "both" else 1.
                p.add_(self.consolidation * gate * activity[k] * (self.slow[k] - p))
                self.slow[k].lerp_(p, 1 / self.slow_tau)
            self.actor_trace[k][done] = 0
            self.critic_trace[k][done] = 0
            if not torch.isfinite(p).all():
                raise FloatingPointError("non-finite learned parameter")
        self.count["frames"] += len(x)
        self.count["updates"] += 1
        self.count["parameter_updates"] += sum(p.numel() for p in self.plastic_params.values())
        return delta

    @torch.no_grad()
    def plasticity_snapshot(self):
        rows = {}
        for k, p in self.plastic_params.items():
            gate = torch.exp(-(self.meta[k] * p).abs())
            used = self.protection in ("meta", "both", "uniform_meta")
            rows[k] = dict(mean_availability=float(gate.mean()) if used else 1.,
                           min_availability=float(gate.min()) if used else 1.,
                           max_availability=float(gate.max()) if used else 1.,
                           mean_metaplastic_state=float(self.meta[k].mean()),
                           reference_displacement_l2=float((p - self.slow[k]).norm()))
        return dict(protection=self.protection, state_by_parameter=rows,
                    temporal_unit="learner update, shared 16-replica step; not biological seconds")


def check():
    torch.set_num_threads(1)
    results = []
    for rule in ("local", "bp", "readout"):
        old = LocalContinualAC(10, 4, (7, 6), n_envs=3, rule=rule, seed=91)
        new = ProtectedAC(10, 4, (7, 6), n_envs=3, rule=rule, seed=91, protection="none")
        rng = torch.Generator().manual_seed(981)
        for step in range(12):
            x = torch.randn(3, 10, generator=rng)
            nx = torch.randn(3, 10, generator=rng)
            action = torch.randint(4, (3,), generator=rng)
            reward = torch.rand(3, generator=rng)
            done = torch.tensor([step % 3 == 0, False, step % 4 == 0])
            old.learn(x, action, reward, done, nx)
            new.learn(x, action, reward, done, nx)
        assert all(torch.equal(old.params[k], v) for k, v in new.params.items())
        assert all(torch.equal(old.actor_trace[k], v) for k, v in new.actor_trace.items())
        assert old.count == new.count
        # Mechanism-only variants match the original checked optional rules.
        for protection in ("meta", "slow", "both"):
            a = LocalContinualAC(10, 4, (7, 6), n_envs=3, rule=rule, seed=91, protection=protection)
            b = ProtectedAC(10, 4, (7, 6), n_envs=3, rule=rule, seed=91, protection=protection)
            for _ in range(5):
                a.learn(x, action, reward, done, nx); b.learn(x, action, reward, done, nx)
            assert all(torch.equal(a.params[k], v) for k, v in b.params.items())
        results.append(dict(rule=rule, original_none_exact=True, original_protection_variants_exact=True))
    print(json.dumps(dict(ok=True, checks=results, source_fits=0), indent=2))


if __name__ == "__main__":
    check()
