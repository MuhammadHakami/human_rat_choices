"""Explicit online actor-critic credit and synaptic protection.

Two spatial credit rules share the same network, TD error, causal eligibility,
Adam, and protection. ``bp`` computes ordinary feedforward chain-rule gradients
explicitly; ``local`` uses fixed direct feedback, never transposed forward
weights for hidden credit. Both use exact local softmax-score output updates.
There is no temporal unrolling, future-return buffer, or backward call in the
learning path. Offline checkpoint pretraining is a separate, disclosed stage.

Protection is a TACOS-inspired rate-network adaptation, NOT a reproduction of
TACOS, not a spiking implementation, and not an established new algorithm.
Slow references and bounded activity metaplasticity have fixed state per
synapse. Batch replicas and Adam moments are included in resource accounting.
"""
import copy
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from ppo_feedforward_reference import FFAgent


class LocalContinualAC:
    def __init__(self, obs_dim=252, n_act=4, layers=(64, 64), n_envs=16,
                 rule="local", lr=3e-4, gamma=.99, lam=.95, entropy=.01,
                 value_coefficient=.5, protection="none", seed=0,
                 device="cpu", source=None, slow_tau=4096., meta_tau=2048.,
                 meta_max=25., consolidation=.001, feedback_scale=.1):
        assert rule in ("bp", "local", "readout")
        assert protection in ("none", "meta", "slow", "both")
        assert len(layers) == 2
        torch.manual_seed(seed)
        self.net = FFAgent(obs_dim, n_act, layers).to(device)
        if source is not None:
            self.net.load_state_dict(source.state_dict())
        # No autograd graph in either deployed update path.
        self.net.requires_grad_(False)
        self.rule, self.protection, self.device = rule, protection, device
        self.n_envs, self.n_act, self.obs_dim = n_envs, n_act, obs_dim
        self.gamma, self.lam = gamma, lam
        self.entropy, self.vf = entropy, value_coefficient
        self.slow_tau, self.meta_tau, self.meta_max = slow_tau, meta_tau, meta_max
        self.consolidation = consolidation
        self.params = dict(self.net.named_parameters())
        self.plastic_params = {k: p for k, p in self.params.items()
                               if rule != "readout" or k.startswith(("pi.", "v."))}
        self.opt = torch.optim.Adam(self.plastic_params.values(), lr=lr, eps=1e-5,
                                    maximize=True)
        rng = torch.Generator(device=device).manual_seed(seed + 917233)
        self.feedback = {}
        for layer, width in enumerate(layers if rule != "readout" else ()):
            self.feedback[layer] = {
                "actor": torch.randn(n_act, width, generator=rng, device=device)
                         * (feedback_scale / math.sqrt(n_act)),
                "critic": torch.randn(width, generator=rng, device=device)
                          * feedback_scale,
            }
        self.slow = {k: p.detach().clone() for k, p in self.plastic_params.items()}
        self.meta = {k: torch.zeros_like(p) for k, p in self.plastic_params.items()}
        self.actor_trace = {k: torch.zeros((n_envs, *p.shape), device=device)
                            for k, p in self.plastic_params.items()}
        self.critic_trace = {k: torch.zeros_like(t)
                             for k, t in self.actor_trace.items()}
        self.tg = torch.Generator(device=device).manual_seed(seed + 782191)
        self.count = dict(frames=0, updates=0, parameter_updates=0)

    @torch.no_grad()
    def forward(self, x):
        h1 = torch.tanh(F.linear(x, self.params["body.0.weight"], self.params["body.0.bias"]))
        h2 = torch.tanh(F.linear(h1, self.params["body.2.weight"], self.params["body.2.bias"]))
        logits = F.linear(h2, self.params["pi.weight"], self.params["pi.bias"])
        value = F.linear(h2, self.params["v.weight"], self.params["v.bias"]).squeeze(-1)
        return h1, h2, logits.softmax(-1), value

    @torch.no_grad()
    def sample(self, x):
        _, _, p, v = self.forward(x)
        a = torch.multinomial(p, 1, generator=self.tg).squeeze(-1)
        return a, p, v

    @torch.no_grad()
    def instantaneous(self, x, actions):
        """Per-replica eligibility and local activity, before reward arrives."""
        h1, h2, prob, val = self.forward(x)
        score = F.one_hot(actions, self.n_act).to(x.dtype) - prob
        ent = -(prob * prob.clamp_min(1e-12).log()).sum(-1)
        ent_signal = -prob * (prob.clamp_min(1e-12).log() + ent[:, None])
        actor = {k: torch.zeros((len(x), *p.shape), device=x.device, dtype=x.dtype)
                 for k, p in self.params.items()}
        critic = {k: torch.zeros_like(t) for k, t in actor.items()}
        entropy = {k: torch.zeros_like(t) for k, t in actor.items()}
        actor["pi.weight"] = score[:, :, None] * h2[:, None, :]
        actor["pi.bias"] = score
        critic["v.weight"] = h2[:, None, :]
        critic["v.bias"] = torch.ones((len(x), 1), device=x.device, dtype=x.dtype)
        entropy["pi.weight"] = ent_signal[:, :, None] * h2[:, None, :]
        entropy["pi.bias"] = ent_signal
        if self.rule == "bp":
            a2 = (score @ self.params["pi.weight"]) * (1 - h2.square())
            c2 = self.params["v.weight"].expand(len(x), -1) * (1 - h2.square())
            e2 = (ent_signal @ self.params["pi.weight"]) * (1 - h2.square())
            a1 = (a2 @ self.params["body.2.weight"]) * (1 - h1.square())
            c1 = (c2 @ self.params["body.2.weight"]) * (1 - h1.square())
            e1 = (e2 @ self.params["body.2.weight"]) * (1 - h1.square())
        elif self.rule == "local":
            # Direct broadcast to every layer; no hidden layer receives credit
            # propagated through another layer or a current forward weight.
            a1 = (score @ self.feedback[0]["actor"]) * (1 - h1.square())
            c1 = self.feedback[0]["critic"] * (1 - h1.square())
            e1 = (ent_signal @ self.feedback[0]["actor"]) * (1 - h1.square())
            a2 = (score @ self.feedback[1]["actor"]) * (1 - h2.square())
            c2 = self.feedback[1]["critic"] * (1 - h2.square())
            e2 = (ent_signal @ self.feedback[1]["actor"]) * (1 - h2.square())
        else:
            a1 = c1 = e1 = torch.zeros_like(h1)
            a2 = c2 = e2 = torch.zeros_like(h2)
        for prefix, pre, aa, cc, ee in [("body.0", x, a1, c1, e1),
                                       ("body.2", h1, a2, c2, e2)]:
            actor[prefix + ".weight"] = aa[:, :, None] * pre[:, None, :]
            actor[prefix + ".bias"] = aa
            critic[prefix + ".weight"] = cc[:, :, None] * pre[:, None, :]
            critic[prefix + ".bias"] = cc
            entropy[prefix + ".weight"] = ee[:, :, None] * pre[:, None, :]
            entropy[prefix + ".bias"] = ee
        activity = {}
        for prefix, pre, post in [("body.0", x, h1), ("body.2", h1, h2),
                                  ("pi", h2, prob), ("v", h2, val[:, None].tanh())]:
            activity[prefix + ".weight"] = (post.abs()[:, :, None] * pre.abs()[:, None, :]).mean(0).clamp(0, 1)
            activity[prefix + ".bias"] = post.abs().mean(0).clamp(0, 1)
        return actor, critic, entropy, activity, prob, val

    @torch.no_grad()
    def learn(self, x, actions, rewards, done, next_x):
        a, c, ent, activity, _, value = self.instantaneous(x, actions)
        nxt = self.forward(next_x)[3]
        delta = rewards + self.gamma * (~done).to(x.dtype) * nxt - value
        if not torch.isfinite(delta).all():
            raise FloatingPointError("non-finite TD signal")
        availability = {}
        before = {}
        for k, p in self.plastic_params.items():
            self.actor_trace[k].mul_(self.gamma * self.lam).add_(a[k])
            self.critic_trace[k].mul_(self.gamma * self.lam).add_(c[k])
            shape = (len(x),) + (1,) * p.ndim
            g = (delta.view(shape) * (self.actor_trace[k] + self.vf * self.critic_trace[k])).mean(0)
            g.add_(ent[k].mean(0), alpha=self.entropy)
            # Component-wise clipping does not require a network-wide norm.
            g.clamp_(-.1, .1)
            if self.protection in ("meta", "both"):
                self.meta[k].lerp_(self.meta_max * activity[k], 1 / self.meta_tau)
                availability[k] = torch.exp(-(self.meta[k] * p).abs())
                before[k] = p.clone()
            p.grad = g
        self.opt.step()
        for k, p in self.plastic_params.items():
            if k in availability:
                # Gate the actual synaptic displacement. Scaling the gradient
                # before Adam would be substantially cancelled by its moments.
                p.copy_(before[k] + availability[k] * (p - before[k]))
            if self.protection in ("slow", "both"):
                avail = availability[k] if self.protection == "both" else 1.
                p.add_(self.consolidation * avail * activity[k] * (self.slow[k] - p))
                self.slow[k].lerp_(p, 1 / self.slow_tau)
            self.actor_trace[k][done] = 0
            self.critic_trace[k][done] = 0
            if not torch.isfinite(p).all():
                raise FloatingPointError("non-finite learned parameter")
        self.count["frames"] += len(x)
        self.count["updates"] += 1
        self.count["parameter_updates"] += sum(p.numel() for p in self.plastic_params.values())
        return delta

    def memory_bytes(self):
        size = lambda ts: sum(t.numel() * t.element_size() for t in ts)
        opt = [v for st in self.opt.state.values() for v in st.values() if torch.is_tensor(v)]
        return dict(weights=size(self.params.values()), optimizer=size(opt),
                    eligibility=size(self.actor_trace.values()) + size(self.critic_trace.values()),
                    slow_reference=size(self.slow.values()), metaplasticity=size(self.meta.values()),
                    fixed_feedback=size(v for row in self.feedback.values() for v in row.values()),
                    plastic_parameters=sum(p.numel() for p in self.plastic_params.values()),
                    stored_trajectories=0,
                    scope="Actual allocated persistent tensors; temporary forward/update tensors and sensory scaffold reported separately.")


def check():
    """Meaningful independent credit/locality/causal-reset checks, CPU only."""
    torch.manual_seed(92)
    m = LocalContinualAC(10, 4, (7, 6), n_envs=3, rule="bp", seed=21)
    x = torch.randn(3, 10); act = torch.tensor([0, 2, 1])
    aa, cc, ee, *_ = m.instantaneous(x, act)
    ref = copy.deepcopy(m.net).requires_grad_(True)
    h = ref.body(x); logits = ref.pi(h); logp = logits.log_softmax(-1)
    policy = logp[torch.arange(3), act].sum()
    value = ref.v(h).sum()
    entropy = -(logp.exp() * logp).sum()
    errors = {}
    for name, objective, manual in [("actor", policy, aa), ("critic", value, cc), ("entropy", entropy, ee)]:
        ref.zero_grad(); objective.backward(retain_graph=True)
        err = 0.
        for k, p in ref.named_parameters():
            gradient = torch.zeros_like(p) if p.grad is None else p.grad
            err = max(err, float((gradient - manual[k].sum(0)).abs().max()))
        assert err < 2e-6, (name, err)
        errors[name] = err
    local = LocalContinualAC(10, 4, (7, 6), n_envs=3, rule="local", seed=21)
    local.feedback[0]["actor"].zero_(); local.feedback[0]["critic"].zero_()
    al, cl, el, *_ = local.instantaneous(x, act)
    assert all(torch.count_nonzero(z["body.0.weight"]) == 0 for z in (al, cl, el))
    # Change a forward output matrix while holding local feedback and hidden
    # activation fixed. Critic hidden eligibility cannot transport that weight.
    old = cl["body.2.weight"].clone()
    local.params["v.weight"].mul_(13)
    new = local.instantaneous(x, act)[1]["body.2.weight"]
    assert torch.equal(old, new)
    local.learn(x, act, torch.tensor([1., 0., 0.]), torch.tensor([True, False, True]), x)
    assert all(torch.count_nonzero(t[[0, 2]]) == 0 for t in local.actor_trace.values())
    assert all(torch.count_nonzero(t[[0, 2]]) == 0 for t in local.critic_trace.values())
    assert all(p.grad_fn is None and not p.requires_grad for p in local.params.values())
    protected = LocalContinualAC(10, 4, (7, 6), n_envs=3, rule="local",
                                 protection="meta", seed=21, meta_tau=1e12)
    protected.params["pi.bias"].fill_(1.)
    protected.meta["pi.bias"].fill_(1e8)
    old_bias = protected.params["pi.bias"].clone()
    protected.learn(x, act, torch.tensor([1., 0., 0.]),
                    torch.tensor([True, False, True]), x)
    assert torch.equal(old_bias, protected.params["pi.bias"])
    readout = LocalContinualAC(10, 4, (7, 6), n_envs=3, rule="readout", seed=21)
    body = {k: p.clone() for k, p in readout.params.items() if k.startswith("body.")}
    readout.learn(x, act, torch.tensor([1., 0., 0.]),
                  torch.tensor([True, False, True]), x)
    assert all(torch.equal(p, readout.params[k]) for k, p in body.items())
    assert not readout.feedback and all(k.startswith(("pi.", "v.")) for k in readout.actor_trace)
    print(json.dumps(dict(ok=True, manual_bp_autograd_errors=errors,
                         fixed_feedback_locality=True, terminal_trace_reset=True,
                         no_learning_autograd_graph=True,
                         metaplasticity_gates_actual_adam_displacement=True,
                         readout_hidden_frozen_and_no_hidden_feedback=True,
                         memory_bytes=local.memory_bytes()), indent=2))


if __name__ == "__main__":
    check()
