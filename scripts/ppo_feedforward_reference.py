"""Feedforward (shared 64x64 tanh) PPO reference. PROTOCOL: docs/WACV_EMBODIED_LONG_CONTROL_PROTOCOL_20261012.json.

Reuses the reviewed RecurrentPPO collector, GAE, clipped objective, value loss, entropy bonus, Adam, gradient clipping
and sequence minibatching UNCHANGED (scripts/recurrent_ppo.py is imported, not edited). Only the network differs:
FFAgent exposes the same unroll(x, state, done) / heads(o) API; it ignores the recurrent state, which is a zero-size
(1, n, 0) tensor ("dummy API state", 0 bytes, labelled used_recurrent_state=0). The LSTM built by the parent
constructor is discarded and the Adam optimiser is rebuilt on the FF parameters only.
Orthogonal init: sqrt(2) on shared layers, 0.01 actor, 1.0 critic, zero biases.
  python scripts/ppo_feedforward_reference.py check
"""
import json, math, sys
import numpy as np
import torch
import torch.nn as nn

from recurrent_ppo import RecurrentPPO, gae


class FFAgent(nn.Module):
    def __init__(self, obs_dim, n_act, layers=(64, 64)):
        super().__init__()
        mods, d = [], obs_dim
        for h in layers:
            lin = nn.Linear(d, h); nn.init.orthogonal_(lin.weight, math.sqrt(2)); nn.init.zeros_(lin.bias)
            mods += [lin, nn.Tanh()]; d = h
        self.body = nn.Sequential(*mods); self.pi = nn.Linear(d, n_act); self.v = nn.Linear(d, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0); nn.init.zeros_(self.v.bias)

    def unroll(self, x, state, done):
        """Same signature as the LSTM agent; no state is used, so episode boundaries need no reset."""
        return self.body(x), state

    def heads(self, o):
        return torch.distributions.Categorical(logits=self.pi(o)), self.v(o).squeeze(-1)


class FeedforwardPPO(RecurrentPPO):
    def __init__(self, obs_dim, n_act, layers=(64, 64), lr=1e-3, seed=0, **kw):
        super().__init__(obs_dim, n_act, hidden=1, lr=lr, seed=seed, **kw)     # reviewed RNG/cfg/counters setup
        torch.manual_seed(seed); self.net = FFAgent(obs_dim, n_act, layers)        # replaces the parent's LSTM
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)      # optimiser over FF params only
        self.hidden = 0; self.layers = tuple(layers)                              # dummy (1, n, 0) API state

    def forward_macs(self):
        dims = [self.obs_dim, *self.layers]
        return sum(a * b for a, b in zip(dims, dims[1:])) + dims[-1] * (self.net.pi.out_features + 1)


def check():
    torch.manual_seed(0)
    obs_dim, n_act = 252, 4
    ppo = FeedforwardPPO(obs_dim, n_act, seed=0)
    P = sum(p.numel() for p in ppo.net.parameters())
    assert P == 20677, P                                                        # protocol expected_parameters
    # 1. optimiser holds exactly the FF parameters (no stale LSTM tensors)
    opt_ids = {id(p) for g in ppo.opt.param_groups for p in g["params"]}
    assert opt_ids == {id(p) for p in ppo.net.parameters()} and ppo.opt.defaults["eps"] == 1e-5
    # 2. orthogonal gains: rows orthonormal x gain (all weight matrices here have rows <= cols)
    gains = [(ppo.net.body[0].weight, math.sqrt(2)), (ppo.net.body[2].weight, math.sqrt(2)),
             (ppo.net.pi.weight, .01), (ppo.net.v.weight, 1.0)]
    err_orth = max(float((W.detach() @ W.detach().T - g * g * torch.eye(W.shape[0])).abs().max()) for W, g in gains)
    assert err_orth < 1e-5, err_orth
    assert all(float(b.detach().abs().max()) == 0 for b in (ppo.net.body[0].bias, ppo.net.body[2].bias, ppo.net.pi.bias, ppo.net.v.bias))
    # 3. action distribution = softmax(actor logits); dummy state passes through untouched; 0 state bytes
    x = torch.randn(3, 5, obs_dim); st = (torch.zeros(1, 5, 0), torch.zeros(1, 5, 0))
    o, st2 = ppo.net.unroll(x, st, torch.zeros(3, 5)); dist, v = ppo.net.heads(o)
    assert st2 is st and torch.allclose(dist.probs, torch.softmax(ppo.net.pi(o), -1)) and v.shape == (3, 5)
    assert ppo.memory_bytes()["persistent_state"] == 0
    # 4. critic/actor gradient vs central finite differences of the PPO loss, float64 copy
    net = FFAgent(obs_dim, n_act).double(); xb = torch.randn(7, obs_dim, dtype=torch.float64)
    a = torch.randint(4, (7,)); adv = torch.randn(7, dtype=torch.float64); ret = torch.randn(7, dtype=torch.float64)
    with torch.no_grad(): old = net.heads(net.body(xb))[0].log_prob(a) + .05 * torch.randn(7, dtype=torch.float64)
    def loss_fn():
        d, vv = net.heads(net.body(xb)); r = (d.log_prob(a) - old).exp()
        pg = torch.max(-adv * r, -adv * r.clamp(.8, 1.2)).mean()
        return pg + .5 * .5 * ((vv - ret) ** 2).mean() - .01 * d.entropy().mean()
    net.zero_grad(); loss_fn().backward(); err_fd = 0.0
    for p in (net.v.weight, net.pi.weight, net.body[2].weight):
        for idx in [(0, 0), (0, 3), (p.shape[0] - 1, p.shape[1] - 1)]:
            with torch.no_grad():
                p[idx] += 1e-6; lp = float(loss_fn()); p[idx] -= 2e-6; lm = float(loss_fn()); p[idx] += 1e-6
            err_fd = max(err_fd, abs((lp - lm) / 2e-6 - float(p.grad[idx])))
    assert err_fd < 1e-6, err_fd
    # 5. gradient clipping: one reviewed update with a huge reward scale moves weights by <= lr-bounded Adam step,
    #    and clip_grad_norm_ caps the pre-step norm at 0.5 (checked by recomputing the clipped norm)
    norms = []; orig = nn.utils.clip_grad_norm_
    def spy(params, max_norm):
        params = list(params); total = orig(params, max_norm)
        norms.append(float(torch.norm(torch.stack([q.grad.norm() for q in params])))); return total
    nn.utils.clip_grad_norm_ = spy
    try:
        env = Bandit(8, seed=1, scale=1e4); obs = env.reset(); st = (torch.zeros(1, 8, 0), torch.zeros(1, 8, 0))
        small = FeedforwardPPO(10, 4, n_steps=16, seed=1)
        B, st0, *_ = small.rollout(env, obs, st, np.zeros(8, np.float32))
        with torch.no_grad():                                     # 6. replay consistency (before the update)
            o, _ = small.net.unroll(B["obs"], st0, B["done_before"]); d, vv = small.net.heads(o)
        err_replay = float(max((d.log_prob(B["act"]) - B["logp"]).abs().max(), (vv - B["val"]).abs().max()))
        small.update(B, st0)
    finally:
        nn.utils.clip_grad_norm_ = orig
    assert norms and max(norms) <= 0.5 + 1e-5, max(norms)
    assert err_replay < 1e-6, err_replay
    # 7. per-instance determinism, global RNGs disturbed
    def two(seed, disturb):
        env = Bandit(8, seed=seed); p = FeedforwardPPO(10, 4, n_steps=16, seed=seed); ob = env.reset()
        s = (torch.zeros(1, 8, 0), torch.zeros(1, 8, 0)); dp = np.zeros(8, np.float32)
        for _ in range(2):
            if disturb: np.random.rand(3); torch.rand(4)
            B, s0, ob, s, dp = p.rollout(env, ob, s, dp); p.update(B, s0)
        return torch.cat([q.detach().ravel() for q in p.net.parameters()])
    assert torch.equal(two(5, False), two(5, True))
    # 8. learning sanity at an exact budget on a memoryless contextual bandit (FF-solvable)
    env = Bandit(8, seed=0); p = FeedforwardPPO(10, 4, n_steps=32, seed=0); ob = env.reset()
    s = (torch.zeros(1, 8, 0), torch.zeros(1, 8, 0)); dp = np.zeros(8, np.float32); hist = []
    while p.count["frames"] < 16384:
        B, s0, ob, s, dp = p.rollout(env, ob, s, dp); p.update(B, s0); hist += B["rew"].ravel().tolist()
    assert p.count["frames"] == 16384
    final = float(np.mean(hist[-1024:])); assert final > .9, final
    print(json.dumps(dict(ok=True, parameters=P, orth_err=err_orth, fd_err=err_fd, max_clipped_norm=max(norms),
                          replay_err=err_replay, bandit_first=float(np.mean(hist[:1024])), bandit_final=final,
                          forward_macs=ppo.forward_macs(), memory_bytes=ppo.memory_bytes()), indent=1))


class Bandit:
    """Memoryless check task: one-hot cue among 4 (+6 noise dims); reward `scale` if action == cue; one-step episodes."""
    def __init__(self, n, seed=0, scale=1.0):
        self.n, self.rng, self.scale = n, np.random.default_rng(seed), scale

    def _obs(self):
        x = self.rng.normal(0, .1, (self.n, 10)).astype(np.float32); x[np.arange(self.n), self.cue] += 1; return x

    def reset(self):
        self.cue = self.rng.integers(4, size=self.n); return self._obs()

    def step(self, a):
        r = (a == self.cue).astype(np.float32) * self.scale; self.cue = self.rng.integers(4, size=self.n)
        return self._obs(), r, np.ones(self.n, bool)


if __name__ == "__main__":
    {"check": check}[sys.argv[1]]()
