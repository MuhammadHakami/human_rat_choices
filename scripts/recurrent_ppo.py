"""Recurrent (LSTM) PPO, the strong backprop control for the e-prop / rate-BPTT prototypes. PROTOTYPE, synthetic only.

Standard recurrent PPO in the CleanRL ppo_atari_lstm pattern: vectorised rollouts, the LSTM state stored at the start
of each rollout and reset on episode ends, GAE(lambda), clipped surrogate, value loss, entropy bonus, Adam, sequence
minibatches split by environment so the LSTM is replayed through whole rollout segments. Hidden size, observation,
actions and discount are set by the caller to match the other learners.
Memory accounting (bytes, float32 unless stated) is reported for every learner in the same categories:
weights, optimiser state, per-agent persistent state (hidden state / eligibility / accumulators), experience buffer.
  python scripts/recurrent_ppo.py check
"""
import json, sys
import numpy as np
import torch
import torch.nn as nn

F32 = 4


class DelayedCue:
    """Vectorised synthetic task shared with the other prototypes' sanity checks (cue 5, delay 10, go, choose)."""
    def __init__(self, n, cue=5, delay=10, seed=0):
        self.n, self.cue, self.delay, self.rng = n, cue, delay, np.random.default_rng(seed)
        self.T = cue + delay + 1; self.obs_dim, self.n_act = 4, 2

    def _obs(self):
        x = np.zeros((self.n, 4), np.float32); x[:, 3] = 0.5 * (self.rng.random(self.n) < 0.2)
        cue = self.t < self.cue; x[cue, self.side[cue]] = 1.0; x[self.t == self.T - 1, 2] = 1.0
        return x

    def reset(self):
        self.side = self.rng.integers(2, size=self.n); self.t = np.zeros(self.n, int); return self._obs()

    def step(self, a):
        last = self.t == self.T - 1
        r = np.where(last, (a == self.side).astype(np.float32), 0.0).astype(np.float32)
        self.t = self.t + 1
        if last.any():
            self.side[last] = self.rng.integers(2, size=int(last.sum())); self.t[last] = 0
        return self._obs(), r, last


class Agent(nn.Module):
    def __init__(self, obs_dim, n_act, hidden):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden); self.pi = nn.Linear(hidden, n_act); self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.orthogonal_(self.v.weight, 1.0)

    def unroll(self, x, state, done):
        """x (T, B, obs), done (T, B) = episode ended BEFORE this step -> state reset."""
        h, c = state; out = []
        for t in range(x.shape[0]):
            keep = (1.0 - done[t]).view(1, -1, 1); h, c = h * keep, c * keep
            o, (h, c) = self.lstm(x[t:t + 1], (h, c)); out.append(o)
        return torch.cat(out), (h, c)

    def heads(self, o):
        return torch.distributions.Categorical(logits=self.pi(o)), self.v(o).squeeze(-1)


def gae(rew, val, done_after, last_val, gamma, lam):
    """GAE with done_after[t] = the episode ended at step t (no bootstrap through it)."""
    T = len(rew); adv = torch.zeros_like(rew); g = 0.0
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]; nd = 1.0 - done_after[t]
        delta = rew[t] + gamma * nv * nd - val[t]; g = delta + gamma * lam * nd * g; adv[t] = g
    return adv, adv + val


class RecurrentPPO:
    def __init__(self, obs_dim, n_act, hidden=40, n_envs=8, n_steps=128, gamma=0.95, lam=0.95, lr=1e-3, clip=0.2,
                 ent=0.01, vf=0.5, epochs=4, minibatches=2, seed=0):
        torch.manual_seed(seed); self.net = Agent(obs_dim, n_act, hidden)    # init only; sampling/shuffles use the
        self.rng = np.random.default_rng(seed); self.tg = torch.Generator().manual_seed(seed)   # per-instance RNGs below
        self.last_buffer = None
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.cfg = dict(n_envs=n_envs, n_steps=n_steps, gamma=gamma, lam=lam, clip=clip, ent=ent, vf=vf, epochs=epochs, mb=minibatches)
        self.hidden, self.obs_dim = hidden, obs_dim
        self.count = dict(frames=0, updates=0, weight_updates=0)

    def memory_bytes(self):
        """ACTUAL tensor storage: parameters, Adam moments (allocated after the first step), the LSTM state, and the
        last rollout buffer as stored (int64 actions included) plus its saved initial LSTM state."""
        b = lambda ts: int(sum(t.element_size() * t.numel() for t in ts))
        opt = [v for st in self.opt.state.values() for v in st.values() if torch.is_tensor(v) and v.dim() > 0]
        buf = None if self.last_buffer is None else b(self.last_buffer[0].values()) + b(self.last_buffer[1])
        return dict(weights=b(self.net.parameters()), optimiser=b(opt),
                    persistent_state=self.cfg["n_envs"] * 2 * self.hidden * next(self.net.parameters()).element_size(),
                    experience_buffer=buf)

    def rollout(self, env, obs, state, done_prev):
        c = self.cfg; buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done_before", "done_after", "probs")}
        state0 = (state[0].clone(), state[1].clone())
        with torch.no_grad():
            for _ in range(c["n_steps"]):
                x = torch.as_tensor(obs)[None]; d = torch.as_tensor(done_prev, dtype=torch.float32)[None]
                o, state = self.net.unroll(x, state, d); dist, v = self.net.heads(o[0])
                a = torch.multinomial(dist.probs, 1, generator=self.tg).squeeze(-1); obs2, r, done = env.step(a.numpy())
                for k, val in (("obs", x[0]), ("act", a), ("logp", dist.log_prob(a)), ("val", v), ("rew", torch.as_tensor(r)),
                               ("done_before", d[0]), ("done_after", torch.as_tensor(done, dtype=torch.float32)),
                               ("probs", dist.probs)):
                    buf[k].append(val)
                obs, done_prev = obs2, done.astype(np.float32); self.count["frames"] += c["n_envs"]
            xl = torch.as_tensor(obs)[None]; dl = torch.as_tensor(done_prev, dtype=torch.float32)[None]
            o, _ = self.net.unroll(xl, (state[0].clone(), state[1].clone()), dl); last_v = self.net.heads(o[0])[1]
        B = {k: torch.stack(v) for k, v in buf.items()}
        B["adv"], B["ret"] = gae(B["rew"], B["val"], B["done_after"], last_v, c["gamma"], c["lam"])
        self.last_buffer = (B, state0)
        return B, state0, obs, state, done_prev

    def update(self, B, state0):
        c = self.cfg; envs = np.arange(c["n_envs"]); P = sum(p.numel() for p in self.net.parameters())
        for _ in range(c["epochs"]):
            self.rng.shuffle(envs)
            for idx in np.array_split(envs, c["mb"]):
                i = torch.as_tensor(idx)
                o, _ = self.net.unroll(B["obs"][:, i], (state0[0][:, i], state0[1][:, i]), B["done_before"][:, i])
                dist, v = self.net.heads(o); logp = dist.log_prob(B["act"][:, i]); ratio = (logp - B["logp"][:, i]).exp()
                adv = B["adv"][:, i]; sd = adv.std() if adv.numel() > 1 else torch.zeros(())   # one-step segment: no NaN std
                if adv.numel() == 1: self.count["singleton_advantage_minibatches"] = self.count.get("singleton_advantage_minibatches", 0) + 1
                adv = (adv - adv.mean()) / (sd + 1e-8)
                pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - c["clip"], 1 + c["clip"])).mean()
                loss = pg + c["vf"] * 0.5 * ((v - B["ret"][:, i]) ** 2).mean() - c["ent"] * dist.entropy().mean()
                self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 0.5); self.opt.step()
                self.count["weight_updates"] += P
        self.count["updates"] += 1


def check():
    torch.manual_seed(0)
    # 1. GAE against a direct sum of discounted TD errors on random data with episode ends
    T = 50; r = torch.randn(T); v = torch.randn(T); d = (torch.rand(T) < 0.1).float(); lv = torch.randn(())
    adv, _ = gae(r, v, d, lv, 0.9, 0.8); ref = torch.zeros(T)
    for t in range(T):
        s, w = 0.0, 1.0
        for u in range(t, T):
            nv = lv if u == T - 1 else v[u + 1]; s += w * (r[u] + 0.9 * nv * (1 - d[u]) - v[u])
            if d[u]: break
            w *= 0.9 * 0.8
        ref[t] = s
    err_gae = float((adv - ref).abs().max()); assert err_gae < 1e-5, err_gae
    # 2. replay consistency: re-unrolling the stored segment from the stored initial state reproduces rollout logp/value
    env = DelayedCue(8, seed=0); ppo = RecurrentPPO(4, 2, n_steps=48, seed=0)
    obs = env.reset(); st = (torch.zeros(1, 8, 40), torch.zeros(1, 8, 40)); dp = np.zeros(8, np.float32)
    B, st0, *_ = ppo.rollout(env, obs, st, dp)
    with torch.no_grad():
        o, _ = ppo.net.unroll(B["obs"], st0, B["done_before"]); dist, vv = ppo.net.heads(o)
    err_replay = float(max((dist.log_prob(B["act"]) - B["logp"]).abs().max(), (vv - B["val"]).abs().max()))
    assert err_replay < 1e-5, err_replay
    # 3. per-instance determinism: same seed -> identical weights after two updates; global RNG use does not matter
    def two_updates(seed, disturb):
        env = DelayedCue(8, seed=seed); ppo = RecurrentPPO(4, 2, n_steps=32, seed=seed); obs = env.reset()
        st = (torch.zeros(1, 8, 40), torch.zeros(1, 8, 40)); dp = np.zeros(8, np.float32)
        for _ in range(2):
            if disturb: np.random.rand(7); torch.rand(5)                   # touch the global RNGs between updates
            B, st0, obs, st, dp = ppo.rollout(env, obs, st, dp); ppo.update(B, st0)
        return torch.cat([p.detach().ravel() for p in ppo.net.parameters()])
    assert torch.equal(two_updates(3, False), two_updates(3, True)), "PPO depends on global RNG state"
    # 4. cross-import dtype: the e-prop / rate modules no longer change torch's default dtype; float32 end to end
    import eprop_actor_critic, rate_rnn_actor_critic                         # noqa: F401  (import side effects only)
    assert torch.get_default_dtype() == torch.float32
    assert all(p.dtype == torch.float32 for p in RecurrentPPO(4, 2).net.parameters())
    # 5. synthetic learning sanity at an EXACT declared budget: 24,000 frames = 24 rollouts x 125 steps x 8 envs
    budget, n_steps, n_envs = 24000, 125, 8
    assert budget % (n_steps * n_envs) == 0
    env = DelayedCue(n_envs, seed=0); ppo = RecurrentPPO(4, 2, n_envs=n_envs, n_steps=n_steps, seed=0); obs = env.reset()
    st = (torch.zeros(1, n_envs, 40), torch.zeros(1, n_envs, 40)); dp = np.zeros(n_envs, np.float32); hist = []
    while ppo.count["frames"] < budget:
        B, st0, obs, st, dp = ppo.rollout(env, obs, st, dp); ppo.update(B, st0)
        ends = B["done_after"].bool(); hist += B["rew"][ends].tolist()
    assert ppo.count["frames"] == budget, ppo.count["frames"]
    res = dict(first_success=float(np.mean(hist[:300])), final_success=float(np.mean(hist[-300:])), episodes=len(hist),
               counters=ppo.count, memory_bytes=ppo.memory_bytes())
    print(json.dumps({"ok": True, "gae_max_abs_err": err_gae, "replay_max_abs_err": err_replay, "delayed_cue": res}, indent=1))


if __name__ == "__main__":
    {"check": check}[sys.argv[1]]()
