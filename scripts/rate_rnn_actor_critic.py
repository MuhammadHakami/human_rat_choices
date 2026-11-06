"""Rate-RNN actor-critic trained with full backprop-through-time: the non-local comparator for
scripts/eprop_actor_critic.py, PROTOTYPE (synthetic checks only).

Same interface as EpropAC (step(x, reward_prev) -> action; end(reward_last)), same observation, actions, hidden size
and readout form (softmax policy and linear value on an exponentially filtered hidden rate), same discount and
entropy bonus. Differences, which are the point of the comparison: tanh rate units instead of LIF/ALIF spikes, and
an exact gradient by BPTT through the whole episode, applied once per episode (offline within the episode).
Loss per episode: sum_t [ -(G_t - V_t) log pi(a_t) + c_V (G_t - V_t)^2 / 2 - ent H(pi_t) ], G_t the discounted return.
Counters: frames, multiply-accumulates (forward + backward ~ 3x forward), weight-update operations.
dtype is per instance (float32 for runs, float64 for audits); importing this module changes no global torch state.
  python scripts/rate_rnn_actor_critic.py check
"""
import json, sys
import numpy as np
import torch


class RateBPTTAC:
    def __init__(self, n_in, n_rec, n_act, tau=20.0, kappa=None, gamma=0.95, c_v=0.5, lr=1e-3, entropy=0.0,
                 dtype=torch.float32, seed=0):
        g = torch.Generator().manual_seed(seed); self.k, self.n, self.dt = n_act, n_rec, dtype; dt = dtype
        self.alpha = np.exp(-1 / tau); self.kappa = self.alpha if kappa is None else kappa
        self.gamma, self.c_v, self.ent = gamma, c_v, entropy
        rn = lambda *sh: torch.randn(*sh, generator=g, dtype=dt); zr = lambda *sh: torch.zeros(*sh, dtype=dt)
        self.P = dict(W_in=rn(n_rec, n_in) / np.sqrt(n_in), W_rec=rn(n_rec, n_rec) / np.sqrt(n_rec), b=zr(n_rec),
                      W_out=rn(n_act, n_rec) / np.sqrt(n_rec), b_out=zr(n_act), w_V=zr(n_rec), b_V=zr(()))
        for p in self.P.values(): p.requires_grad_(True)
        self.opt = torch.optim.SGD(self.P.values(), lr=lr); self.rng = np.random.default_rng(seed)
        self.n_params = sum(p.numel() for p in self.P.values())
        self.count = dict(frames=0, macs=0, weight_updates=0, episodes=0)
        self.reset()

    def memory_bytes(self, T):
        """ACTUAL tensor storage for weights/state, same categories as recurrent_ppo. The experience buffer is the
        per-episode activation store BPTT needs (h, filtered h, pi, V per step) at this dtype; autograd also keeps
        intermediate activations (pre-tanh, softmax inputs), so it is reported as a LOWER BOUND."""
        es = torch.empty((), dtype=self.dt).element_size()
        return dict(weights=int(sum(p.element_size() * p.numel() for p in self.P.values())), optimiser=0,
                    persistent_state=int(self.h.element_size() * self.h.numel() + self.hf.element_size() * self.hf.numel()),
                    experience_buffer_lower_bound=T * (2 * self.n + self.k + 1) * es, dtype=str(self.dt))

    def reset(self):
        self.h = torch.zeros(self.n, dtype=self.dt); self.hf = torch.zeros(self.n, dtype=self.dt); self.buf = []

    def step(self, x, reward_prev=0.0, action=None):
        P = self.P; x = torch.as_tensor(x, dtype=self.dt)
        self.h = self.alpha * self.h + (1 - self.alpha) * torch.tanh(P["W_in"] @ x + P["W_rec"] @ self.h + P["b"])
        self.hf = self.kappa * self.hf + (1 - self.kappa) * self.h
        pi = torch.softmax(P["W_out"] @ self.hf + P["b_out"], 0); V = P["w_V"] @ self.hf + P["b_V"]
        a = int(self.rng.choice(self.k, p=pi.detach().numpy())) if action is None else int(action)
        if self.buf: self.buf[-1][3] = reward_prev
        self.buf.append([pi, V, a, 0.0])
        self.count["frames"] += 1; self.count["macs"] += P["W_in"].numel() + P["W_rec"].numel() + P["W_out"].numel() + self.n
        return a

    def loss(self, reward_last):
        self.buf[-1][3] = reward_last; G = 0.0; out = 0.0
        for pi, V, a, r in reversed(self.buf):
            G = r + self.gamma * G; adv = G - V
            H = -(pi * torch.log(pi + 1e-12)).sum()
            out = out - adv.detach() * torch.log(pi[a]) + self.c_v * 0.5 * adv ** 2 - self.ent * H
        return out

    def end(self, reward_last):
        L = self.loss(reward_last); self.opt.zero_grad(); L.backward(); self.opt.step()
        self.count["macs"] += 2 * len(self.buf) * (self.P["W_in"].numel() + self.P["W_rec"].numel() + self.P["W_out"].numel())
        self.count["weight_updates"] += self.n_params; self.count["episodes"] += 1
        self.reset()


def check():
    assert torch.get_default_dtype() == torch.float32, "importing this module must not change torch's default dtype"
    m32, m64 = RateBPTTAC(4, 40, 2).memory_bytes(16), RateBPTTAC(4, 40, 2, dtype=torch.float64).memory_bytes(16)
    assert m64["weights"] == 2 * m32["weights"] and m64["experience_buffer_lower_bound"] == 2 * m32["experience_buffer_lower_bound"]
    # 1. BPTT gradient == central finite differences of the same objective with the policy-term advantages held at
    #    their base values (the loss detaches them), on a fixed episode (actions and rewards fixed)
    rng = np.random.default_rng(0); T = 12
    net = RateBPTTAC(5, 8, 3, gamma=0.9, entropy=0.01, dtype=torch.float64, seed=1); X = rng.random((T, 5)); A = rng.integers(3, size=T); R = rng.normal(size=T)

    def episode():
        net.reset()
        for t in range(T): net.step(X[t], reward_prev=R[t - 1] if t else 0.0, action=A[t])
        net.buf[-1][3] = R[-1]

    episode(); grads = dict(zip(net.P, torch.autograd.grad(net.loss(R[-1]), list(net.P.values()))))
    G0, g = [], 0.0
    for b in reversed(net.buf): g = b[3] + net.gamma * g; G0.append(g)
    G0 = G0[::-1]; adv0 = [G0[i] - float(b[1]) for i, b in enumerate(net.buf)]

    def frozen():
        episode()
        return float(sum(-adv0[i] * torch.log(b[0][b[2]]) + net.c_v * 0.5 * (G0[i] - b[1]) ** 2
                         + net.ent * (b[0] * torch.log(b[0] + 1e-12)).sum() for i, b in enumerate(net.buf)))
    fd = {}
    for nm, p in net.P.items():
        idx = tuple(int(i) for i in np.unravel_index(rng.integers(p.numel()), p.shape)) if p.dim() else ()
        with torch.no_grad(): old = p[idx].item(); p[idx] = old + 1e-6
        up = frozen()
        with torch.no_grad(): p[idx] = old - 1e-6
        dn = frozen()
        with torch.no_grad(): p[idx] = old
        fd[nm] = abs((up - dn) / 2e-6 - float(grads[nm][idx]))
    assert max(fd.values()) < 1e-6, fd
    # 2. synthetic delayed cue, same task and sizes as eprop_actor_critic.delayed_cue
    res = delayed_cue()
    print(json.dumps({"ok": True, "bptt_vs_finite_difference_abs_err": fd, "delayed_cue": res, "memory_bytes_T16_float32": m32}, indent=1))


def delayed_cue(episodes=1500, cue=5, delay=10, seed=0, **kw):
    net = RateBPTTAC(4, 40, 2, lr=2e-2, gamma=0.95, entropy=0.01, seed=seed, **kw); rng = np.random.default_rng(seed); hits = []
    for _ in range(episodes):
        side = int(rng.integers(2))
        for t in range(cue + delay + 1):
            x = np.zeros(4, np.float32); x[3] = 0.5 * (rng.random() < 0.2)
            if t < cue: x[side] = 1.0
            if t == cue + delay: x[2] = 1.0
            a = net.step(x)
        r = float(a == side); hits.append(r); net.end(r)
    return dict(final_success=float(np.mean(hits[-300:])), first_success=float(np.mean(hits[:300])), counters=dict(net.count))


if __name__ == "__main__":
    {"check": check}[sys.argv[1]]()
