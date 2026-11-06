"""Spiking LIF/ALIF actor-critic trained by full surrogate-gradient BPTT: the comparator that separates the LEARNING
RULE (e-prop vs BPTT) from the ARCHITECTURE (spiking vs rate). PROTOTYPE, synthetic checks only.

Same network as scripts/eprop_actor_critic.EpropAC, constructed FROM an EpropAC instance with the same arguments and
seed, so neuron equations, ALIF fraction, kappa readout filter (normalised), connectivity mask (no self-synapses),
initial weights and readouts are identical. Differences, which are the comparison:
  * gradients flow through the recurrent spike inputs z^{t-1} of other neurons (full BPTT through the episode), where
    e-prop truncates them; the pseudo-derivative psi is the same;
  * the reset term v_th z^{t-1} carries NO gradient (stop-gradient, as in EpropAC); reset_grad=True removes that
    stop-gradient and is used only for the smooth finite-difference check;
  * learning is per episode: loss = sum_t [-(G_t - V_t) log pi(a_t) + c_V (G_t - V_t)^2 / 2 - c_H H(pi_t)],
    G_t the discounted Monte-Carlo return, advantage detached in the policy term, independent entropy term;
    optimizer "sgd" or "adam". This is an episode-BPTT schedule, not e-prop's online SGD schedule: identical neuron
    dynamics do not imply an identical optimisation schedule.
Per-instance dtype and seed; no global torch state is changed. Same step/end interface and counters as EpropAC.
  python scripts/spiking_bptt_actor_critic.py check
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eprop_actor_critic import EpropAC, _Spike


class SpikingBPTTAC:
    def __init__(self, n_in, n_rec, n_act, frac_alif=0.5, tau_m=20.0, tau_a=200.0, beta=1.0, v_th=0.6, gamma_pd=0.3,
                 kappa=None, gamma=0.95, c_v=0.5, lr=1e-3, entropy=0.0, optimizer="sgd", dtype=torch.float32, seed=0,
                 spike_fn="heaviside", reset_grad=False):
        ref = EpropAC(n_in, n_rec, n_act, frac_alif=frac_alif, tau_m=tau_m, tau_a=tau_a, beta=beta, v_th=v_th,
                      gamma_pd=gamma_pd, kappa=kappa, gamma=gamma, c_v=c_v, dtype=dtype, seed=seed)
        self.n_in, self.n, self.k, self.dt = n_in, n_rec, n_act, dtype
        self.alpha, self.rho, self.v_th, self.gpd, self.kappa = ref.alpha, ref.rho, ref.v_th, ref.gpd, ref.kappa
        self.beta, self.mask = ref.beta.clone(), ref.mask.clone()
        self.gamma, self.c_v, self.ent = gamma, c_v, entropy
        self.P = {nm: getattr(ref, nm).clone().requires_grad_(True) for nm in ("W", "W_out", "b_out", "w_V", "b_V")}
        opt = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[optimizer]
        self.opt = opt(self.P.values(), lr=lr); self.optimizer = optimizer
        self.spike_fn, self.reset_grad = spike_fn, reset_grad
        self.rng = np.random.default_rng(seed)
        self.count = dict(frames=0, spikes=0, synaptic_events=0, weight_updates=0, episodes=0)
        self.reset()

    def reset(self):
        z = lambda: torch.zeros(self.n, dtype=self.dt)
        self.v, self.a, self.z, self.zf = z(), z(), z(), z(); self.buf = []

    def _spike(self, u):
        if self.spike_fn == "smooth":                       # differentiable forward, only for finite-difference checks
            return torch.sigmoid(u / 0.1)
        return _Spike.apply(u, self.v_th, self.gpd)

    def step(self, x, reward_prev=0.0, action=None):
        P = self.P; z_prev = self.z
        p = torch.cat([torch.as_tensor(x, dtype=self.dt), z_prev])            # recurrent spikes NOT detached
        self.a = self.rho * self.a + z_prev
        self.v = self.alpha * self.v + (P["W"] * self.mask) @ p - self.v_th * (z_prev if self.reset_grad else z_prev.detach())
        u = self.v - self.v_th - self.beta * self.a
        self.z = self._spike(u); self.zf = self.kappa * self.zf + (1 - self.kappa) * self.z
        pi = torch.softmax(P["W_out"] @ self.zf + P["b_out"], 0); V = P["w_V"] @ self.zf + P["b_V"]
        q = pi.detach().double().numpy(); a = int(self.rng.choice(self.k, p=q / q.sum())) if action is None else int(action)
        if self.buf: self.buf[-1][3] = reward_prev
        self.buf.append([pi, V, a, 0.0])
        s = int((self.z.detach() > 0.5).sum()); self.count["frames"] += 1; self.count["spikes"] += s
        self.count["synaptic_events"] += s * ((self.n - 1) + self.k + 1)
        return a

    def loss(self, reward_last):
        self.buf[-1][3] = reward_last; G = 0.0; out = 0.0
        for pi, V, a, r in reversed(self.buf):
            G = r + self.gamma * G; adv = G - V
            out = out - adv.detach() * torch.log(pi[a]) + self.c_v * 0.5 * adv ** 2 + self.ent * (pi * torch.log(pi + 1e-12)).sum()
        return out

    def end(self, reward_last):
        L = self.loss(reward_last); self.opt.zero_grad(); L.backward(); self.opt.step()
        with torch.no_grad(): self.P["W"] *= self.mask                     # keep the no-self-synapse structure exact
        self.count["weight_updates"] += int(self.mask.sum()) + sum(self.P[k].numel() for k in ("W_out", "b_out", "w_V", "b_V"))
        self.count["episodes"] += 1; self.reset()

    def memory_bytes(self, T):
        """ACTUAL storage for weights, optimiser moments and state; the BPTT activation store (v, a, z, zf per step plus
        pi, V) is a LOWER BOUND on autograd's saved tensors."""
        b = lambda ts: int(sum(t.element_size() * t.numel() for t in ts))
        opt = [v for st in self.opt.state.values() for v in st.values() if torch.is_tensor(v) and v.dim() > 0]
        es = torch.empty((), dtype=self.dt).element_size()
        return dict(weights=b(self.P.values()), optimiser=b(opt), fixed_structure=b([self.mask, self.beta]),
                    persistent_state=b([self.v, self.a, self.z, self.zf]),
                    experience_buffer_lower_bound=T * (4 * self.n + self.k + 1) * es, dtype=str(self.dt))


def check():
    F64 = torch.float64
    assert torch.get_default_dtype() == torch.float32
    T, n_in, n_rec, k = 30, 5, 12, 3; rng = np.random.default_rng(0)
    X = rng.random((T, n_in)) * 2.0; A = rng.integers(k, size=T); R = rng.normal(size=T)

    # 1. frozen-forward equivalence with EpropAC (same args/seed): identical spikes, policy and value every step
    ep = EpropAC(n_in, n_rec, k, kappa=0.8, dtype=F64, seed=4); ep.W *= 1.5
    sb = SpikingBPTTAC(n_in, n_rec, k, kappa=0.8, dtype=F64, seed=4)
    with torch.no_grad(): sb.P["W"] *= 1.5
    ep.reset(); sb.reset(); dz = dp = dv = 0.0
    for t in range(T):
        ep._neurons(X[t]); pi_e, V_e = ep._readout(); sb.step(X[t], action=A[t]); pi_b, V_b = sb.buf[-1][0], sb.buf[-1][1]
        dz = max(dz, float((ep.z - sb.z.detach()).abs().max())); dp = max(dp, float((pi_e - pi_b.detach()).abs().max()))
        dv = max(dv, abs(float(V_e) - float(V_b.detach())))
    spikes = sb.count["spikes"]; assert spikes > 0 and dz == 0.0 and dp < 1e-12 and dv < 1e-12, (dz, dp, dv, spikes)

    # 2. graph check: with a smooth forward and reset_grad=True, autograd == central finite differences (all params)
    sm = SpikingBPTTAC(n_in, n_rec, k, kappa=0.8, entropy=0.05, dtype=F64, seed=4, spike_fn="smooth", reset_grad=True)
    def L_of():
        sm.reset()
        for t in range(T): sm.step(X[t], reward_prev=R[t - 1] if t else 0.0, action=A[t])
        return sm.loss(R[-1])
    grads = dict(zip(sm.P, torch.autograd.grad(L_of(), list(sm.P.values()))))
    # the policy-term advantage is detached: freeze it at the base point for the finite difference
    base = [(float(b[1]),) for b in sm.buf]; G0, g = [], 0.0
    for b in reversed(sm.buf): g = b[3] + sm.gamma * g; G0.append(g)
    G0 = G0[::-1]; adv0 = [G0[i] - base[i][0] for i in range(T)]
    def frozen():
        sm.reset()
        for t in range(T): sm.step(X[t], reward_prev=R[t - 1] if t else 0.0, action=A[t])
        return float(sum(-adv0[i] * torch.log(b[0][b[2]]) + sm.c_v * 0.5 * (G0[i] - b[1]) ** 2 + sm.ent * (b[0] * torch.log(b[0] + 1e-12)).sum()
                         for i, b in enumerate(sm.buf)))
    fd = {}
    for nm, p in sm.P.items():
        for _ in range(3):
            idx = tuple(int(i) for i in np.unravel_index(rng.integers(p.numel()), p.shape)) if p.dim() else ()
            if nm == "W" and float(sm.mask[idx]) == 0: continue
            with torch.no_grad(): old = p[idx].item(); p[idx] = old + 1e-6
            up = frozen()
            with torch.no_grad(): p[idx] = old - 1e-6
            dn = frozen()
            with torch.no_grad(): p[idx] = old
            fd[nm] = max(fd.get(nm, 0.0), abs((up - dn) / 2e-6 - float(grads[nm][idx])))
    assert max(fd.values()) < 1e-6, fd

    # 3. recurrence matters: with W_rec = 0 the full BPTT gradient of sum c.z equals e-prop's truncated one; with
    #    W_rec != 0 they differ (the comparator really differs from e-prop only through recurrent-spike paths)
    def grads_sumcz(w_rec_zero):
        e = EpropAC(n_in, n_rec, k, kappa=0.8, dtype=F64, seed=4); e.W *= 1.5
        if w_rec_zero: e.W[:, n_in:] = 0
        C = torch.randn(T, n_rec, dtype=F64, generator=torch.Generator().manual_seed(9))
        s = SpikingBPTTAC(n_in, n_rec, k, kappa=0.8, dtype=F64, seed=4)
        with torch.no_grad(): s.P["W"].copy_(e.W)
        s.reset(); zs = []
        for t in range(T): s.step(X[t], action=A[t]); zs.append(s.z)
        g_full = torch.autograd.grad(sum(c @ z for c, z in zip(C, zs)), s.P["W"])[0]
        e.reset(); Eg = torch.zeros_like(e.W)
        for t in range(T):
            eb = e.ebar.clone(); e._neurons(X[t]); Eg += C[t][:, None] * (e.ebar - e.kappa * eb) / (1 - e.kappa)
        return float((g_full - Eg * e.mask).abs().max())
    d0, d1 = grads_sumcz(True), grads_sumcz(False)
    assert d0 < 1e-10 and d1 > 1e-6, (d0, d1)

    # 4. determinism and dtype: same seed -> identical weights; float64 storage exactly 2x float32
    def short(seed):
        net = SpikingBPTTAC(4, 20, 2, lr=1e-2, optimizer="adam", seed=seed); r = np.random.default_rng(seed)
        for _ in range(5):
            for t in range(8): net.step(r.random(4).astype(np.float32))
            np.random.rand(3); net.end(float(r.random() < .5))
        return torch.cat([p.detach().ravel() for p in net.P.values()])
    assert torch.equal(short(1), short(1))
    m32, m64 = SpikingBPTTAC(4, 40, 2).memory_bytes(16), SpikingBPTTAC(4, 40, 2, dtype=F64).memory_bytes(16)
    assert all(m64[c] == 2 * m32[c] for c in ("weights", "persistent_state", "experience_buffer_lower_bound"))

    # 5. synthetic sanity (float32), exactly 24,000 frames, SGD and Adam
    res = {opt: delayed_cue(optimizer=opt, lr={"sgd": 2e-2, "adam": 2e-3}[opt]) for opt in ("sgd", "adam")}
    print(json.dumps({"ok": True, "forward_equiv": dict(spikes=dz, policy=dp, value=dv, n_spikes=spikes),
                      "smooth_bptt_vs_fd": fd, "wrec0_full_vs_eprop": d0, "wrec_full_vs_eprop": d1,
                      "delayed_cue": res, "memory_bytes_T16_float32": m32}, indent=1))


def delayed_cue(episodes=1500, cue=5, delay=10, seed=0, **kw):
    net = SpikingBPTTAC(4, 40, 2, gamma=0.95, entropy=0.01, seed=seed, **kw); rng = np.random.default_rng(seed); hits = []
    for _ in range(episodes):
        side = int(rng.integers(2))
        for t in range(cue + delay + 1):
            x = np.zeros(4, np.float32); x[3] = 0.5 * (rng.random() < 0.2)
            if t < cue: x[side] = 1.0
            if t == cue + delay: x[2] = 1.0
            a = net.step(x)
        r = float(a == side); hits.append(r); net.end(r)
    assert net.count["frames"] == episodes * (cue + delay + 1)
    return dict(first_success=float(np.mean(hits[:300])), final_success=float(np.mean(hits[-300:])), counters=dict(net.count))


if __name__ == "__main__":
    {"check": check}[sys.argv[1]]()
