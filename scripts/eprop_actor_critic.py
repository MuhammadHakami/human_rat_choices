"""Reward-based e-prop actor-critic on a recurrent LIF/ALIF spiking network, PROTOTYPE (synthetic checks only).

Follows the reward-based e-prop derivation of Bellec et al. 2020 (Nat Commun 11:3625, Methods "Reward-based e-prop";
reference code github.com/IGITUGraz/eligibility_propagation). No real data are fitted here.

Neuron (discrete time, per step t), pre-synaptic vector p^t = [x^t, z^{t-1}] for W = [W_in | W_rec] (no self-synapses):
    v^t   = alpha v^{t-1} + W p^t - v_th z^{t-1}          reset by subtraction (no gradient through the reset)
    a^t   = rho a^{t-1} + z^{t-1}                          ALIF threshold adaptation (beta = 0 for LIF units)
    z^t   = H(v^t - v_th - beta a^t)                       spike; pseudo-derivative
    psi^t = (gamma_pd / v_th) max(0, 1 - |v^t - v_th - beta a^t| / v_th)
Eligibility (synapse-local, forward in time):
    eps_v^t = alpha eps_v^{t-1} + p^t                     (= dv^t/dW_ji holding other neurons' spikes fixed)
    e^t     = psi^t (eps_v^t - beta eps_a^t)               (= dz^t/dW_ji, same restriction)
    eps_a^{t+1} = psi^t eps_v^t + (rho - psi^t beta) eps_a^t
Readouts of kappa-filtered spikes zf^t = kappa zf^{t-1} + (1-kappa) z^t (normalised so zf is a rate in [0,1]; the paper's
unnormalised filter is the same model with W_out rescaled by 1/(1-kappa)): policy pi = softmax(W_out zf + b), value V = w_V.zf + b_V.
Broadcast learning signal (the only non-local quantity, one scalar per neuron per step, through feedback B):
    L_j^t = sum_k B_jk (pi_k^t - 1[a^t = k]) - c_V B^V_j     B = W_out^T ("symmetric") or fixed random ("random")
Per-synapse accumulators, discount-filtered:  G^t = gamma G^{t-1} + L^t ebar^t,  ebar^t = kappa ebar^{t-1} + (1-kappa) e^t
TD error, the third factor, applied ONLY to accumulators complete through t:
    delta^t = r^t + gamma V^{t+1} - V^t  (V^{T+1} = 0 at termination);   dW = -lr delta^t G^t
Ordering is the point of this file: delta^t is computed at step t+1 and multiplies G^t, before step t+1's eligibility
enters G (arc_fly.eprop.TDEligibility tags the new state first, so its credit is shifted by one transition).

Options: feedback symmetric|random; plastic_core=False freezes W (readout-only ablation); entropy bonus, as an
INDEPENDENT immediate term by default (gradient of -c_H H(pi^t) through e-bar^t, not multiplied by delta), as in the
official Atari code's separate entropy loss (Figure_4_and_5_ATARI/main.py, compute_entropy_loss; commit efd02e6);
entropy_mode="td_modulated" keeps the earlier variant (entropy folded into the delta-modulated accumulator) by name.
dtype is per instance (float32 for runs, float64 for derivative audits); importing this module changes no global
torch state.
NOTE on fidelity: this implements the paper's online Methods form (delta^t x discount-filtered accumulators, plain SGD).
The released Atari code instead trains an A2C-style unroll (n-step bootstrapped returns, Adam) with the e-prop
gradient inside the recurrent cell; that variant is not implemented here.
Counters: frames, spikes, synaptic events (spikes x out-degree), weight-update operations.
Not implemented yet (named for the protocol): refractoriness, firing-rate regulariser, adaptive feedback,
consolidation variants (TACOS w_ref, availability gating), each with its own memory accounting.
  python scripts/eprop_actor_critic.py check      # autograd equivalence checks + synthetic delayed-cue learning
"""
import json, sys
import numpy as np
import torch


class EpropAC:
    def __init__(self, n_in, n_rec, n_act, frac_alif=0.5, tau_m=20.0, tau_a=200.0, beta=1.0, v_th=0.6,
                 gamma_pd=0.3, kappa=None, gamma=0.95, c_v=0.5, lr=1e-3, lr_core=None, feedback="symmetric", plastic_core=True,
                 entropy=0.0, entropy_mode="independent", dtype=torch.float32, seed=0):
        assert entropy_mode in ("independent", "td_modulated")
        g = torch.Generator().manual_seed(seed); self.dt = dt = dtype; self.ent_mode = entropy_mode
        self.n_in, self.n, self.k = n_in, n_rec, n_act
        self.alpha, self.rho, self.v_th, self.gpd = np.exp(-1 / tau_m), np.exp(-1 / tau_a), v_th, gamma_pd
        self.kappa = self.alpha if kappa is None else kappa
        self.beta = torch.zeros(n_rec, dtype=dt); self.beta[: int(round(frac_alif * n_rec))] = beta
        self.gamma, self.c_v, self.lr, self.ent = gamma, c_v, lr, entropy
        self.lr_core = lr if lr_core is None else lr_core
        self.W = torch.cat([torch.randn(n_rec, n_in, generator=g, dtype=dt) / np.sqrt(n_in),
                            torch.randn(n_rec, n_rec, generator=g, dtype=dt) / np.sqrt(n_rec)], 1)
        self.mask = torch.ones_like(self.W); self.mask[:, n_in:].fill_diagonal_(0); self.W *= self.mask
        self.W_out = torch.randn(n_act, n_rec, generator=g, dtype=dt) / np.sqrt(n_rec); self.b_out = torch.zeros(n_act, dtype=dt)
        self.w_V = torch.zeros(n_rec, dtype=dt); self.b_V = torch.zeros((), dtype=dt)
        self.feedback, self.plastic = feedback, plastic_core
        self.B_rand = torch.randn(n_rec, n_act, generator=g, dtype=dt) / np.sqrt(n_act); self.BV_rand = torch.randn(n_rec, generator=g, dtype=dt)
        self.rng = np.random.default_rng(seed)
        self.count = dict(frames=0, spikes=0, synaptic_events=0, weight_updates=0, td_errors=0)
        self.reset()

    def memory_bytes(self):
        """ACTUAL tensor storage (element_size x numel), same categories as recurrent_ppo.RecurrentPPO.memory_bytes,
        plus fixed structure/feedback. Eligibility vectors, filtered traces and discount accumulators are per-synapse
        persistent state (4 copies of [W_in|W_rec], ALIF eps_a included)."""
        b = lambda *ts: int(sum(t.element_size() * t.numel() for t in ts))
        return dict(weights=b(self.W, self.W_out, self.b_out, self.w_V, self.b_V), optimiser=0,
                    fixed_structure=b(self.mask, self.beta),
                    fixed_feedback=b(self.B_rand, self.BV_rand),          # allocated; used only when feedback="random"
                    persistent_state=b(self.v, self.a, self.z, self.zf, self.eps_v, self.eps_a, self.ebar, self.G,
                                       self.G_out, self.G_bout, self.G_V, self.G_bV),
                    experience_buffer=0, dtype=str(self.dt))

    def reset(self):
        """Episode start: neuron state, traces and accumulators cleared (weights kept)."""
        n, P = self.n, self.n_in + self.n
        z = lambda *sh: torch.zeros(*sh, dtype=self.dt)
        self.v, self.a, self.z, self.zf = (z(n) for _ in range(4))
        self.eps_v, self.eps_a, self.ebar, self.G = (z(n, P) for _ in range(4))
        self.G_out, self.G_bout = z(self.k, n), z(self.k)
        self.G_V, self.G_bV = z(n), z(())
        self.prev = None                    # (V^{t-1}, r^{t-1}) waiting for V^t

    # ---- forward -------------------------------------------------------------------------------------------
    def _neurons(self, x):
        p = torch.cat([torch.as_tensor(x, dtype=self.dt), self.z])
        self.a = self.rho * self.a + self.z
        self.v = self.alpha * self.v + self.W @ p - self.v_th * self.z
        u = self.v - self.v_th - self.beta * self.a
        z = (u > 0).to(self.dt); psi = self.gpd / self.v_th * torch.clamp(1 - u.abs() / self.v_th, min=0)
        self.eps_v = self.alpha * self.eps_v + p[None, :]
        e = psi[:, None] * (self.eps_v - self.beta[:, None] * self.eps_a)
        self.eps_a = psi[:, None] * self.eps_v + (self.rho - psi * self.beta)[:, None] * self.eps_a
        self.z = z; self.zf = self.kappa * self.zf + (1 - self.kappa) * z; self.ebar = self.kappa * self.ebar + (1 - self.kappa) * e
        return psi

    def _readout(self):
        y = self.W_out @ self.zf + self.b_out; pi = torch.softmax(y, 0); V = self.w_V @ self.zf + self.b_V
        return pi, V

    def _B(self):
        return (self.W_out.T, self.w_V) if self.feedback == "symmetric" else (self.B_rand, self.BV_rand)

    # ---- learning ------------------------------------------------------------------------------------------
    def _apply(self, delta, accumulate=None):
        """dW = -lr delta G for accumulators complete through the transition delta belongs to."""
        self.count["td_errors"] += 1
        self._apply_grads(dict(W=delta * self.G, W_out=delta * self.G_out, b_out=delta * self.G_bout,
                               w_V=delta * self.G_V, b_V=delta * self.G_bV), accumulate)

    def _apply_grads(self, grads, accumulate=None):
        if accumulate is not None:
            for k, v in grads.items(): accumulate[k] = accumulate.get(k, 0) + v
            return
        if self.plastic:
            self.W -= self.lr_core * grads["W"] * self.mask; self.count["weight_updates"] += int(self.mask.sum())
        self.W_out -= self.lr * grads["W_out"]; self.b_out -= self.lr * grads["b_out"]
        self.count["weight_updates"] += self.W_out.numel() + self.k
        if "w_V" in grads:
            self.w_V -= self.lr * grads["w_V"]; self.b_V -= self.lr * grads["b_V"]; self.count["weight_updates"] += self.w_V.numel() + 1

    def step(self, x, reward_prev=0.0, action=None, accumulate=None):
        """Observe x^t (reward_prev = r^{t-1}, earned by a^{t-1}); return a^t. The TD error of the previous
        transition is applied before this step's eligibility enters the accumulators."""
        self._neurons(x); pi, V = self._readout()
        if self.prev is not None:
            V_prev, _ = self.prev
            self._apply(float(reward_prev + self.gamma * V - V_prev), accumulate)
        self.last_pi, self.last_value = pi.clone(), float(V)          # pre-update snapshot actually used (logs only)
        q = pi.double().numpy(); a = int(self.rng.choice(self.k, p=q / q.sum())) if action is None else int(action)   # float32-safe
        onehot = torch.zeros(self.k, dtype=self.dt); onehot[a] = 1.0
        dlogp = pi - onehot                                           # d(-log pi_a)/dy
        B, BV = self._B()
        if self.ent:
            logpi = torch.log(pi + 1e-12); H = -(pi * logpi).sum()
            dent = self.ent * pi * (logpi + H)                        # d(-c_H H)/dy
            if self.ent_mode == "td_modulated":
                dlogp = dlogp + dent
            else:                                                     # immediate, not delta-modulated
                self._apply_grads(dict(W=(B @ dent)[:, None] * self.ebar, W_out=dent[:, None] * self.zf[None, :], b_out=dent), accumulate)
        L = B @ dlogp - self.c_v * BV                                  # broadcast learning signal per neuron
        self.G = self.gamma * self.G + L[:, None] * self.ebar
        self.G_out = self.gamma * self.G_out + dlogp[:, None] * self.zf[None, :]; self.G_bout = self.gamma * self.G_bout + dlogp
        self.G_V = self.gamma * self.G_V - self.c_v * self.zf; self.G_bV = self.gamma * self.G_bV - self.c_v
        self.prev = (float(V), None)
        s = int(self.z.sum()); self.count["frames"] += 1; self.count["spikes"] += s
        self.count["synaptic_events"] += s * ((self.n - 1) + self.k + 1)
        return a

    def end(self, reward_last, accumulate=None):
        """Terminal transition: delta^T = r^T - V^T, then clear traces for the next episode."""
        if self.prev is not None: self._apply(float(reward_last - self.prev[0]), accumulate)
        self.reset()


# ---- independent checks (all in float64, set per instance; no global dtype change) ----------------------------
F64 = torch.float64


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v_th, gpd):
        ctx.save_for_backward(u); ctx.c = (v_th, gpd); return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors; v_th, gpd = ctx.c
        return g * gpd / v_th * torch.clamp(1 - u.abs() / v_th, min=0), None, None


def _autodiff_run(net, W, W_out, b_out, w_V, b_V, X):
    """Same network, written with autograd; other neurons' spikes (recurrent input) and the reset are detached,
    which is exactly the restriction under which e-prop's eligibility is the true derivative."""
    n = net.n; v = a = z = zf = torch.zeros(n, dtype=F64); out = []
    for x in X:
        p = torch.cat([torch.as_tensor(x, dtype=F64), z.detach()])
        a = net.rho * a + z
        v = net.alpha * v + (W * net.mask) @ p - net.v_th * z.detach()
        u = v - net.v_th - net.beta * a
        z = _Spike.apply(u, net.v_th, net.gpd); zf = net.kappa * zf + (1 - net.kappa) * z
        out.append((z, torch.softmax(W_out @ zf + b_out, 0), w_V @ zf + b_V))
    return out


def _rule_vs_autodiff(net, X, acts, rew):
    """Accumulated e-prop update vs autograd of: sum_t delta^t sum_{s<=t} gamma^{t-s}[-log pi(a_s) - c_V V_s]
    + sum_t c_H sum_k pi_k log pi_k (the independent entropy term), delta held fixed. Returns (max errors, acc)."""
    T = len(X); net.reset(); acc = {}
    for t in range(T): net.step(X[t], reward_prev=rew[t - 1] if t else 0.0, action=acts[t], accumulate=acc)
    net.end(rew[T - 1], accumulate=acc)
    P = {nm: getattr(net, nm).clone().requires_grad_(True) for nm in ("W", "W_out", "b_out", "w_V", "b_V")}
    out = _autodiff_run(net, P["W"], P["W_out"], P["b_out"], P["w_V"], P["b_V"], X)
    Vs = torch.stack([o[2] for o in out]).detach()
    delta = [float(rew[t] + (net.gamma * Vs[t + 1] if t + 1 < T else 0.0) - Vs[t]) for t in range(T)]
    J = 0.0
    for t in range(T):
        for s in range(t + 1):
            pi, V = out[s][1], out[s][2]
            J = J + delta[t] * net.gamma ** (t - s) * (-torch.log(pi[acts[s]]) - net.c_v * V)
        J = J + net.ent * (out[t][1] * torch.log(out[t][1] + 1e-12)).sum()
    J.backward()
    err = {nm: float(((acc[nm] * (net.mask if nm == "W" else 1)) - P[nm].grad).abs().max()) if nm in acc else
           float(P[nm].grad.abs().max()) for nm in P}
    return err, acc, delta


def check():
    assert torch.get_default_dtype() == torch.float32, "importing this module must not change torch's default dtype"
    T, n_in, n_rec, k = 40, 5, 12, 3
    rng = np.random.default_rng(1); X = rng.random((T, n_in)) * 2.0
    net = EpropAC(n_in, n_rec, k, gamma=0.9, kappa=0.8, dtype=F64, seed=3)
    net.w_V = torch.randn(n_rec, dtype=F64) * 0.3; net.W *= 1.5

    # 1. eligibility trace == local partial derivative dz^t/dW (ALIF + LIF units, recurrent W included)
    W = net.W.clone().requires_grad_(True); C = torch.randn(T, n_rec, dtype=F64)
    zs = _autodiff_run(net, W, net.W_out, net.b_out, net.w_V, net.b_V, X)
    (sum(c @ z for c, (z, _, _) in zip(C, zs))).backward()
    E = torch.zeros_like(net.W); net.reset()
    for t in range(T):
        e_before = net.ebar.clone(); net._neurons(X[t]); e = (net.ebar - net.kappa * e_before) / (1 - net.kappa)
        E += C[t][:, None] * e
    err1 = float((E * net.mask - W.grad).abs().max()); spikes = int(sum(float(z.sum()) for z, _, _ in zs))
    assert spikes > 0 and err1 < 1e-10, (err1, spikes)

    # 2. full rule == truncated autodiff, WITH a nonzero independent entropy term (symmetric feedback, delta fixed)
    acts = rng.integers(k, size=T); rew = rng.normal(size=T)
    net.ent = 0.05; err2, acc, delta = _rule_vs_autodiff(net, X, acts, rew)
    assert max(err2.values()) < 1e-9, err2

    # 2b. entropy survives a zero TD error: V == 0 (w_V = b_V = 0, c_V = 0) and r == 0 make every delta exactly 0;
    #     the accumulated update must then equal the entropy gradient alone, and be nonzero
    net0 = EpropAC(n_in, n_rec, k, gamma=0.9, kappa=0.8, c_v=0.0, entropy=0.05, dtype=F64, seed=3); net0.W *= 1.5
    err2b, acc0, delta0 = _rule_vs_autodiff(net0, X, acts, np.zeros(T))
    assert max(abs(d) for d in delta0) == 0.0 and max(err2b.values()) < 1e-9, (err2b, max(map(abs, delta0)))
    ent_norm = float(acc0["W"].abs().max()); assert ent_norm > 1e-6, ent_norm
    net0m = EpropAC(n_in, n_rec, k, gamma=0.9, kappa=0.8, c_v=0.0, entropy=0.05, entropy_mode="td_modulated", dtype=F64, seed=3); net0m.W *= 1.5
    net0m.reset(); accm = {}
    for t in range(T): net0m.step(X[t], action=acts[t], accumulate=accm)
    net0m.end(0.0, accumulate=accm)
    assert float(accm["W"].abs().max()) == 0.0, "td_modulated entropy should vanish when every delta is 0"

    # 3. ordering: a one-transition-late application (tagging step t+1 before delta^t) gives a different update
    G_hist = []; net.ent = 0.0; net.reset()
    for t in range(T):
        net.step(X[t], action=acts[t], accumulate={}); G_hist.append(net.G.clone())
    err_pg, acc_pg, delta_pg = _rule_vs_autodiff(net, X, acts, rew)
    shifted = sum(delta_pg[t] * G_hist[min(t + 1, T - 1)] for t in range(T))
    assert float((shifted - acc_pg["W"]).abs().max()) > 1e-6, "ordering check is not sensitive"

    # 4. memory accounting reports actual storage: float64 is exactly twice float32
    m32, m64 = EpropAC(4, 40, 2).memory_bytes(), EpropAC(4, 40, 2, dtype=F64).memory_bytes()
    assert all(m64[c] == 2 * m32[c] for c in ("weights", "persistent_state", "fixed_feedback", "fixed_structure"))

    # 5. synthetic learning sanity (float32): delayed cue (cue 5 steps, delay 10, choose at the end; reward +1 if correct)
    res = {}
    for name, kw in (("eprop_alif", {}), ("frozen_core", dict(plastic_core=False)), ("random_feedback", dict(feedback="random"))):
        res[name] = delayed_cue(**kw)
    print(json.dumps({"ok": True, "eligibility_max_abs_err": err1, "rule_max_abs_err_with_entropy": err2,
                      "entropy_only_zero_delta_err": err2b, "entropy_only_update_max": ent_norm, "spikes_in_check": spikes,
                      "delayed_cue": res, "memory_bytes_float32": m32}, indent=1))


def delayed_cue(episodes=1500, cue=5, delay=10, seed=0, **kw):
    net = EpropAC(4, 40, 2, lr=2e-2, lr_core=2e-3, gamma=0.95, entropy=0.01, seed=seed, **kw); rng = np.random.default_rng(seed); hits = []
    for _ in range(episodes):
        side = int(rng.integers(2)); r = 0.0
        for t in range(cue + delay + 1):
            x = np.zeros(4, np.float32); x[3] = 0.5 * (rng.random() < 0.2)  # background noise channel
            if t < cue: x[side] = 1.0
            if t == cue + delay: x[2] = 1.0                                # go signal
            a = net.step(x, reward_prev=0.0)
        r = float(a == side); hits.append(r); net.end(r)
    return dict(final_success=float(np.mean(hits[-300:])), first_success=float(np.mean(hits[:300])),
                counters=dict(net.count))


if __name__ == "__main__":
    {"check": check}[sys.argv[1]]()
