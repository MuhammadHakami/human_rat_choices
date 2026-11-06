"""Nevo-CRL-INSPIRED PPO adaptation on the E3 development pair (Claude, 2026-10-17). NOT a faithful reproduction.

Source method: Li et al., Neuro-evolutionary Continual RL, ICML 2026 (github yeshenpy/Nevo-CRL @ fe0456e1): per-task
population of sparse neuron masks evolved by episodic fitness (importance-guided crossover, low-activity pruning,
mutation); best mask stored at task end; parameters of its activated neurons frozen; later tasks reuse frozen neurons.
Port (deviations disclosed): PPO (reviewed RecurrentPPO loop, unchanged) instead of SAC; one shared weight set across
the mask population (original keeps one actor per member); no RND distillation; reuse of all previous-task neurons
(original: similarity-weighted from task text hints; two rooms of one task family). Masks are assigned to the 16
parallel environments, so mask search consumes the SAME 2 x 65,536 interactions (no extra environment frames).
Inner learning is backpropagation (PPO); masks/freezing are the only Nevo components. TASK ID IS GIVEN at evaluation
(mask/head of the probed room), unlike registered E3. Arms:
  nevo   : evolved masks + frozen activated weights (gradients of protected entries zeroed before clipping, fresh phase-B Adam, restore+assert after every Adam step)
  tid_ppo: task-ID-matched PPO control: full network, separate actor/critic head per room, shared body trained.
  python scripts/v4n_nevo_e3_claude_20261017.py selftest | fit ARM SEED
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wacv_continual_geometry_development_20261012 as dev
from ppo_feedforward_reference import FeedforwardPPO, FFAgent
from wacv_native_behavior_prior_20261012 import DeviceFFAgent  # noqa: F401 (unpickling source checkpoints)

ROOT = dev.ROOT; OUT = ROOT / "data/wacv/nevo_e3_20261017_v2"
CFG = dict(pop=5, sample_ratio_A=.5, top_ratio=.1, bar=.005, mutate=.1, evo_every=32, phase=65536)   # evo_every in PPO updates


class MaskedNet(torch.nn.Module):
    """FF 252-64-64 body; per-sample neuron masks via an appended population-index column (training) or eval masks."""
    def __init__(self, source, tasks=2):
        super().__init__()
        f = FFAgent(252, 4); f.load_state_dict(source.state_dict())
        self.l1, self.l2 = f.body[0], f.body[2]
        self.pi = torch.nn.ModuleList([copy.deepcopy(f.pi) for _ in range(tasks)])
        self.v = torch.nn.ModuleList([copy.deepcopy(f.v) for _ in range(tasks)])
        self.register_buffer("pop", torch.ones(CFG["pop"], 2, 64)); self.register_buffer("eval_mask", torch.ones(2, 64))
        self.task = 0; self.eval_task = 0; self.multihead = False; self.act = None

    def unroll(self, x, state, done):
        dev_ = self.l1.weight.device; x = x.to(dev_)
        if x.shape[-1] == 253: m = self.pop[x[..., -1].long()]; x = x[..., :252]; t = self.task
        else: m = self.eval_mask.expand(*x.shape[:-1], 2, 64); t = self.eval_task
        h1 = torch.tanh(self.l1(x)) * m[..., 0, :]; h2 = torch.tanh(self.l2(h1)) * m[..., 1, :]
        self.act = (h1.detach(), h2.detach()); self._t = t if self.multihead else 0
        return h2, state

    def heads(self, o):
        o = o.to(self.l1.weight.device); t = self._t
        return torch.distributions.Categorical(logits=self.pi[t](o).cpu()), self.v[t](o).squeeze(-1).cpu()


class PopEnv:
    """Appends each environment's population index (env i -> member i % pop)."""
    def __init__(self, env, n): self.env = env; self.idx = (np.arange(n) % CFG["pop"]).astype(np.float32); self.phase = "A"; self.expo = {}
    def add(self, o): return np.concatenate([o, self.idx[:, None]], 1).astype(np.float32)
    def step(self, a):
        for w in self.env.worlds: key = f"{self.phase}_room{w['configuration']}"; self.expo[key] = self.expo.get(key, 0) + 1   # actual room exposures
        o, r, d = self.env.step(a); return self.add(o), r, d


class Nevo:
    def __init__(self, net, rng, arm):
        self.net, self.rng, self.arm = net, rng, arm; self.frozen = torch.zeros(2, 64, dtype=torch.bool)
        self.best = {}; self.last_elite = 0; self.fit_sum = np.zeros(CFG["pop"]); self.fit_n = np.zeros(CFG["pop"]); self.imp = torch.zeros(CFG["pop"], 2, 64)
        self.log = []

    def start_task(self, t):
        self.net.task = t; self.fit_sum[:] = 0; self.fit_n[:] = 0
        if self.arm == "tid_ppo": self.net.pop.fill_(1.); return
        free = ~self.frozen; P = torch.zeros(CFG["pop"], 2, 64)
        for k in range(CFG["pop"]):
            for l in range(2):
                idx = torch.nonzero(free[l]).squeeze(-1); n = max(1, int(round(len(idx) * (CFG["sample_ratio_A"] if t == 0 else 1.0))))
                pick = idx[torch.as_tensor(self.rng.choice(len(idx), n, replace=False))]
                P[k, l, pick] = 1.; P[k, l] = torch.maximum(P[k, l], self.frozen[l].float())   # reuse frozen neurons
        self.net.pop.copy_(P.to(self.net.pop.device))

    def observe(self, B):
        """Fitness = goals per completed episode for each member; importance = mean |activation| per neuron."""
        rew, done = B["rew"].numpy(), B["done_after"].numpy(); member = B["obs"][0, :, -1].long().numpy()
        for e, k in enumerate(member): self.fit_sum[k] += rew[:, e].sum(); self.fit_n[k] += max(done[:, e].sum(), 1e-9)
        with torch.no_grad(): self.net.unroll(B["obs"], None, None)
        h1, h2 = self.net.act
        for k in range(CFG["pop"]):
            sel = torch.as_tensor(member == k)
            if sel.any(): self.imp[k, 0] = h1[:, sel.to(h1.device)].abs().mean((0, 1)).cpu(); self.imp[k, 1] = h2[:, sel.to(h2.device)].abs().mean((0, 1)).cpu()

    def evolve(self):
        if self.arm == "tid_ppo": return
        fit = self.fit_sum / np.maximum(self.fit_n, 1e-9); order = np.argsort(-fit); P = self.net.pop.cpu().clone()
        a, b = order[0], order[1]; free = ~self.frozen
        for k in order[2:]:                                              # replace non-elites by children of the two elites
            child = torch.zeros(2, 64)
            for l in range(2):
                pa, pb = P[a, l] > 0, P[b, l] > 0; ia, ib = self.imp[a, l], self.imp[b, l]
                top = int(np.ceil(CFG["top_ratio"] * 64))
                keep = torch.zeros(64, dtype=torch.bool)
                keep[torch.topk(ia * pa, top).indices] = True; keep[torch.topk(ib * pb, top).indices] = True
                rest = (pa | pb) & ~keep; pick = rest & torch.as_tensor(self.rng.random(64) < .5)
                c = (keep | pick) & pa.logical_or(pb)
                c &= ~(free[l] & (((ia + ib) / 2) < CFG["bar"]))          # prune low-activity free neurons
                flip = free[l] & c & torch.as_tensor(self.rng.random(64) < CFG["mutate"])
                c &= ~flip; c |= self.frozen[l]
                if not (c & free[l]).any(): c |= pa                       # never empty the trainable part
                child[l] = c.float()
            P[k] = child
        self.last_elite = int(a)
        self.log.append(dict(fitness=fit.tolist(), active=[int(x) for x in P.sum(-1).sum(-1)]))
        self.net.pop.copy_(P.to(self.net.pop.device)); self.fit_sum[:] = 0; self.fit_n[:] = 0

    def end_task(self, t):
        fit = self.fit_sum / np.maximum(self.fit_n, 1e-9) if self.fit_n.sum() else np.zeros(CFG["pop"])
        if self.arm != "nevo": k = 0
        elif self.fit_n.sum() > 0: k = int(np.argmax(fit))
        else: k = getattr(self, "last_elite", 0)          # final update was an evolution: elite masks are kept unchanged
        self.best[t] = self.net.pop[k].cpu().clone()
        if self.arm == "nevo": self.frozen |= self.best[t] > 0
        return k


def frozen_params(net, frozen):
    """Native policies.py 308-331 semantics: protect weights whose pre AND post units are both in the frozen set (inputs and
    outputs count as always active): full l1 rows of frozen h1 units, l2[frozen h2, frozen h1], head columns from frozen h2,
    biases of frozen units; shared head biases are frozen too (registered, more conservative)."""
    f1, f2 = frozen[0].to(net.l1.weight.device), frozen[1].to(net.l1.weight.device)
    out = {"l1.weight": f1[:, None].expand_as(net.l1.weight), "l1.bias": f1, "l2.weight": f2[:, None] & f1[None, :], "l2.bias": f2}
    for t in range(len(net.pi)):
        out[f"pi.{t}.weight"] = f2[None].expand_as(net.pi[t].weight); out[f"v.{t}.weight"] = f2[None].expand_as(net.v[t].weight)
        out[f"pi.{t}.bias"] = torch.ones_like(net.pi[t].bias, dtype=torch.bool) if f2.any() else torch.zeros_like(net.pi[t].bias, dtype=torch.bool)
        out[f"v.{t}.bias"] = torch.ones_like(net.v[t].bias, dtype=torch.bool) if f2.any() else torch.zeros_like(net.v[t].bias, dtype=torch.bool)
    return out


@torch.no_grad()
def a_outputs(m, nevo, obs):
    """Room-A actor logits and value on fixed observations under A's saved mask/head (mechanics check)."""
    net = m.net; net.eval_task = 0; net.eval_mask.copy_(nevo.best[0]); x = torch.as_tensor(obs[:, :252])[None]
    o, _ = net.unroll(x, None, None); t = 0
    return torch.cat([net.pi[t](o.to(net.l1.weight.device)), net.v[t](o.to(net.l1.weight.device))], -1).cpu()


def build(arm, seed):
    method = next(m for m in json.loads(dev.MANIFEST.read_text())["methods"] if m["id"] == "ppo")
    settings = json.loads(dev.ppo_source.PROTOCOL.read_text())["ppo"]; settings.update(method["ppo_settings"])
    m = FeedforwardPPO(253, 4, n_envs=16, lr=method["rate"], seed=27000 + seed, **{k: v for k, v in settings.items() if k not in ("gradient_clip", "adam_eps")})
    m.net = MaskedNet(dev.source_model(seed).net).to("cuda:0"); m.net.multihead = arm == "tid_ppo"
    m.opt = torch.optim.Adam(m.net.parameters(), lr=method["rate"], eps=1e-5)
    return m, method


def probe(m, nevo, stage, seed):
    t = 0 if stage == "A" else 1; net = m.net
    net.eval_task = t; net.eval_mask.copy_(nevo.best[t] if t in nevo.best else net.pop[0])
    return dev.probe(m, stage, seed, "clean")


def protect(m, fm, snap, counter):
    """Protected entries: zero gradient BEFORE clipping (tensor hooks), fresh Adam (no carried moments), and restore +
    assert after EVERY optimizer.step (post-step hook). Returns removable handles."""
    hs = [p.register_hook(lambda g, k=k: g.masked_fill(fm[k], 0.)) for k, p in m.net.named_parameters() if fm[k].any()]
    def post(opt, *_):
        counter["steps"] += 1
        with torch.no_grad():
            for k, p in m.net.named_parameters():
                if fm[k].any():
                    counter["max_drift_before_restore"] = max(counter["max_drift_before_restore"], float((p - snap[k])[fm[k]].abs().max()))
                    p[fm[k]] = snap[k][fm[k]]
    hs.append(m.opt.register_step_post_hook(post)); return hs


def fit(arm, seed, out=OUT, phase=None):
    phase = phase or CFG["phase"]; d = out / arm / f"seed{seed}"; d.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter(); m, method = build(arm, seed); rng = np.random.default_rng([20261017, seed, int(arm == "nevo")])
    nevo = Nevo(m.net, rng, arm); env = dev.environment(16, seed, "clean", phase); pe = PopEnv(env, 16)
    obs = pe.add(env.reset()); obsA = None; state = (torch.zeros(1, 16, 0), torch.zeros(1, 16, 0)); dp = np.zeros(16, np.float32)
    phases = {}; nevo.start_task(0); updates = 0; handles = []; counter = dict(steps=0, max_drift_before_restore=0.)
    c = m.cfg; steps_per_update = c["epochs"] * c["mb"]; updates_B = 0; snap = fm = None
    while m.count["frames"] < 2 * phase:
        if m.count["frames"] == phase:
            nevo.end_task(0); phases["A_after_A_learning"] = probe(m, nevo, "A", seed); obsA = obs.copy(); outA0 = a_outputs(m, nevo, obsA)
            torch.save(dict(model=m, best=dict(nevo.best), frozen=nevo.frozen.clone(), cfg=CFG), d / "checkpoint_afterA.pt"); nevo.start_task(1)
            m.opt = torch.optim.Adam(m.net.parameters(), lr=method["rate"], eps=1e-5)        # fresh per-phase optimizer, BOTH arms (disclosed)
            pe.phase = "B"; snap = {k: p.detach().clone() for k, p in m.net.named_parameters()}; fm = frozen_params(m.net, nevo.frozen)
            if arm == "nevo": handles = protect(m, fm, snap, counter)
        B, s0, obs, state, dp = m.rollout(pe, obs, state, dp)
        m.update(B, s0); nevo.observe(B); updates += 1; updates_B += pe.phase == "B"
        if updates % CFG["evo_every"] == 0: nevo.evolve()
    for h in handles: h.remove()
    nevo.end_task(1); outA1 = a_outputs(m, nevo, obsA)
    a_identity = dict(max_abs_logit_diff=float((outA1 - outA0).abs().max()))
    frozen_final = max((float((p.detach() - snap[k])[fm[k]].abs().max()) for k, p in m.net.named_parameters() if fm[k].any()), default=0.)
    free_moved = max(float((p.detach() - snap[k])[~fm[k]].abs().max()) for k, p in m.net.named_parameters() if (~fm[k]).any())
    checks = dict(adam_steps_phaseB=counter["steps"], expected_steps=updates_B * steps_per_update, max_drift_before_restore=counter["max_drift_before_restore"],
                  frozen_final_max_change=frozen_final, free_max_change=free_moved)
    if arm == "nevo":
        assert counter["steps"] == updates_B * steps_per_update and frozen_final == 0 and free_moved > 0 and a_identity["max_abs_logit_diff"] == 0, (checks, a_identity)
    phases["A_after_B_learning"] = probe(m, nevo, "A", seed); phases["B_after_learning"] = probe(m, nevo, "B", seed)
    rec = dict(arm=arm, seed=seed, status="complete", deviations_note=__doc__.split("Port")[1][:900], cfg=CFG, phase_frames_nominal=phase,
               frames=m.count["frames"], env_frames=env.frames, extra_mask_search_frames=0, method_rate=method["rate"],
               room_exposure_frames=pe.expo, phase_id_lag_note="training task index switches at the nominal frame; rooms switch at natural terminals",
               probes={k: {a: b for a, b in v.items() if a not in ("episodes", "rng_initial", "pose_audit")} for k, v in phases.items()},
               masks={str(t): [int(x) for x in v.sum(-1)] for t, v in nevo.best.items()}, frozen_neurons=[int(x) for x in nevo.frozen.sum(-1)],
               weight_checks=checks, a_output_identity=a_identity, evo_log=nevo.log[-4:], n_evolutions=len(nevo.log), wall_s=time.perf_counter() - t0,
               gpu=torch.cuda.get_device_name(0), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    torch.save(dict(model=m, best=dict(nevo.best), frozen=nevo.frozen.clone(), cfg=CFG), d / "checkpoint_afterB.pt")
    dev.ppo_source.controls.parent.gz_write(d / "probes.json.gz", phases)
    rec["files"] = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in ("checkpoint_afterA.pt", "checkpoint_afterB.pt", "probes.json.gz")}
    rec["source_checkpoint"] = next(r for r in json.loads(dev.MANIFEST.read_text())["source_models"] if r["seed"] == seed)
    (d / "result.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(arm=arm, seed=seed, status="complete", goals={k: v["goals"] for k, v in rec["probes"].items()}, checks=checks, wall_s=round(rec["wall_s"]))), flush=True)


def selftest():
    import tempfile
    for arm in ("nevo", "tid_ppo"):
        with tempfile.TemporaryDirectory() as td: fit(arm, 60, Path(td), phase=2048)
    print("SELFTEST_OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["selftest", "fit"]); ap.add_argument("arm", nargs="?"); ap.add_argument("seed", nargs="?", type=int)
    a = ap.parse_args(); torch.set_num_threads(1)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1") and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    selftest() if a.cmd == "selftest" else fit(a.arm, a.seed)
