"""Pretrained FF policy + zero-initialised learning-reservoir residual on the E3 development pair (Claude, 2026-10-17). v5.

Repair of the failed reservoir v4 (v4 kept intact, out of paper): v4's ridge readout had weight norm 46-61 and the policy
collapsed to ~0 entropy once the core drifted. Here the composite policy is
  logits = ff_pi(ff_body(x)) + res_pi(h_t),   value = ff_v(ff_body(x)) + res_v(h_t),
FF = the EXACT pretrained source body/heads (rat_bc seeds 60-62, loaded, trainable); h_t = capstone reservoir (random
capstone initialisation, 1024 units, fixed 10% support, rho 1.1, leak .5; Win/Wr/bias LEARN via truncated BPTT);
res_pi, res_v ZERO-initialised, so the initial policy equals the pretrained policy everywhere (asserted); no warm-up frames.
Backpropagation (reviewed PPO loop), not local, not spiking; task ID given at evaluation. Established concepts, not novel:
residual RL (Johannink et al., arXiv 1812.03201) and progressive networks (Rusu et al., arXiv 1606.04671). The residual is a
DESIGN DECISION motivated by v4's source-side collapse; that a lower learning rate could not fix v4 is NOT shown.
Arms:
  res_alloc: Nevo-inspired allocation mechanism control (no evolution). Room A: FF branch + random 50% of reservoir units;
             after A, freeze the ENTIRE FF branch and A's reservoir paths (Win rows, bias, Wr[A,A]), A residual columns and
             residual head biases; room B can learn only through the other reservoir units and their residual columns.
  res_tid  : task-ID control: separate (FF head + residual head) per room, B copies A, shared plastic FF body and reservoir.
  python scripts/v4n_ffres_e3_claude_20261017.py selftest | fit ARM SEED
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wacv_continual_geometry_development_20261012 as dev
import v4n_reservoir_e3_claude_20261017 as v4                         # frozen v4: reservoir(), rprobe(), ExpoEnv, a_outputs()
from ppo_feedforward_reference import FeedforwardPPO, FFAgent
from wacv_native_behavior_prior_20261012 import DeviceFFAgent  # noqa: F401 (unpickling source checkpoints)

ROOT = dev.ROOT; OUT = ROOT / "data/wacv/ffres_e3_20261017_v5"; NR, O, H = v4.NR, v4.O, 64; PHASE = 65536


class FFRes(torch.nn.Module):
    def __init__(self, source, seed):
        super().__init__(); f = FFAgent(O, 4); f.load_state_dict(source.state_dict())
        self.ff_body = f.body                                                             # ff_body.0 / ff_body.2 = l1 / l2 (tanh)
        self.ff_pi = torch.nn.ModuleList([copy.deepcopy(f.pi) for _ in range(2)]); self.ff_v = torch.nn.ModuleList([copy.deepcopy(f.v) for _ in range(2)])
        Wr, Win, b = v4.reservoir(9100 + seed); self.register_buffer("S", (Wr != 0).float())
        self.Wr = torch.nn.Parameter(Wr); self.Win = torch.nn.Parameter(Win); self.bres = torch.nn.Parameter(b)
        self.res_pi = torch.nn.ModuleList([torch.nn.Linear(NR, 4) for _ in range(2)]); self.res_v = torch.nn.ModuleList([torch.nn.Linear(NR, 1) for _ in range(2)])
        for m in (*self.res_pi, *self.res_v): torch.nn.init.zeros_(m.weight); torch.nn.init.zeros_(m.bias)
        self.register_buffer("mask", torch.ones(2, NR)); self.task = 0; self.eval_task = None; self.multihead = False

    def unroll(self, x, state, done):
        dv = self.Wr.device; x = x.to(dv); h = state[0][0].to(dv); out = []
        t_ = self.task if self.eval_task is None else self.eval_task; um = self.mask[t_]; W = self.Wr * self.S
        z = self.ff_body(x)
        for t in range(x.shape[0]):
            h = h * (1 - done[t].to(dv))[:, None]
            h = (.5 * h + .5 * torch.tanh(h @ W.T + x[t] @ self.Win.T + self.bres)) * um
            out.append(torch.cat([z[t], h], -1))
        return torch.stack(out), (h[None].cpu(), state[1])

    def heads(self, o):
        t = self.task if self.eval_task is None else self.eval_task; k = t if self.multihead else 0; o = o.to(self.Wr.device)
        z, h = o[..., :H], o[..., H:]
        return (torch.distributions.Categorical(logits=(self.ff_pi[k](z) + self.res_pi[k](h)).cpu()),
                (self.ff_v[k](z) + self.res_v[k](h)).squeeze(-1).cpu())


def build(arm, seed):
    method = next(m for m in json.loads(dev.MANIFEST.read_text())["methods"] if m["id"] == "ppo")
    st = json.loads(dev.ppo_source.PROTOCOL.read_text())["ppo"]; st.update(method["ppo_settings"])
    m = FeedforwardPPO(O, 4, n_envs=16, lr=method["rate"], seed=27000 + seed, **{k: v for k, v in st.items() if k not in ("gradient_clip", "adam_eps")})
    src = dev.source_model(seed).net.cpu(); m.net = FFRes(src, seed).to("cuda:0"); m.net.multihead = arm == "res_tid"
    m.opt = torch.optim.Adam(m.net.parameters(), lr=method["rate"], eps=1e-5)
    return m, method, src


@torch.no_grad()
def teacher_identity(net, src, seed):
    """Initial composite policy == pretrained source policy (logits and value) on real room-A observations."""
    env = dev.environment(16, seed, "clean", 256); x = torch.as_tensor(env.reset(), dtype=torch.float32)
    o, _ = net.unroll(x[None], (torch.zeros(1, 16, NR), None), torch.zeros(1, 16)); d, v = net.heads(o[0]); z = src.body(x)
    return float(max((d.logits - torch.log_softmax(src.pi(z), -1)).abs().max(), (v - src.v(z).squeeze(-1)).abs().max()))


def fit(arm, seed, out=OUT, phase=PHASE):
    d = out / arm / f"seed{seed}"; d.mkdir(parents=True, exist_ok=False); t0 = time.perf_counter()
    m, method, src = build(arm, seed); net = m.net; tid = teacher_identity(net, src, seed); assert tid < 1e-5, tid
    g = np.random.default_rng([20261017, seed, 77]); uA = torch.zeros(NR); uA[torch.as_tensor(g.choice(NR, NR // 2, replace=False))] = 1
    if arm == "res_alloc": net.mask[0].copy_(uA.to(net.mask.device))
    init = {k: p.detach().clone() for k, p in net.named_parameters()}; torch.save(dict(net=net.state_dict(), arm=arm), d / "checkpoint_init.pt")
    raw = dev.environment(16, seed, "clean", phase); env = v4.ExpoEnv(raw); obs = raw.reset()
    state = (torch.zeros(1, 16, NR), torch.zeros(1, 16, 0)); dp = np.zeros(16, np.float32)
    phases = {}; counter = dict(steps=0, drift=0.); handles = []; updates_B = 0; c = m.cfg
    while m.count["frames"] < 2 * phase:
        if m.count["frames"] == phase:
            phases["A_after_A_learning"] = v4.rprobe(m, "A", seed); obsA = np.asarray(obs).copy(); outA0 = v4.a_outputs(net, obsA)
            torch.save(dict(net=net.state_dict(), arm=arm), d / "checkpoint_afterA.pt")
            net.task = 1; env.phase = "B"; m.opt = torch.optim.Adam(net.parameters(), lr=method["rate"], eps=1e-5)
            if arm == "res_tid":
                with torch.no_grad():
                    for a_, b_ in ((net.ff_pi, None), (net.ff_v, None), (net.res_pi, None), (net.res_v, None)): a_[1].load_state_dict(a_[0].state_dict())
            snap = {k: p.detach().clone() for k, p in net.named_parameters()}; P = dict(net.named_parameters())
            fm = {k: torch.zeros_like(p, dtype=torch.bool) for k, p in P.items()}
            if arm == "res_alloc":
                u = net.mask[0] > 0
                for k in P:
                    if k.startswith(("ff_body.", "ff_pi.0.", "ff_v.0.", "res_pi.0.bias", "res_v.0.bias")): fm[k] = torch.ones_like(P[k], dtype=torch.bool)
                fm.update({"Win": u[:, None].expand_as(net.Win).clone(), "bres": u.clone(), "Wr": u[:, None] & u[None, :],
                           "res_pi.0.weight": u[None].expand_as(P["res_pi.0.weight"]).clone(), "res_v.0.weight": u[None].expand_as(P["res_v.0.weight"]).clone()})
                handles = [p.register_hook(lambda gr, k=k: gr.masked_fill(fm[k], 0.)) for k, p in P.items() if fm[k].any()]
                def post(opt, *_):
                    counter["steps"] += 1
                    with torch.no_grad():
                        for k, p in net.named_parameters():
                            if fm[k].any(): counter["drift"] = max(counter["drift"], float((p - snap[k])[fm[k]].abs().max())); p[fm[k]] = snap[k][fm[k]]
                handles.append(m.opt.register_step_post_hook(post))
        B, s0, obs, state, dp = m.rollout(env, obs, state, dp); m.update(B, s0); updates_B += env.phase == "B"
    for h in handles: h.remove()
    outA1 = v4.a_outputs(net, obsA); a_id = float((outA1 - outA0).abs().max())
    phases["A_after_B_learning"] = v4.rprobe(m, "A", seed); phases["B_after_learning"] = v4.rprobe(m, "B", seed)
    frozen_change = max((float((p.detach() - snap[k])[fm[k]].abs().max()) for k, p in net.named_parameters() if fm[k].any()), default=0.)
    free_A = {k: float((snap[k] - init[k]).abs().max()) for k in init}
    free_B = {k: float((p.detach() - snap[k])[~fm[k]].abs().max()) for k, p in net.named_parameters() if (~fm[k]).any()}
    traj_id = all(x["actions"] == y["actions"] and x["path"] == y["path"] for x, y in zip(phases["A_after_A_learning"]["episodes"], phases["A_after_B_learning"]["episodes"]))
    expected = updates_B * c["epochs"] * c["mb"]
    if arm == "res_alloc":
        assert counter["steps"] == expected and counter["drift"] == 0 and frozen_change == 0 and a_id == 0 and traj_id, (counter, expected, frozen_change, a_id, traj_id)
        assert all(free_B[k] > 0 for k in ("Win", "Wr", "res_pi.0.weight")), free_B
    assert all(free_A[k] > 0 for k in ("Win", "Wr", "res_pi.0.weight")), free_A                  # RNN + residual learn in A (both arms)
    ent = {k: float(np.mean([-(np.array(p) * np.log(np.clip(p, 1e-12, 1))).sum() for e in v["episodes"] for p in e["probabilities"]])) for k, v in phases.items()}
    torch.save(dict(net=net.state_dict(), arm=arm), d / "checkpoint_afterB.pt"); dev.ppo_source.controls.parent.gz_write(d / "probes.json.gz", phases)
    rec = dict(arm=arm, seed=seed, status="complete", note=__doc__[:2400], teacher_identity_max_abs=tid, frames_training=m.count["frames"], frames_warmup=0,
               phase_frames_nominal=phase, room_exposure_frames=env.expo,
               protection=dict(adam_steps_phaseB=counter["steps"], expected_steps=expected, drift_before_restore=counter["drift"], frozen_final_change=frozen_change,
                               phaseA_change=free_A, free_phaseB_change=free_B, a_output_identity_zero_state=a_id, a_trajectory_identity=traj_id),
               probe_entropy_nats=ent, probes={k: {a: b for a, b in v.items() if a not in ("episodes", "rng_initial")} for k, v in phases.items()},
               masks=dict(A_units=[int(i) for i in torch.nonzero(net.mask[0]).squeeze(-1)], A_unit_count=int(net.mask[0].sum())),
               sparsity=dict(recurrent_support=int(net.S.sum()), note="dense masked matmuls; no compute/spiking/energy claim"),
               wall_s=time.perf_counter() - t0, gpu=torch.cuda.get_device_name(0), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               v4_driver_sha256=hashlib.sha256((ROOT / "scripts/v4n_reservoir_e3_claude_20261017.py").read_bytes()).hexdigest(),
               source_checkpoint=next(r for r in json.loads(dev.MANIFEST.read_text())["source_models"] if r["seed"] == seed), shared_binding=dev.binding())
    rec["files"] = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in ("checkpoint_init.pt", "checkpoint_afterA.pt", "checkpoint_afterB.pt", "probes.json.gz")}
    (d / "result.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(arm=arm, seed=seed, status="complete", teacher_id=tid, prot={k: v for k, v in rec["protection"].items() if not k.startswith(("free_", "phaseA"))}, wall_s=round(rec["wall_s"]))), flush=True)


def selftest():
    import tempfile
    for arm in ("res_alloc", "res_tid"):
        with tempfile.TemporaryDirectory() as td: fit(arm, 60, Path(td), phase=1024)
    print("SELFTEST_OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["selftest", "fit"]); ap.add_argument("arm", nargs="?"); ap.add_argument("seed", nargs="?", type=int)
    a = ap.parse_args(); torch.set_num_threads(1)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1") and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    selftest() if a.cmd == "selftest" else fit(a.arm, a.seed)
