"""Capstone-initialised LEARNING reservoir + linear readout on the E3 development pair (Claude, 2026-10-17). Descriptive pilot, v3.

Substrate (weights LEARN on a FIXED sparse structure; user directive 10:20 UTC): the capstone echo-state reservoir (NeuroAI_EmbodiedAi_Capstone notebooks/plausible_learners.py @ bebf793,
_reservoir: Nr 4096 (here 1024 for time), 10% recurrent connectivity, spectral radius 1.1, leak a .5, bias .1 randn). The core starts at this random capstone initialisation (no pretrained reservoir weights); only the
readout is distilled from the pretrained source policy. Sparse support enforced by masking dense matmuls: no compute,
spiking or energy gain is claimed.
Adaptation: input gain 1/sqrt(252) (capstone sin 1.0 was set for ~20-D MotorNet observations); no obs normalisation.
Learning: the reviewed PPO loop (truncated BPTT over each 16-step segment) updates input, recurrent (on the fixed 10%
support only) and readout weights; backprop, not a local rule. Warm start: readout ridge-fitted (lambda 1e-2) to the pretrained source policy's logits and value on
A-mode reservoir features (arm's A unit mask applied first) of 4,096 extra room-A interactions collected by that source
policy (offline least squares; counted: total interactions 135,168).
Arms (task ID given at evaluation, like the Nevo pilot):
  res_alloc: task allocation, Nevo-inspired mechanism control (NOT Nevo-CRL: no evolution): room A's readout may use a
             random 50% of reservoir units (+ all observation inputs); after A those readout columns and both biases are
             frozen; room B uses all units; A is evaluated with A's unit mask. Protection as in the Nevo rev2 driver, native pre AND post semantics: Win rows,
             biases and Wr[A,A] of A units plus A readout columns/biases; unallocated units are silenced in A mode.
  res_tid  : task-ID-matched control: separate readout per room, B's readout initialised from A's, no freezing.
  python scripts/v4n_reservoir_e3_claude_20261017.py selftest | fit ARM SEED
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wacv_continual_geometry_development_20261012 as dev
from ppo_feedforward_reference import FeedforwardPPO
from wacv_native_behavior_prior_20261012 import DeviceFFAgent  # noqa: F401 (unpickling source checkpoints)

ROOT = dev.ROOT; OUT = ROOT / "data/wacv/reservoir_e3_20261017_v3"; NR, O = 1024, 252; PHASE = 65536


def reservoir(seed):
    g = torch.Generator().manual_seed(seed); Wr = torch.randn(NR, NR, generator=g); Wr *= (torch.rand(NR, NR, generator=g) < .1).float()
    Wr *= 1.1 / torch.linalg.eigvals(Wr).abs().max().item()
    return Wr, torch.randn(NR, O, generator=g) / np.sqrt(O), .1 * torch.randn(NR, generator=g)


class ResNet(torch.nn.Module):
    def __init__(self, seed):
        super().__init__(); Wr, Win, b = reservoir(9100 + seed)
        self.register_buffer("S", (Wr != 0).float())                                   # FIXED sparse structure (10%)
        self.Wr = torch.nn.Parameter(Wr); self.Win = torch.nn.Parameter(Win); self.bres = torch.nn.Parameter(b)   # weights LEARN
        self.pi = torch.nn.ModuleList([torch.nn.Linear(NR + O, 4) for _ in range(2)]); self.v = torch.nn.ModuleList([torch.nn.Linear(NR + O, 1) for _ in range(2)])
        self.register_buffer("mask", torch.ones(2, NR + O)); self.task = 0; self.eval_task = None; self.multihead = False

    def unroll(self, x, state, done):
        dv = self.Wr.device; x = x.to(dv); h = state[0][0].to(dv); out = []
        t_ = self.task if self.eval_task is None else self.eval_task; um = self.mask[t_, :NR]; W = self.Wr * self.S
        for t in range(x.shape[0]):
            h = h * (1 - done[t].to(dv))[:, None]
            h = (.5 * h + .5 * torch.tanh(h @ W.T + x[t] @ self.Win.T + self.bres)) * um   # unallocated units silent
            out.append(torch.cat([h, x[t]], -1))
        return torch.stack(out), (h[None].cpu(), state[1])

    def heads(self, o):
        t = self.task if self.eval_task is None else self.eval_task; k = t if self.multihead else 0
        o = o.to(self.Wr.device) * self.mask[t]
        return torch.distributions.Categorical(logits=self.pi[k](o).cpu()), self.v[k](o).squeeze(-1).cpu()


@torch.no_grad()
def rprobe(m, stage, seed):
    """dev.probe semantics (same schedule, generator seed, episode records) with the reservoir state carried within an
    episode and RESET at every actual environment terminal; weights verified unchanged."""
    cfg = 100 + int(stage == "B"); geometry = json.loads((dev.worlds.fresh.BASE / "geometry_manifest.json").read_text())
    schedule = [dict(configuration=cfg, spawn=s_, rotation_index=r) for r in range(3) for s_ in geometry["starts"]]
    net = copy.deepcopy(m.net); net.eval_task = int(stage == "B"); tg = torch.Generator(device="cpu").manual_seed(230017 + seed)
    initial_rng = tg.get_state().tolist(); env = dev.environment(1, seed, "clean", 45 * len(schedule) // 2, schedule)
    obs = env.reset(); before = {k: p.clone() for k, p in net.named_parameters()}; h = torch.zeros(1, 1, NR); probs, values = [], []
    while not env.exhausted and env.frames < env.limit:
        o, (h, _) = net.unroll(torch.as_tensor(obs, dtype=torch.float32)[None], (h, None), torch.zeros(1, 1))
        dist, v = net.heads(o[0]); a = torch.multinomial(dist.probs, 1, generator=tg).squeeze(-1)
        probs.append(dist.probs[0].tolist()); values.append(float(v[0])); obs, _, done = env.step(a.numpy())
        if done[0]: h = torch.zeros(1, 1, NR)
    assert all(torch.equal(p, before[k]) for k, p in net.named_parameters())
    episodes = dev.ppo_source.controls.parent.base.episodes(env, probs, values)
    return dict(goals=sum(e["success"] for e in episodes), trials=len(episodes), frames=env.frames, episodes=episodes, rng_initial=initial_rng,
                weights_frozen_verified=True, state_reset_at_terminal=True)


class ExpoEnv:
    """Counts actual room exposures per environment step and phase label."""
    def __init__(self, env): self.env = env; self.phase = "A"; self.expo = {}
    def step(self, a):
        for w in self.env.worlds: k = f"{self.phase}_room{w['configuration']}"; self.expo[k] = self.expo.get(k, 0) + 1
        return self.env.step(a)


@torch.no_grad()
def a_outputs(net, obs):
    """Room-A logits/value from a ZEROED reservoir state on fixed observations (mechanics identity check)."""
    n = net.eval_task; net.eval_task = 0; o, _ = net.unroll(torch.as_tensor(obs, dtype=torch.float32)[None], (torch.zeros(1, len(obs), NR), None), torch.zeros(1, len(obs)))
    d, v = net.heads(o[0]); net.eval_task = n; return torch.cat([d.logits, v[:, None]], -1)


def build(arm, seed):
    method = next(m for m in json.loads(dev.MANIFEST.read_text())["methods"] if m["id"] == "ppo")
    st = json.loads(dev.ppo_source.PROTOCOL.read_text())["ppo"]; st.update(method["ppo_settings"])
    m = FeedforwardPPO(O, 4, n_envs=16, lr=method["rate"], seed=27000 + seed, **{k: v for k, v in st.items() if k not in ("gradient_clip", "adam_eps")})
    m.net = ResNet(seed).to("cuda:0"); m.net.multihead = arm == "res_tid"; m.opt = torch.optim.Adam(m.net.parameters(), lr=method["rate"], eps=1e-5)
    return m, method


@torch.no_grad()
def warm_start(m, seed, frames=4096):
    """Ridge-fit the readout to the pretrained source policy on reservoir features of source-policy room-A frames."""
    src = dev.source_model(seed).net.cpu(); env = dev.environment(16, seed, "clean", frames); obs = env.reset()
    h = torch.zeros(1, 16, NR); F, Y = [], []; tg = torch.Generator().manual_seed(551000 + seed); rng0 = tg.get_state().tolist()
    while env.frames < frames:
        x = torch.as_tensor(obs, dtype=torch.float32); o, (h, _) = m.net.unroll(x[None], (h, None), torch.zeros(1, 16))
        z = src.body(x); logits, val = src.pi(z), src.v(z)
        F.append(o[0].cpu()); Y.append(torch.cat([logits, val], -1))
        a = torch.multinomial(torch.softmax(logits, -1), 1, generator=tg).squeeze(-1); obs, _, done = env.step(a.numpy()); h = h * torch.as_tensor(1 - done.astype(np.float32))[None, :, None]
    F, Y = torch.cat(F).double(), torch.cat(Y).double(); F1 = torch.cat([F, torch.ones(len(F), 1, dtype=F.dtype)], 1)
    W = torch.linalg.solve(F1.T @ F1 + 1e-2 * torch.eye(F1.shape[1], dtype=F.dtype), F1.T @ Y).float().T.to(m.net.Wr.device)
    for k in range(2):
        m.net.pi[k].weight.copy_(W[:4, :-1]); m.net.pi[k].bias.copy_(W[:4, -1]); m.net.v[k].weight.copy_(W[4:, :-1]); m.net.v[k].bias.copy_(W[4:, -1])
    fit_err = float(((F1.float() @ W.cpu().T) - Y.float()).pow(2).mean() / Y.float().var(0).mean())
    return dict(frames=len(F), relative_mse=fit_err, teacher_action_rng_initial=rng0, teacher_generator_seed=551000 + seed)


def fit(arm, seed, out=OUT, phase=PHASE, warm_frames=4096):
    d = out / arm / f"seed{seed}"; d.mkdir(parents=True, exist_ok=False); t0 = time.perf_counter()
    m, method = build(arm, seed); net = m.net
    g = np.random.default_rng([20261017, seed, 77]); unitsA = torch.zeros(NR + O); unitsA[torch.as_tensor(g.choice(NR, NR // 2, replace=False))] = 1; unitsA[NR:] = 1
    if arm == "res_alloc": net.mask[0].copy_(unitsA.to(net.mask.device))                   # A mask BEFORE distillation
    ws = warm_start(m, seed, warm_frames); init = {k: p.detach().clone() for k, p in net.named_parameters()}
    torch.save(dict(net=net.state_dict(), arm=arm, warm_start=ws), d / "checkpoint_afterwarm.pt")
    raw = dev.environment(16, seed, "clean", phase); env = ExpoEnv(raw); obs = raw.reset()
    state = (torch.zeros(1, 16, NR), torch.zeros(1, 16, 0)); dp = np.zeros(16, np.float32)
    phases = {}; counter = dict(steps=0, drift=0.); handles = []; updates_B = 0; c = m.cfg
    while m.count["frames"] < 2 * phase:
        if m.count["frames"] == phase:
            phases["A_after_A_learning"] = rprobe(m, "A", seed); obsA = np.asarray(obs).copy(); outA0 = a_outputs(net, obsA)
            torch.save(dict(net=net.state_dict(), arm=arm), d / "checkpoint_afterA.pt")
            net.task = 1; env.phase = "B"; m.opt = torch.optim.Adam(net.parameters(), lr=method["rate"], eps=1e-5)
            if arm == "res_tid":
                with torch.no_grad(): net.pi[1].load_state_dict(net.pi[0].state_dict()); net.v[1].load_state_dict(net.v[0].state_dict())
            snap = {k: p.detach().clone() for k, p in net.named_parameters()}; P = dict(net.named_parameters())
            fm = {k: torch.zeros_like(p, dtype=torch.bool) for k, p in P.items()}
            if arm == "res_alloc":
                col = net.mask[0] > 0; u = col[:NR]
                fm.update({"Win": u[:, None].expand_as(net.Win).clone(), "bres": u.clone(), "Wr": u[:, None] & u[None, :]})
                for k in ("pi.0", "v.0"): fm[k + ".weight"] = col[None].expand_as(P[k + ".weight"]).clone(); fm[k + ".bias"] = torch.ones_like(P[k + ".bias"], dtype=torch.bool)
                handles = [p.register_hook(lambda gr, k=k: gr.masked_fill(fm[k], 0.)) for k, p in P.items() if fm[k].any()]
                def post(opt, *_):
                    counter["steps"] += 1
                    with torch.no_grad():
                        for k, p in net.named_parameters():
                            if fm[k].any(): counter["drift"] = max(counter["drift"], float((p - snap[k])[fm[k]].abs().max())); p[fm[k]] = snap[k][fm[k]]
                handles.append(m.opt.register_step_post_hook(post))
        B, s0, obs, state, dp = m.rollout(env, obs, state, dp); m.update(B, s0); updates_B += env.phase == "B"
    for h in handles: h.remove()
    outA1 = a_outputs(net, obsA); a_id = float((outA1 - outA0).abs().max())
    phases["A_after_B_learning"] = rprobe(m, "A", seed); phases["B_after_learning"] = rprobe(m, "B", seed)
    frozen_change = max((float((p.detach() - snap[k])[fm[k]].abs().max()) for k, p in net.named_parameters() if fm[k].any()), default=0.)
    free_change = {k: float((p.detach() - init[k])[~fm[k]].abs().max()) for k, p in net.named_parameters() if (~fm[k]).any()}      # whole training (secondary)
    free_phaseB = {k: float((p.detach() - snap[k])[~fm[k]].abs().max()) for k, p in net.named_parameters() if (~fm[k]).any()}      # phase-B plasticity
    traj_id = all(x["actions"] == y["actions"] and x["path"] == y["path"] for x, y in zip(phases["A_after_A_learning"]["episodes"], phases["A_after_B_learning"]["episodes"]))
    expected = updates_B * c["epochs"] * c["mb"]
    if arm == "res_alloc":
        assert counter["steps"] == expected and counter["drift"] == 0 and frozen_change == 0 and a_id == 0 and traj_id, (counter, expected, frozen_change, a_id, traj_id)
        assert all(free_phaseB[k] > 0 for k in ("Win", "Wr", "pi.0.weight")), free_phaseB
    torch.save(dict(net=net.state_dict(), arm=arm), d / "checkpoint_afterB.pt"); dev.ppo_source.controls.parent.gz_write(d / "probes.json.gz", phases)
    rec = dict(arm=arm, seed=seed, status="complete", note=__doc__[:2200], warm_start=ws, frames_training=m.count["frames"], frames_warmup=ws["frames"],
               frames_total=m.count["frames"] + ws["frames"], phase_frames_nominal=phase, room_exposure_frames=env.expo,
               protection=dict(adam_steps_phaseB=counter["steps"], expected_steps=expected, drift_before_restore=counter["drift"], frozen_final_change=frozen_change,
                               free_max_change_whole_training=free_change, free_phaseB_change=free_phaseB, a_output_identity_zero_state=a_id, a_trajectory_identity=traj_id),
               probes={k: {a: b for a, b in v.items() if a not in ("episodes", "rng_initial")} for k, v in phases.items()},
               masks=dict(A_units=[int(i) for i in torch.nonzero(net.mask[0, :NR]).squeeze(-1)], A_unit_count=int(net.mask[0, :NR].sum()), B_unit_count=int(net.mask[1, :NR].sum())),
               sparsity=dict(recurrent_support=int(net.S.sum()), recurrent_support_fraction=float(net.S.mean()),
                             recurrent_support_within_A=int((net.S * net.mask[0, :NR][:, None] * net.mask[0, :NR][None]).sum()), note="dense masked matmuls; no compute/spiking/energy claim"),
               wall_s=time.perf_counter() - t0, gpu=torch.cuda.get_device_name(0), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               source_checkpoint=next(r for r in json.loads(dev.MANIFEST.read_text())["source_models"] if r["seed"] == seed), shared_binding=dev.binding())
    rec["files"] = {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in ("checkpoint_afterwarm.pt", "checkpoint_afterA.pt", "checkpoint_afterB.pt", "probes.json.gz")}
    (d / "result.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(arm=arm, seed=seed, status="complete", goals={k: v["goals"] for k, v in rec["probes"].items()}, prot={k: v for k, v in rec["protection"].items() if not k.startswith("free_")}, freeB={k: round(v, 5) for k, v in rec["protection"]["free_phaseB_change"].items() if k in ("Win", "Wr", "pi.0.weight", "pi.1.weight")}, wall_s=round(rec["wall_s"]))), flush=True)


def selftest():
    import tempfile
    for arm in ("res_alloc", "res_tid"):
        with tempfile.TemporaryDirectory() as td: fit(arm, 60, Path(td), phase=1024, warm_frames=512)
    print("SELFTEST_OK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["selftest", "fit"]); ap.add_argument("arm", nargs="?"); ap.add_argument("seed", nargs="?", type=int)
    a = ap.parse_args(); torch.set_num_threads(1)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1") and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
    selftest() if a.cmd == "selftest" else fit(a.arm, a.seed)
