"""Direct post-hoc policy-confidence control for E2 (Codex design 2026-10-13; PRIMARY confidence control).
Hypothesis tested: the human-state shortest-path expert loses zero-shot only because its trained actor is over-confident.
Per seed s: take the trained source checkpoints human_expert(s) (seeds30_followup) and human_bc(s) (seeds30); on the
identical human SOURCE observations (6000 prior rows x 3 rotations = 18,000 BC views) find T in [.05, 20] by bisection
so that the mean conditional entropy of softmax(logits_expert / T) equals that of the human_bc actor. Objective =
entropy only; no OOD/E1/E2 success is read. A target outside the bounds is a retained bound failure (T at the bound).
The checkpoint is then frozen with pi.weight and pi.bias scaled by 1/T (critic untouched, no BC/PPO retraining) and
probed zero-shot in the same 16 rooms as E2. HARDWARE-MATCHED (Codex 2026-10-13): each expert is calibrated against the
human_bc actor trained on the SAME GPU model (3090: seeds30 human_bc + hw_counterbalance experts; 4090: hw_counterbalance
human_bc + seeds30_followup experts), and T1 averages the two within-hardware contrasts per seed. Human only.
  python scripts/temperature_expert_control_claude.py check            (synthetic net; no real checkpoint)
  python scripts/temperature_expert_control_claude.py dry 3090|4090    (seed 1000 calibration into dry/, nothing probed)
  python scripts/temperature_expert_control_claude.py calibrate        (needs bound GO; writes scaled checkpoints)
  python scripts/temperature_expert_control_claude.py ood W N | analyse
"""
import copy, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "data/audit/wacv_native_behavior_prior_20261012"
S30, FU, HWC = ROOT / "data/wacv/seeds30/cohort", ROOT / "data/wacv/seeds30_followup/cohort", ROOT / "data/wacv/hw_counterbalance/cohort"
SRC = {"3090": dict(expert=HWC, bc=S30), "4090": dict(expert=FU, bc=HWC)}          # GPU model -> checkpoint roots
OUT = ROOT / "data/wacv/temperature_control"; GO = OUT / "GO.json"; OOD_OUT = ROOT / "data/audit/wacv_ood_temperature_claude"
SEEDS = list(range(1000, 1030)); BOUNDS = (.05, 20.); ITERS = 60


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path):
    r = json.loads((path / "result.json").read_text()); f = path / "checkpoint_source.pt"
    assert r["status"] == "complete" and sha(f) == r["checkpoint_sha256"]; return torch.load(f, map_location="cpu", weights_only=False)


@torch.no_grad()
def logits(net, X):
    p = next(net.net.parameters()); return net.net.pi(net.net.body(torch.as_tensor(X, dtype=p.dtype, device=p.device))).double().cpu()


def entropy(z, T):
    lp = torch.log_softmax(z / T, -1); return float(-(lp.exp() * lp).sum(-1).mean())


def solve(z, target):
    """Entropy of softmax(z/T) is non-decreasing in T: bisection on log T within fixed bounds."""
    lo, hi = np.log(BOUNDS[0]), np.log(BOUNDS[1]); e_lo, e_hi = entropy(z, BOUNDS[0]), entropy(z, BOUNDS[1])
    if target <= e_lo: return BOUNDS[0], "lower_bound", e_lo
    if target >= e_hi: return BOUNDS[1], "upper_bound", e_hi
    for _ in range(ITERS):
        mid = (lo + hi) / 2
        if entropy(z, float(np.exp(mid))) < target: lo = mid
        else: hi = mid
    T = float(np.exp((lo + hi) / 2)); return T, "solved", entropy(z, T)


def scaled(net, T):
    m = copy.deepcopy(net)
    with torch.no_grad(): m.net.pi.weight.div_(T); m.net.pi.bias.div_(T)
    return m


def calibrate_seed(hw, s, X, out, species="human"):
    ed, hd = SRC[hw]["expert"] / f"{species}_expert" / f"seed{s}", SRC[hw]["bc"] / f"{species}_bc" / f"seed{s}"
    e, h = load(ed), load(hd)
    for d in (ed, hd): assert hw in json.loads((d / "result.json").read_text())["gpu_name"], (d, hw)
    ze, zh = logits(e, X), logits(h, X); target = entropy(zh, 1.); T, status, got = solve(ze, target)
    m = scaled(e, T); assert torch.allclose(logits(m, X), ze / T, atol=1e-5)
    se, sm = e.net.state_dict(), m.net.state_dict(); assert se.keys() == sm.keys()
    assert all(torch.equal(se[k], sm[k]) for k in se if k not in ("pi.weight", "pi.bias")), "only pi.weight/pi.bias may change"
    d = out / f"{species}_expert_T{hw}" / f"seed{s}"; d.mkdir(parents=True, exist_ok=True); torch.save(m, d / "checkpoint_source.pt")
    rec = dict(status="complete", arm=f"{species}_expert_T{hw}", hardware=hw, seed=s, T=T, solve_status=status, target_entropy=target, achieved_entropy=got,
               expert_entropy_T1=entropy(ze, 1.), views=len(X), checkpoint_sha256=sha(d / "checkpoint_source.pt"),
               source_expert_sha256=sha(ed / "checkpoint_source.pt"), source_bc_sha256=sha(hd / "checkpoint_source.pt"),
               body_and_critic_unchanged=True, objective="mean conditional entropy on 18,000 human source BC views only")
    (d / "result.json").write_text(json.dumps(rec, indent=1) + "\n"); return rec


def views(species):
    with np.load(PRIOR / f"{species}_prior.npz") as a: return a["X"].reshape(-1, 252)     # 6000 rows x 3 rotations


def check():
    torch.manual_seed(0); z = torch.randn(5000, 4).double() * 3
    for target in (.3, .8, 1.2):
        T, st, got = solve(z, target); assert st == "solved" and abs(got - target) < 1e-8, (T, got)
    assert solve(z, 0.)[1] == "lower_bound" and solve(z, np.log(4) + 1)[1] == "upper_bound"
    from ppo_feedforward_reference import FeedforwardPPO
    n = FeedforwardPPO(252, 4); X = np.random.default_rng(0).normal(size=(64, 252)).astype(np.float32)
    assert torch.allclose(logits(scaled(n, 2.5), X), logits(n, X) / 2.5, atol=1e-6)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "technical_check.json").write_text(json.dumps(dict(ok=True, scope="synthetic only", bounds=BOUNDS, iters=ITERS)) + "\n")
    print("check ok")


def bound():
    """Script, BC views and every source checkpoint the calibration will read (bound before any T is computed)."""
    ck = {f"{hw}/{k}/seed{s}": sha(SRC[hw][k] / f"human_{k}" / f"seed{s}" / "checkpoint_source.pt") for hw in SRC for k in ("expert", "bc") for s in SEEDS}
    return dict(script=sha(__file__), human_prior=sha(PRIOR / "human_prior.npz"), checkpoints=ck, bounds=list(BOUNDS), iters=ITERS)


def analyse():
    from analyse_ood_followup_claude import ood, MAIN
    from prior_seed_power_claude import contrast
    HWO = ROOT / "data/audit/wacv_ood_hw_claude"
    bc = {"3090": ood(MAIN, "human_bc", SEEDS), "4090": ood(HWO, "human_bc", SEEDS)}
    tt = {hw: ood(OOD_OUT, f"human_expert_T{hw}", SEEDS) for hw in SRC}
    D = {hw: {s: bc[hw][s] - tt[hw][s] for s in SEEDS} for hw in SRC}; zero = {s: 0. for s in SEEDS}
    avg = {s: (D["3090"][s] + D["4090"][s]) / 2 for s in SEEDS}
    cal = {hw: [json.loads((OUT / "cohort" / f"human_expert_T{hw}" / f"seed{s}" / "result.json").read_text()) for s in SEEDS] for hw in SRC}
    out = dict(ood_mean_per30={f"human_expert_T{hw}": float(np.mean(list(tt[hw].values()))) for hw in SRC},
               T1_primary=contrast(dict(d=avg, z=zero), "d", "z", 3.),
               T2_interaction=contrast(dict(a=D["3090"], b=D["4090"]), "a", "b", 3.),
               within={hw: contrast(dict(d=D[hw], z=zero), "d", "z", 3.) for hw in SRC},
               calibration={hw: dict(T=[r["T"] for r in v], bound_failures=[r["seed"] for r in v if r["solve_status"] != "solved"]) for hw, v in cal.items()},
               reading="T1 positive (t and sign-flip p<.05): gain not explained by expert confidence; TOST +-3/30 (10 pp) p<.05: within margin; otherwise UNRESOLVED")
    (OOD_OUT / "analysis_temperature.json").write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1)[:2500])


if __name__ == "__main__":
    torch.set_num_threads(1); cmd = sys.argv[1]
    if cmd == "check": check()
    elif cmd == "dry":
        r = calibrate_seed(sys.argv[2], 1000, views("human"), OUT / "dry"); print({k: r[k] for k in ("T", "solve_status", "target_entropy", "achieved_entropy", "expert_entropy_T1")})
    else:
        g = json.loads(GO.read_text()); assert g["status"] == "GO" and g["hashes"] == bound()
        if cmd == "calibrate":
            X = views("human"); [calibrate_seed(hw, s, X, OUT / "cohort") for hw in SRC for s in SEEDS]
        elif cmd == "ood":
            import wacv_ood_zero_shot_claude as O
            O.ARMS = tuple(f"human_expert_T{hw}" for hw in SRC); O.F.SEEDS30 = OUT / "cohort"; O.OUT = OOD_OUT; O.run(int(sys.argv[2]), int(sys.argv[3]))
        elif cmd == "analyse": analyse()
