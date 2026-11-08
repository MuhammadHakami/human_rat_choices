"""Python reference episodes for the agents the page shows (human, rat, learned PPO, learned stage-3 circuit, stage-7 goal map) in
trained and never-seen rooms with both goals. site/test/parity_many.mjs replays them in the browser code step by step.
  python tools/golden_many.py [--cohort DIR] [--learned-root DIR] [--out outputs/golden_many.json]
CPU only. Episodes use the registered env (M.env_for) and policies sampled from softmax(logits), as in the probe."""
import argparse, json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import build_site_data as BS


def net_for(M, kind, prior, seed, cohort, root):
    import v4n_brainlike_claude_20261103 as B
    M.build = lambda p, s: BS.build(M, p, s, cohort)                      # B.build calls M.build: read the same cohort
    if kind == "bl7":                                                     # stage 7: stage-6 circuit + hippocampal goal map
        import v4n_brainlike7_claude_20261107 as B7
        B.KC = B7.KC7; m = B7.build7(prior, seed, BS.circuits()["bl7"])
    else: m = B.build(prior, seed, BS.circuits()["none"]) if kind == "bl3" else M.build(prior, seed)
    if kind in BS.LEARNED:
        f = Path(root) / BS.LEARNED[kind].format(p=prior, s=seed)
        m.net.load_state_dict(torch.load(f, map_location="cpu", weights_only=False))
        if kind == "bl7": m.net.pi.hippo.G = BS.goal_map(json.loads(f.with_name(f.name.replace(".final.pt", ".json")).read_text()))
    return m.net


def episode(FR, M, net, cfg, goal, start, rot, seed):
    gm = {c: tuple(goal) for c in M.goal_map(FR.MAN)}
    sched = [dict(configuration=cfg, spawn=start, rotation_index=rot)]
    env = M.env_for(FR.MAN, gm, 0, 1, seed, "clean", 45, sched); obs = env.reset(); rng = np.random.default_rng(seed); steps = []
    while not env.exhausted:
        with torch.no_grad(): lg = net.pi(net.body(torch.as_tensor(obs, dtype=torch.float32))).double().numpy()[0]
        p = np.exp(lg - lg.max()); p /= p.sum(); a = int(rng.choice(4, p=p))
        steps.append(dict(obs=np.round(obs[0], 6).tolist(), logits=lg.round(6).tolist(), a=a))
        obs, r, d = env.step(np.array([a]))
        if d[0]: return steps, bool(r[0] > .5)
    return steps, False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort"); ap.add_argument("--learned-root", default=str(BS.ROOT))
    ap.add_argument("--out", default=str(BS.ROOT / "outputs/golden_many.json")); a = ap.parse_args()
    FR, M, _ = BS.imports(); man = json.loads(FR.MAN.read_text()); gmap = M.goal_map(FR.MAN); eps = []
    agents = [("frozen", "human_bc"), ("frozen", "rat_bc"), ("ppo", "human_bc"), ("bl3", "human_bc"), ("bl7", "human_bc")]
    for i, (kind, prior) in enumerate(agents):
        for seed in (1000, 1001, 1002, 1003, 1004):
            net = net_for(M, kind, prior, seed, a.cohort, a.learned_root); pair = seed % 8
            for cfg in (100 + 2 * pair, 101 + 2 * pair):
                for gname in ("familiar", "moved"):
                    goal = list(FR.GOAL) if gname == "familiar" else list(map(int, gmap[cfg]))
                    k = (7 * i + seed + cfg + len(gname)) % len(man["starts"]); rot = (i + seed) % 3
                    steps, ok = episode(FR, M, net, cfg, goal, man["starts"][k], rot, seed)
                    eps.append(dict(kind=kind, prior=prior, seed=seed, configuration=cfg, goal=goal, spawn=man["starts"][k], rotation_index=rot, success=ok, steps=steps))
    Path(a.out).parent.mkdir(exist_ok=True); Path(a.out).write_text(json.dumps(eps, separators=(",", ":")))
    print(len(eps), "episodes ->", a.out, "successes", sum(e["success"] for e in eps))


if __name__ == "__main__":
    main()
