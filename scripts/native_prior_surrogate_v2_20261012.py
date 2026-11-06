"""Behavioural-surrogate and legal-uniform label controls, v2 (Claude; Codex review 2026-10-12T08:56Z).
v1 (scripts/native_prior_surrogate_20261012.py, unchanged) -> v2: native paths restricted to registered train individuals and
layouts 1-5; normalised CDF (last=1, side='right'); registered L-BFGS-B options; registered seeds; legal-uniform arm
(kappa=beta=0); prepare requires a GO binding protocol, parent protocol, data manifest and this script.

A 2-parameter conditional choice model, fitted source-only per species to the SAME 6000 native rows as the prior arrays:
    P(a | row) = softmax over LEGAL moves of  b_persist * [a == previous move] + b_goal * (d(cell) - d(cell + a))
d = shortest-path distance to the goal on the recorded layout (offline labeller geometry, as for the expert labels).
Labels are then SAMPLED at exactly the same rows/coverage, so the arm differs from rat_bc only in the conditional choice
mapping: it tests whether persistence + goal bias + label stochasticity suffice. Not a brain rule; no new inputs; X untouched.

  python scripts/native_prior_surrogate_v2_20261012.py check            # technical checks only (synthetic labels; no real fit)
  python scripts/native_prior_surrogate_v2_20261012.py prepare --go GO   # real fit + labels; refuses without a bound root GO
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import wacv_native_behavior_prior_20261012 as prior

D = ROOT / "data/audit/wacv_native_behavior_prior_20261012"
OUT = D / "surrogate_v2"
PROTOCOL = ROOT / "docs/WACV_NATIVE_PRIOR_SURROGATE_CONTROL_PROTOCOL_20261012.json"
PARENT = ROOT / "docs/WACV_NATIVE_BEHAVIOR_PRIOR_SOURCE_PROTOCOL_20261012.json"
MANIFEST = D / "data_manifest.json"
MOVES = prior.MOVES
SEED_SURROGATE, SEED_UNIFORM = 641793, 642793
SOLVER = dict(maxiter=1000, ftol=2e-12, gtol=1e-9)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def features(species, cache, native):
    """Per row: legal mask (4,), persistence indicator (4,), goal-progress (4,). Built from the native prefix only."""
    z = np.load(D / f"{species}_prior.npz")
    par = json.loads(PARENT.read_text()); subj, cfgs = set(par["native_subjects"][species]), set(par["native_configurations"])
    cells = {(t["subject"], t["configuration"], t["trial"]): [tuple(c) for c in t["cells"]] for t in native["trials"]
             if t["species"] == species and t["subject"] in subj and t["configuration"] in cfgs}   # source rows only
    dist = {cfg: prior.goal_distances(cache.geometry(cfg)[0], cache.goal) for cfg in range(1, 6)}
    n = len(z["action"]); legal = np.zeros((n, 4), bool); pers = np.zeros((n, 4)); prog = np.zeros((n, 4))
    for r, (s, cfg, t, k) in enumerate(z["identity"]):
        path = cells[(s, cfg, t)]; here = path[k]; d = dist[cfg]
        for a, (dr, dc) in enumerate(MOVES):
            nxt = (here[0] + dr, here[1] + dc)
            if nxt in d: legal[r, a] = True; prog[r, a] = d[here] - d[nxt]
        if k > 0:
            prev = MOVES.index((here[0] - path[k - 1][0], here[1] - path[k - 1][1])); pers[r, prev] = 1.0
        assert legal[r, z["action"][r]], "recorded move must be legal"
        assert np.array_equal(pers[r], z["X"][r, 0, 146:150]), "persistence feature must equal the policy's own last-action input"
    return dict(legal=legal, pers=pers, prog=prog, action=z["action"], identity=z["identity"])


def logp(beta, F):
    u = beta[0] * F["pers"] + beta[1] * F["prog"]; u = np.where(F["legal"], u, -np.inf)
    u = u - u.max(1, keepdims=True); return u - np.log(np.exp(u).sum(1, keepdims=True))


def nll(beta, F, y):
    lp = logp(beta, F); p = np.exp(lp); i = np.arange(len(y))
    g_feat = np.stack([F["pers"], F["prog"]], -1)                                  # (n,4,2)
    grad = -(g_feat[i, y] - (p[..., None] * np.where(F["legal"][..., None], g_feat, 0)).sum(1)).sum(0)
    return -lp[i, y].sum(), grad


def fit(F, y):
    r = minimize(nll, np.zeros(2), args=(F, y), jac=True, method="L-BFGS-B", bounds=[(-20, 20)] * 2, options=SOLVER)
    assert r.success, r.message
    return r.x, float(r.fun)


def sample(beta, F, seed):
    p = np.exp(logp(beta, F)); rng = np.random.default_rng(seed)
    c = np.cumsum(p, 1); c /= c[:, -1:]; c[:, -1] = 1.0                           # normalised; u < 1 always lands
    u = rng.random(len(p)); lab = np.array([np.searchsorted(row, x, side="right") for row, x in zip(c, u)])
    assert lab.max() < 4 and (p[np.arange(len(p)), lab] > 0).all() and F["legal"][np.arange(len(p)), lab].all()
    return lab


def check():
    cache = prior.load()[1]; native = json.loads((ROOT / "data/unified/tartarus_trajectories.json").read_text()); rep = {}
    for species in ("rat", "human"):
        F = features(species, cache, native); y = F["action"]
        # 1 analytic gradient vs central finite difference (at the REAL labels: tests code, not a reported fit)
        b0 = np.array([.7, -.4]); _, g = nll(b0, F, y); eps = 1e-6
        fd = np.array([(nll(b0 + eps * e, F, y)[0] - nll(b0 - eps * e, F, y)[0]) / (2 * eps) for e in np.eye(2)])
        # 2 parameter recovery from labels SAMPLED at the same rows with known betas
        rec = []
        for truth in ([0., 0.], [1., .5], [.5, 2.], [-.5, 1.]):
            ys = sample(np.array(truth), F, 7); bh, _ = fit(F, ys); rec.append(dict(truth=truth, recovered=bh.round(3).tolist()))
            assert np.abs(bh - truth).max() < .15, (species, truth, bh)
        # 3 sampled labels legal, deterministic, same rows; uniform betas reproduce legal-uniform choice
        ys = sample(np.array([1., 1.]), F, 3)
        assert F["legal"][np.arange(len(ys)), ys].all() and np.array_equal(ys, sample(np.array([1., 1.]), F, 3))
        ul = sample(np.zeros(2), F, 11); freq = np.bincount(ul, minlength=4)   # test seed, not a registered label seed
        exp_ = (F["legal"] / F["legal"].sum(1, keepdims=True)).sum(0)
        chi = float(((freq - exp_) ** 2 / exp_).sum())
        Fe = dict(F, legal=F["legal"][:1], pers=F["pers"][:1], prog=F["prog"][:1])                    # extreme beta: one-hot rows
        for b in ([20., 20.], [-20., -20.], [20., -20.]):
            for sd in range(200): sample(np.array(b), Fe, sd)
        u = np.exp(logp(np.zeros(2), F)); assert np.allclose(u.sum(1), 1) and np.allclose(u[F["legal"]], np.repeat(1 / F["legal"].sum(1), F["legal"].sum(1)))
        rep[species] = dict(rows=len(y), grad_fd_max_abs_err=float(np.abs(g - fd).max()), recovery=rec,
                            legal_moves_mean=float(F["legal"].sum(1).mean()), step0_rows=int((F["pers"].sum(1) == 0).sum()),
                            legal_uniform_chi2_df3=chi, extreme_beta_cdf_edges="ok", solver=SOLVER)
        assert chi < 16.27, "legal-uniform label marginals off (p<.001)"
        assert rep[species]["grad_fd_max_abs_err"] < 1e-3
        print(species, rep[species], flush=True)
    (D / "surrogate_v2_technical_check.json").write_text(json.dumps(dict(ok=True, scope="technical only: no real-label fit reported, no labels written", **rep), indent=1) + "\n")


def bound():
    return dict(protocol=sha(PROTOCOL), parent_protocol=sha(PARENT), data_manifest=sha(MANIFEST), script=sha(__file__),
                rat_prior=sha(D / "rat_prior.npz"), human_prior=sha(D / "human_prior.npz"))


def prepare(go):
    g = json.loads(Path(go).read_text())
    assert g.get("status") == "GO" and g.get("hashes") == bound(), "Needs a root GO whose hashes equal bound()"
    P = json.loads(PROTOCOL.read_text()); assert P["fit"]["maxiter"] == SOLVER["maxiter"] and P["fit"]["ftol"] == SOLVER["ftol"] and P["fit"]["gtol"] == SOLVER["gtol"]
    cache = prior.load()[1]; native = json.loads((ROOT / "data/unified/tartarus_trajectories.json").read_text())
    OUT.mkdir(exist_ok=True); man = {}
    for si, species in enumerate(("rat", "human")):
        F = features(species, cache, native); beta, f = fit(F, F["action"])
        labels = dict(surrogate=sample(beta, F, SEED_SURROGATE + si), legal_uniform=sample(np.zeros(2), F, SEED_UNIFORM + si))
        out = OUT / f"{species}_labels.npz"; assert not out.exists(), "never overwrite"
        np.savez_compressed(out, identity=F["identity"], kappa_beta=beta, **labels)
        man[species] = dict(path=str(out.relative_to(ROOT)), sha256=sha(out), kappa_persistence=float(beta[0]), beta_goal=float(beta[1]),
                            nll_per_move=f / len(F["action"]), seeds=dict(surrogate=SEED_SURROGATE + si, legal_uniform=SEED_UNIFORM + si),
                            counts={k: np.bincount(v, minlength=4).tolist() for k, v in labels.items()})
    (OUT / "manifest.json").write_text(json.dumps(dict(hashes=bound(), go=str(go), **man), indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check", "prepare"]); ap.add_argument("--go")
    a = ap.parse_args()
    check() if a.cmd == "check" else prepare(a.go)
