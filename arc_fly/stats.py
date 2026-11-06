"""The intervals and tests the ledger needs, and nothing else.
ci_prop: Wilson interval for a proportion (what the reviewers asked for on 19/21).
boot_ci: bootstrap interval of any statistic over episodes.
paired_levels: exact sign test on per-level differences (the unit of analysis is the level when
the seeds of one arm are not independent, e.g. a deterministic oracle).
cluster_summary: task-clustered bootstrap of per-point correctness -- intervals per arm, against chance, and paired
differences on the same resampled tasks (points within a task are not independent)."""
import numpy as np
from math import comb


def ci_prop(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(x, stat=np.mean, n_boot=2000, seed=0, alpha=0.05):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    b = [stat(x[rng.integers(len(x), size=len(x))]) for _ in range(n_boot)]
    return (float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2))))


def paired_levels(a, b):
    """Exact two-sided sign test on per-level means a_i vs b_i (ties dropped). Returns (wins, losses, p)."""
    d = np.asarray(a, float) - np.asarray(b, float); d = d[d != 0]
    w, n = int((d > 0).sum()), len(d)
    if n == 0: return (0, 0, 1.0)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(w, n - w) + 1)) / 2 ** n)
    return (w, n - w, p)


def cluster_summary(groups, arms, chance=None, pairs=(), n_boot=2000, seed=0):
    """groups: (n,) cluster id per test point; arms: {name: (n,) per-point correctness}; pairs: [(a, b)].
    Returns {arm: {acc, ci95, p_vs_chance}, 'a-b': {diff, ci95, p}} with one-sided bootstrap p (share <= 0)."""
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = [np.flatnonzero(groups == g) for g in ug]
    draws = [np.concatenate([idx[i] for i in rng.integers(len(ug), size=len(ug))]) for _ in range(n_boot)]
    bs = {k: np.array([np.asarray(v, float)[d].mean() for d in draws]) for k, v in arms.items()}
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {k: dict(acc=float(np.mean(v)), ci95=ci(bs[k]), **({} if chance is None else {"p_vs_chance": float((bs[k] <= chance).mean())}))
           for k, v in arms.items()}
    for a, b in pairs:
        d = bs[a] - bs[b]; out[f"{a}-{b}"] = dict(diff=float(np.mean(arms[a]) - np.mean(arms[b])), ci95=ci(d), p=float((d <= 0).mean()))
    return out


def _selfcheck():
    lo, hi = ci_prop(19, 21); assert abs(lo - 0.71) < 0.02 and abs(hi - 0.97) < 0.02, (lo, hi)   # the paper's interval
    lo, hi = boot_ci([1] * 19 + [0] * 2); assert 0.7 < lo < 0.95 and hi == 1.0
    w, l, p = paired_levels([1, 1, 1, 1, 1, 1, .33], [1, 1, 1, 1, 1, 1, 0]); assert (w, l) == (1, 0) and p == 1.0
    w, l, p = paired_levels([1] * 8, [0] * 8); assert (w, l) == (8, 0) and abs(p - 2 / 256) < 1e-9
    g = np.repeat(np.arange(30), 20); a = np.r_[np.ones(300), np.zeros(300)]; r = cluster_summary(g, {"a": a, "b": a * 0}, 0.1, [("a", "b")])
    assert r["a"]["ci95"][0] < .5 < r["a"]["ci95"][1] and r["a-b"]["p"] == 0.0 and r["b"]["p_vs_chance"] == 1.0
    print(f"stats ok: Wilson 19/21 = [{ci_prop(19,21)[0]:.2f}, {ci_prop(19,21)[1]:.2f}]; sign test 8/0 p={p:.4f}")


if __name__ == "__main__":
    _selfcheck()
