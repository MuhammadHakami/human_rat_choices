"""Held-out behaviour-cloning predictability (reviewer FtUu; registered as BC in v4n_camera_ready PROTOCOL.json).
Same network (run.build), recipe (run.pretrain: Adam, batch, clip, 20 epochs, 3 texture views) and prior rows as the
seeds30 priors; only a held-out split is added. Splits: 5 folds grouped by trial (subject, layout, trial), and 3 folds
grouped by subject. Labels: real (recorded), legal_uniform, sham (rat-style shuffle), expert (shortest path).
Metrics on held-out moves (averaged over the 3 views): accuracy and NLL (nats) of the held-out labels.
'choice content' = NLL of the real held-out labels under the legal-trained model minus under the real-trained model.
  CUDA_VISIBLE_DEVICES=1 python scripts/v4n_bc_heldout_claude_20261031.py
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wacv_native_behavior_prior_20261012 as run
import native_prior_controls_runner_20261012 as ctl

OUT = run.ROOT / "data/audit/v4n_camera_ready_20261031/bc_heldout.json"
SEEDS, KINDS = (0, 1, 2), ("real", "legal_uniform", "sham", "expert")


def data(species):
    meta = next(r for r in json.loads((run.BASE / "data_manifest.json").read_text())["datasets"] if r["species"] == species)
    with np.load(run.ROOT / meta["path"]) as a: X, ident, lab = a["X"], a["identity"], dict(real=a["action"], sham=a["sham"], expert=a["expert"])
    _, lab["legal_uniform"] = ctl.labels(f"{species}_legal_uniform", False)
    return X, ident, lab


def folds(ident, by):
    key = ident[:, 0] if by == "subject" else ident[:, 0] * 10**6 + ident[:, 1] * 10**3 + ident[:, 2]
    groups = np.unique(key); k = 3 if by == "subject" else 5
    order = np.random.default_rng(20261031).permutation(groups)
    return [np.isin(key, order[i::k]) for i in range(k)]


def fit(X, y, seed, P, L, dev):
    net = run.build(seed, dev, L); x = torch.as_tensor(X.reshape(-1, 252), device=dev); t = torch.as_tensor(np.repeat(y, 3), device=dev)
    rng = np.random.default_rng([20261012, 601, seed]); opt = torch.optim.Adam(net.net.parameters(), lr=P["bc"]["learning_rate"], eps=1e-5)
    for _ in range(P["bc"]["epochs"]):
        order = rng.permutation(len(x))
        for s in range(0, len(order), P["bc"]["batch_size"]):
            ii = torch.as_tensor(order[s:s + P["bc"]["batch_size"]], device=dev)
            loss = Fn.cross_entropy(net.net.pi(net.net.body(x[ii])), t[ii]); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.net.parameters(), P["bc"]["gradient_clip"]); opt.step()
    return net


@torch.no_grad()
def logp(net, X, dev):
    lp = torch.log_softmax(net.net.pi(net.net.body(torch.as_tensor(X.reshape(-1, 252), device=dev))), -1).cpu().numpy()
    return lp.reshape(len(X), 3, 4)


def main():
    assert torch.cuda.device_count() == 1; dev = "cuda:0"; P, _, L = run.load(); out = {}
    for species in ("rat", "human"):
        X, ident, lab = data(species)
        for by in ("trial", "subject"):
            rows = []
            for seed in SEEDS:
                for k, test in enumerate(folds(ident, by)):
                    lps = {kind: logp(fit(X[~test], lab[kind][~test], seed, P, L, dev), X[test], dev) for kind in KINDS}
                    r = dict(seed=seed, fold=k, n_test=int(test.sum()))
                    for kind, lp in lps.items():
                        y = lab[kind][test]; nll = -np.take_along_axis(lp, np.repeat(y[:, None, None], 3, 1), 2).mean()
                        r[kind] = dict(acc=float((lp.argmax(-1) == y[:, None]).mean()), nll=float(nll))
                    yr = lab["real"][test]
                    pick = lambda lp: -np.take_along_axis(lp, np.repeat(yr[:, None, None], 3, 1), 2).mean()
                    r["real_label_nll_under"] = {kind: float(pick(lp)) for kind, lp in lps.items()}
                    r["choice_content_nats"] = r["real_label_nll_under"]["legal_uniform"] - r["real_label_nll_under"]["real"]
                    r["real_acc_of_legal_model"] = float((lps["legal_uniform"].argmax(-1) == yr[:, None]).mean())
                    rows.append(r); print(species, by, seed, k, json.dumps({q: r[q] for q in ("real", "legal_uniform", "choice_content_nats")}), flush=True)
            mean = lambda f: float(np.mean([f(r) for r in rows]))
            out[f"{species}/{by}"] = dict(rows=rows, summary=dict(
                **{f"{kind}_acc": mean(lambda r, kind=kind: r[kind]["acc"]) for kind in KINDS},
                **{f"{kind}_nll": mean(lambda r, kind=kind: r[kind]["nll"]) for kind in KINDS},
                choice_content_nats=mean(lambda r: r["choice_content_nats"]), real_acc_of_legal_model=mean(lambda r: r["real_acc_of_legal_model"]),
                majority_real=float(np.bincount(lab["real"]).max() / len(lab["real"]))))
    OUT.write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps({k: v["summary"] for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(2); main()
