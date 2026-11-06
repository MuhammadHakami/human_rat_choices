"""Registered source feasibility: native decision priors -> visual PPO control.

Real data are recorded moving choices. Images/contact/heading are reconstructed,
not measured animal sensations. Observations use only the causal path prefix.
No biological rule identification or new imitation/RL algorithm is claimed.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import wacv_embodied_long_control as parent
from ppo_feedforward_reference import FeedforwardPPO, FFAgent
from wacv_embodied_belief_source import LandmarkBelief

ROOT = parent.ROOT
PROTOCOL = ROOT / 'docs/WACV_NATIVE_BEHAVIOR_PRIOR_SOURCE_PROTOCOL_20261012.json'
EXPECTED = '686d4d2e49e93c3502c37832991ee3b272081806b9845084c439a79cc826daa6'
BASE = ROOT / 'data/audit/wacv_native_behavior_prior_20261012'
ADDENDUM = ROOT / 'docs/WACV_NATIVE_BEHAVIOR_PRIOR_SOURCE_ADDENDUM1_20261012.json'
ADDENDUM_SHA = 'b711f8f13e6bfdbee9181c718d09075ff9b08cdc350b2357492896207c505341'
MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))


def load():
    assert parent.sha(PROTOCOL) == EXPECTED
    assert parent.sha(ADDENDUM) == ADDENDUM_SHA and json.loads(ADDENDUM.read_text())['amends_sha256'] == EXPECTED
    P = json.loads(PROTOCOL.read_text())
    assert parent.sha(ROOT / P['native_data']) == P['native_data_sha256']
    assert parent.sha(ROOT / P['source_parent']) == P['source_parent_sha256']
    cache, _, L = parent.load()
    return P, cache, L


def hashes():
    P, _, _ = load()
    return dict(protocol=EXPECTED, addendum=ADDENDUM_SHA, native_data=P['native_data_sha256'],
                parent=parent.hashes(), script=parent.sha(__file__),
                parent_ppo=parent.sha(ROOT / 'scripts/recurrent_ppo.py'),
                data_manifest=parent.sha(BASE / 'data_manifest.json'))


def inward_heading(cell):
    dr, dc = 4.5-cell[0], 4.5-cell[1]
    return (0 if dr >= 0 else 1) if abs(dr) >= abs(dc) else (2 if dc >= 0 else 3)


def causal_rows(cache, trial, rotation):
    """Yield observation BEFORE inspecting the next cell or action label."""
    atlas = cache.raw.features[0, :, :, :, 0, :64].reshape(100, 4, 64)
    filt = LandmarkBelief(atlas, .03)
    previous = None
    cfg = trial['configuration']
    for i, cell in enumerate(trial['cells'][:-1]):
        cell = tuple(cell)
        heading = inward_heading(cell) if previous is None else previous
        frame = cache.raw.frame(cfg, cell, MOVES[heading], rotation)
        p = filt.observe(frame[:64], previous, False)
        del p
        last = np.zeros(4, np.float32)
        if previous is not None:
            last[previous] = 1
        raw = np.concatenate((frame, np.zeros(2, np.float32), last))
        posterior = filt.p.ravel().astype(np.float32)
        obs = np.concatenate((raw, posterior, posterior @ parent.COORD))
        assert obs.shape == (252,) and np.isfinite(obs).all()
        # Everything above is a function of the prefix through cell i only.
        yield obs
        nxt = trial['cells'][i+1]
        previous = MOVES.index((nxt[0]-cell[0], nxt[1]-cell[1]))


def goal_distances(walk, goal):
    """Offline expert labels only; no policy runtime geometry/goal handle."""
    from collections import deque
    d = {tuple(goal): 0}
    q = deque([tuple(goal)])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES:
            v = r+dr, c+dc
            if 0 <= v[0] < 10 and 0 <= v[1] < 10 and walk[v] and v not in d:
                d[v] = d[(r, c)]+1
                q.append(v)
    return d


def prepare():
    P, cache, _ = load()
    D = json.loads((ROOT / P['native_data']).read_text())
    BASE.mkdir(parents=True, exist_ok=True)
    records = []
    for si, species in enumerate(('rat', 'human')):
        trials = sorted((t for t in D['trials'] if t['species'] == species and
                         t['subject'] in P['native_subjects'][species] and
                         t['configuration'] in P['native_configurations']),
                        key=lambda t: (t['subject'], t['configuration'], t['trial']))
        raw_moves = sum(len(t['cells'])-1 for t in trials)
        for ti, t in enumerate(trials):
            cells = list(map(tuple, t['cells']))
            stop = cells.index(cache.goal)+1 if cache.goal in cells else len(cells)
            trials[ti] = dict(t, cells=t['cells'][:stop])
        inventory = [(ti, j) for ti, t in enumerate(trials) for j in range(len(t['cells'])-1)]
        selected = np.sort(np.random.default_rng(641293+si).choice(
            len(inventory), P['data']['unique_moves_each_species'], replace=False))
        by_trial = {}
        identities = []
        for idx, flat in enumerate(selected):
            ti, j = inventory[int(flat)]
            by_trial.setdefault(ti, {})[j] = idx
            t = trials[ti]
            identities.append([t['subject'], t['configuration'], t['trial'], j])
        X = np.empty((6000, 3, 252), np.float32)
        Y = np.empty(6000, np.int64)
        E = np.empty(6000, np.int64)
        distances = {cfg: goal_distances(cache.geometry(cfg)[0], cache.goal) for cfg in range(1, 6)}
        for ti, wanted in by_trial.items():
            t = trials[ti]
            for rotation in P['data']['rotations']:
                for j, x in enumerate(causal_rows(cache, t, rotation)):
                    if j not in wanted:
                        continue
                    idx = wanted[j]
                    X[idx, rotation] = x
                    if rotation == 0:
                        a, b = t['cells'][j:j+2]
                        Y[idx] = MOVES.index((b[0]-a[0], b[1]-a[1]))
                        dist = distances[t['configuration']]
                        candidates = [(a[0]+dr, a[1]+dc) for dr, dc in MOVES]
                        good = [k for k, cell in enumerate(candidates) if cell in dist and dist[cell] == dist[tuple(a)]-1]
                        assert good, 'No optimal expert label at recorded current cell'
                        E[idx] = good[0]
        assert np.isfinite(X).all() and len(set(map(tuple, identities))) == 6000
        sham = Y.copy()
        groups = {}
        for i, ident in enumerate(identities):
            groups.setdefault(tuple(ident[:2]), []).append(i)
        rng = np.random.default_rng(641293+100+si)
        for idx in groups.values():
            sham[idx] = Y[rng.permutation(idx)]
            assert np.array_equal(np.bincount(sham[idx], minlength=4), np.bincount(Y[idx], minlength=4))
        f = BASE / f'{species}_prior.npz'
        arrays = dict(X=X, action=Y, sham=sham, expert=E, identity=np.array(identities, np.int64))
        if f.exists():
            with np.load(f) as previous:
                assert all(np.array_equal(previous[k], v) for k,v in arrays.items()), 'Prepared data changed; never overwrite'
        else:
            np.savez_compressed(f, **arrays)
        records.append(dict(species=species, path=str(f.relative_to(ROOT)), sha256=parent.sha(f),
                            raw_moves=raw_moves, eligible_moves=len(inventory), chosen_unique_moves=6000, synthetic_views=18000,
                            action_counts=np.bincount(Y, minlength=4).tolist(),
                            sham_counts=np.bincount(sham, minlength=4).tolist(),
                            subjects=sorted(set(x[0] for x in identities)),
                            configurations=sorted(set(x[1] for x in identities))))
    meta = dict(protocol_sha256=EXPECTED, addendum_sha256=ADDENDUM_SHA, builder_sha256=parent.sha(__file__), datasets=records,
                sensory_provenance='Synthetic causal-prefix render and pose belief; no measured retinal/gaze/contact data.',
                native_moves_read='Only predeclared training individuals, configurations1–5; no6–25 choices.')
    with (BASE / 'data_manifest.json').open('x') as f:
        json.dump(meta, f, indent=2)
        f.write('\n')
    print(json.dumps(records), flush=True)


class DeviceFFAgent(FFAgent):
    """CPU collection/loss API, GPU affine layers; gradients cross device copies.

    Parent GAE, categorical RNG, objective, clipping and optimizer loop untouched.
    No GPU sampling differences or newly stated PPO objective.
    """
    def unroll(self, x, state, done):
        return self.body(x.to(next(self.parameters()).device)), state

    def heads(self, o):
        device = next(self.parameters()).device
        o = o.to(device)
        return torch.distributions.Categorical(logits=self.pi(o).cpu()), self.v(o).squeeze(-1).cpu()


def build(seed, device, L):
    net = parent.build(parent.ARCHS[0], .001, seed, L)
    module = DeviceFFAgent(252, 4)
    module.load_state_dict(net.net.state_dict())
    net.net = module.to(device)
    net.opt = torch.optim.Adam(net.net.parameters(), lr=.001, eps=1e-5)
    return net


def pretrain(net, arm, seed, P, dry=False):
    if arm == 'cold_ppo':
        return dict(unique_native_moves=0, labeled_presentations=0, updates=0)
    species, mode = arm.split('_')
    meta = next(r for r in json.loads((BASE / 'data_manifest.json').read_text())['datasets'] if r['species'] == species)
    f = ROOT / meta['path']
    assert parent.sha(f) == meta['sha256']
    with np.load(f) as a:
        x = torch.as_tensor(a['X'].reshape(-1, 252), device=next(net.net.parameters()).device)
        y = torch.as_tensor(np.repeat(a['action' if mode == 'bc' else mode], 3), device=x.device)
    rng = np.random.default_rng([20261012, 601, seed])
    opt = torch.optim.Adam(net.net.parameters(), lr=P['bc']['learning_rate'], eps=1e-5)
    before_critic = [p.detach().clone() for p in net.net.v.parameters()]
    presentations = updates = 0
    stats = []
    for ep in range(1 if dry else P['bc']['epochs']):
        order = rng.permutation(len(x))
        if dry:
            order = order[:512]
        loss_sum = 0.
        for start in range(0, len(order), P['bc']['batch_size']):
            ii = torch.as_tensor(order[start:start+P['bc']['batch_size']], device=x.device)
            logits = net.net.pi(net.net.body(x[ii]))
            loss = F.cross_entropy(logits, y[ii])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.net.parameters(), P['bc']['gradient_clip'])
            opt.step()
            presentations += len(ii); updates += 1
            loss_sum += float(loss.detach())*len(ii)
        stats.append(loss_sum/len(order))
    assert all(torch.equal(a, b) for a, b in zip(before_critic, net.net.v.parameters()))
    # Adam BC moments are never transferred into the RL arm.
    net.opt = torch.optim.Adam(net.net.parameters(), lr=.001, eps=1e-5)
    assert not net.opt.state
    return dict(unique_native_moves=6000, labeled_presentations=presentations,
                updates=updates, cross_entropy_per_epoch=stats, critic_weights_unchanged=True,
                sensory_frames_measured=False)


def select_arm(net, cache, seed, arm, rep, dry=False):
    selected = copy.deepcopy(net)
    selected.rng = np.random.default_rng([20261012, 602, seed, rep, int(arm == 'adaptive')])
    selected.tg = torch.Generator().manual_seed(int(selected.rng.integers(2**31)))
    before = [p.detach().clone() for p in selected.net.parameters()]
    counters = dict(selected.count)
    schedule = parent.selection_schedule(cache, rep)
    if dry:
        schedule = schedule[:2]
    env = parent.LongBatch(cache, n=1, seed=0, frames=45*len(schedule), schedule=schedule)
    probs = parent.base.run_ppo_select(selected, env, arm == 'adaptive')
    episodes = parent.base.episodes(env, probs, selected._values_log)
    assert len(episodes) == len(schedule)
    if arm == 'frozen':
        assert all(torch.equal(p, q) for p, q in zip(before, selected.net.parameters()))
    delta = {k: selected.count.get(k, 0)-counters.get(k, 0) for k in set(selected.count)|set(counters)}
    return dict(successes=sum(e['success'] for e in episodes), trials=len(episodes),
                decisions=env.frames, episodes=episodes, counters_delta=delta,
                frozen_weights_verified=arm == 'frozen')


def fit(arm, seed, out, dry=False, device='cuda:0'):
    P, cache, L = load()
    assert arm in P['arms'] and (dry or seed in P['seeds'])
    if not dry:
        go = json.loads((BASE / 'GO.json').read_text())
        assert go['status'] == 'GO' and go['hashes'] == hashes()
        assert Path(out).resolve() == Path(go['output']).resolve()
    d = Path(out)/arm/f'seed{seed}'
    d.mkdir(parents=True, exist_ok=True)
    with (d/'started.json').open('x') as f:
        json.dump(dict(hashes=hashes(), arm=arm, seed=seed, dry=dry), f)
    r = dict(arm=arm, seed=seed, dry=dry, hashes=hashes(), status='running')
    begin = time.perf_counter(); env = None
    try:
        net = build(seed, device, L)
        r['bc'] = pretrain(net, arm, seed, P, dry)
        torch.save(net, d/'checkpoint_prior.pt')
        r['prior_sha256'] = parent.sha(d/'checkpoint_prior.pt')
        frames = 1024 if dry else P['rl']['frames_each']
        env = parent.LongBatch(cache, n=8, seed=seed, frames=frames)
        probs = parent.base.run_ppo_train(net, env)
        assert env.frames == net.count['frames'] == frames
        eps = parent.base.episodes(env, probs, net._values_log)
        torch.save(net, d/'checkpoint_source.pt')
        r['checkpoint_sha256'] = parent.sha(d/'checkpoint_source.pt')
        r['train'] = dict(frames=frames, resources=parent.resources(net, [env]),
                          successes=sum(e['success'] for e in env.completed), episodes=len(env.completed))
        sels = {f'{a}_{rep}':select_arm(net, cache, seed, a, rep, dry)
                for a in ('frozen', 'adaptive') for rep in range(3)}
        r['selection'] = {k:{q:v for q,v in s.items() if q != 'episodes'} for k,s in sels.items()}
        r['episodes_sha256'] = parent.gz_write(d/'episodes.json.gz',
            dict(train=eps, selection={k:s['episodes'] for k,s in sels.items()}))
        for a in ('frozen', 'adaptive'):
            r[f'{a}_successes'] = sum(sels[f'{a}_{rep}']['successes'] for rep in range(3))
        r['status'] = 'complete'
    except FloatingPointError as exc:
        r.update(status='failed', error=repr(exc), triage='numerical_nonfinite',
                 training_frames_completed=env.frames if env is not None else 0)
    except Exception as exc:
        r.update(status='invalid', error=repr(exc), traceback=traceback.format_exc(),
                 triage='implementation_or_io_invalidates_cohort')
        (d/'result.json').write_text(json.dumps(r,indent=2)+'\n')
        raise
    r.update(wall_s=time.perf_counter()-begin, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             device=device, gpu_peak_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0)
    (d/'result.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r.get(k) for k in ['arm','seed','dry','status','wall_s']}), flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','dry','fit'])
    ap.add_argument('arm',nargs='?');ap.add_argument('seed',nargs='?',type=int)
    ap.add_argument('--out');ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.device.startswith('cuda') and args.command != 'prepare':
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
        assert '4090' in torch.cuda.get_device_name(0)
        free,_=torch.cuda.mem_get_info();assert free > 3*2**30, 'Preserve desktop memory headroom'
    if args.command=='prepare':prepare()
    else:fit(args.arm,args.seed,args.out,args.command=='dry',args.device)
