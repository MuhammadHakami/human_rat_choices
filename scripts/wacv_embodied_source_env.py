"""Cached, source-only visual worlds with exact interaction and reward accounting.

The training curriculum and internal geometry are simulator metadata. Policies
receive only the registered 150 observations; no coordinates or context labels.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from arc_fly.unimaze import bfs, NB
from wacv_embodied_source_cache import SourceCache


class SourceBatch:
    def __init__(self, cache, n=1, seed=0, frames=None, schedule=None):
        self.cache, self.n = cache, int(n)
        self.rng = np.random.default_rng(seed)
        self.protocol = cache.protocol
        self.limit = int(frames or self.protocol["source"]["training_frames"])
        if self.limit % self.n:
            raise ValueError("Frame budget must divide environment slots")
        self.schedule = schedule
        if schedule is not None and self.n != 1:
            raise ValueError("Fixed selection schedule uses one slot")
        self.obs_dim, self.n_act = cache.obs_dim, 4
        self.frames = 0
        self.exhausted = False
        self.episode_counter = 0
        self.completed = []
        self.worlds = [None] * self.n
        self._bands = {}
        self._obs = None

    def _phase(self):
        for p in self.protocol["source"]["curriculum"]:
            if self.frames < p["until_total_frames"]:
                return p
        return self.protocol["source"]["curriculum"][-1]

    def _new_world(self):
        if self.schedule is not None:
            if self.episode_counter >= len(self.schedule):
                self.exhausted = True
                return None
            row = self.schedule[self.episode_counter]
            cfg, spawn, ti = row["configuration"], tuple(row["spawn"]), row["rotation_index"]
        else:
            phase = self._phase()
            cfg = int(self.rng.choice(phase["configurations"]))
            walk, native_starts = self.cache.geometry(cfg)
            if phase["native_starts"]:
                spawn = tuple(native_starts[int(self.rng.integers(len(native_starts)))])
            else:
                key = cfg, phase["spawn_distance_min"], phase["spawn_distance_max"]
                if key not in self._bands:
                    d = bfs(walk, self.cache.goal)
                    candidates = np.argwhere((d >= key[1]) & (d <= key[2]) & walk)
                    if not len(candidates):
                        raise ValueError("Empty registered source spawn band")
                    self._bands[key] = candidates
                cells = self._bands[key]
                spawn = tuple(map(int, cells[int(self.rng.integers(len(cells)))]))
            # Nuisance texture independently sampled, never tied to cfg.
            ti = int(self.rng.integers(len(self.cache.rotations)))
        if cfg not in self.cache.index or cfg > 7:
            raise ValueError("Non-source configuration")
        walk, _ = self.cache.geometry(cfg)
        if not walk[spawn] or spawn == self.cache.goal:
            raise ValueError("Invalid source spawn")
        delta = np.array(walk.shape) / 2 - .5 - np.array(spawn)
        k = int(np.argmax(np.abs(delta)))
        heading = NB[[0,1][int(delta[k] < 0)] if k == 0 else [2,3][int(delta[k] < 0)]]
        w = dict(configuration=cfg, spawn=spawn, position=spawn,
                 heading=heading, rotation_index=int(ti), decisions=0,
                 travelled=0, contact=False, last_action=None, reward=0.0,
                 episode=self.episode_counter, path=[spawn], actions=[])
        self.episode_counter += 1
        return w

    def _sense(self, w):
        if w is None:
            return np.zeros(self.obs_dim, np.float32)
        frame = self.cache.frame(w["configuration"], w["position"], w["heading"], w["rotation_index"])
        act = np.zeros(4, np.float32)
        if w["last_action"] is not None:
            act[w["last_action"]] = 1
        return np.concatenate((frame, np.array([w["contact"],w["reward"]],np.float32),act))

    def reset(self):
        if self.frames:
            raise RuntimeError("Do not restart an already-used source batch")
        self.worlds = [self._new_world() for _ in range(self.n)]
        self._obs = np.stack([self._sense(w) for w in self.worlds])
        return self._obs.copy()

    def step(self, actions):
        if self.frames >= self.limit or self.exhausted:
            raise RuntimeError("Source allowance exhausted")
        a = np.asarray(actions)
        if a.shape != (self.n,) or not np.issubdtype(a.dtype,np.integer) or np.any((a < 0)|(a > 3)):
            raise ValueError("Invalid registered policy action")
        next_frames = self.frames + self.n
        boundaries = [p["until_total_frames"] for p in self.protocol["source"]["curriculum"]]
        forced = next_frames == self.limit or (self.schedule is None and next_frames in boundaries)
        if self.schedule is None and any(self.frames < b < next_frames for b in boundaries):
            raise ValueError("Curriculum boundary must align with vectorized slots")
        reward = np.zeros(self.n,np.float32)
        done = np.zeros(self.n,bool)
        next_obs = []
        # Frame counter advances before resampling so new worlds use next phase.
        self.frames = next_frames
        for i,action in enumerate(a):
            w = self.worlds[i]
            if w is None:
                raise RuntimeError("Empty source world")
            action=int(action)
            walk,_ = self.cache.geometry(w["configuration"])
            old = w["position"]
            candidate = tuple(x+y for x,y in zip(old,NB[action]))
            valid = all(0 <= p < size for p,size in zip(candidate,walk.shape)) and walk[candidate]
            w["position"] = candidate if valid else old
            w["contact"] = not bool(valid)
            w["heading"] = NB[action]
            w["last_action"] = action
            w["decisions"] += 1
            w["travelled"] += int(candidate != old and valid)
            w["path"].append(w["position"])
            w["actions"].append(action)
            success = w["position"] == self.cache.goal
            reward[i] = w["reward"] = float(success)
            done[i] = success or w["decisions"] >= 45 or forced
            terminal_obs = self._sense(w)
            if done[i]:
                opt = int(bfs(walk,w["spawn"])[self.cache.goal])
                self.completed.append(dict(episode=w["episode"],slot=i,
                    configuration=w["configuration"],spawn=list(w["spawn"]),
                    rotation_index=w["rotation_index"],decisions=w["decisions"],
                    travelled=w["travelled"],success=bool(success),
                    reward_sum=float(success),
                    spl=float(success)*opt/max(opt,w["travelled"],1),
                    forced_boundary=bool(forced),path=[list(p) for p in w["path"]],
                    actions=list(w["actions"])))
                if self.frames < self.limit:
                    self.worlds[i] = self._new_world()
                    next_obs.append(self._sense(self.worlds[i]))
                else:
                    next_obs.append(terminal_obs)
            else:
                next_obs.append(terminal_obs)
        self._obs=np.stack(next_obs).astype(np.float32)
        return self._obs.copy(),reward,done


def selection_schedule(cache, repeat):
    # Full native start set, ordered by published start label. This is an
    # artificial schedule, not claimed native individual chronological order.
    rows=[]
    for cfg in cache.protocol["source"]["select_configurations"]:
        _,starts=cache.geometry(cfg)
        for spawn in starts:
            rows.append(dict(configuration=cfg,spawn=tuple(spawn),
                             rotation_index=repeat % len(cache.rotations)))
    return rows


def check(cache):
    results=[]
    for n in (1,8):
        env=SourceBatch(cache,n=n,seed=417)
        obs=env.reset()
        rng=np.random.default_rng(925)
        credited=0
        previous=0
        while env.frames < env.limit:
            acts=rng.integers(4,size=n)
            old=[(w["configuration"],w["position"],w["heading"],w["rotation_index"])
                 for w in env.worlds]
            obs,r,done=env.step(acts)
            credited += float(r.sum())
            assert obs.shape==(n,150) and obs.dtype==np.float32 and np.isfinite(obs).all()
            # Independent step/reward calculation from geometry and previous pose.
            for i,(cfg,position,_,_) in enumerate(old):
                walk,_=cache.geometry(cfg)
                q=tuple(x+y for x,y in zip(position,NB[int(acts[i])]))
                inside=all(0<=x<s for x,s in zip(q,walk.shape))
                q=q if inside and walk[q] else position
                assert r[i]==float(q==cache.goal)
            if env.frames in (4096,8192,32768):
                assert done.all()
                if env.frames < env.limit:
                    allowed=env._phase()["configurations"]
                    assert all(w["configuration"] in allowed for w in env.worlds)
            assert env.frames==previous+n
            previous=env.frames
        assert env.frames==32768
        assert sum(x["decisions"] for x in env.completed)==32768
        assert credited==sum(x["reward_sum"] for x in env.completed)
        assert all(len(x["actions"])==x["decisions"] and len(x["path"])==x["decisions"]+1
                   and x["decisions"]<=45 for x in env.completed)
        results.append(dict(slots=n,frames=env.frames,
                            terminal_accounting=True,reward_credited_once=True,
                            episodes=len(env.completed)))
    report=dict(ok=True,checks=results,fits=0,
                purpose="Environment accounting, not learned control performance")
    print(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    cache=SourceCache(sys.argv[1],sys.argv[2])
    check(cache)
