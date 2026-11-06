"""First-person VISUAL Tartarus prototype (CPU, headless): agents see what the human VR cohort could see, and are NOT
told their position. Environment and camera checks only; no learner, no animal data fitted, no success evaluation.

Source (de Cothi et al. 2022, Curr Biol; pinned text data/audit/wacv_embodied_prior_20261012/decothi2022.txt):
  * HTC Vive VR for humans; body size/speed matched to a rat at 20 cm/s (main text).
  * "a thick virtual fog lined the floor of the maze enabling them to only see adjacent maze modules and the distal
    black wall cue" (STAR e2); one black curtain over most of one surrounding edge, white elsewhere; goal, layouts
    and starts are defined relative to it.
  * modules "visually indistinct"; seamless textures rotated every 10 trials; the goal is invisible until reached.
  * the source grid agents instead got a one-hot position ("not explicitly required to self-localise").
Dynamics, layouts, starts, the fixed 45-decision budget and SPL are Local45Environment's (scripts/
tartarus_closedloop_env45.py), unchanged. Only the observation differs: [image (H*W), previous_contact, reward].

RECONSTRUCTION PARAMETERS (not in the source; exposed for sensitivity analyses, never claimed exact):
  eye_h (module units, 1 module = 0.2 m), hfov, image size, pitch, fog_top / fog_clear / fog_density, curtain
  margin/height, which edge carries the black curtain (cue_edge: not stated in the text), pit depth of removed
  modules, texture amplitude/period, initial heading, luminances. Rat vision is a real room and is NOT reconstructed;
  this matches the human VR information class only, and is not a reproduction of the authors' Unity scene.
Modes: 'allocentric' (actions = the source agents' 4 world moves + wait; the camera faces the last attempted move)
and 'egocentric' (forward / turn left / turn right / wait; a turn consumes a decision as a wait).
  python scripts/tartarus_visual_env.py check
"""
import json, sys, time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tartarus_closedloop_env45 import Local45Environment
from tartarus_closedloop_env import TASKS
from arc_fly.unified_env import Task, load_tasks
from arc_fly.unimaze import NB

HEADINGS = NB                       # (d_row, d_col) per action index 0..3, the contract's action order


@dataclass(frozen=True)
class View:
    width: int = 64
    height: int = 36
    hfov: float = 110.0             # HTC Vive nominal; sensitivity 60-140
    eye_h: float = 0.25             # rat-scale eye in module units (5 cm on 0.2 m modules); sensitivity 0.15-0.5
    pitch: float = 0.0              # degrees, + looks up
    fog_top: float = 0.35           # fog layer height; sensitivity
    fog_clear: float = 1.0          # fog-path length with no attenuation (adjacent module stays visible)
    fog_density: float = 2.0        # per module of fog path beyond fog_clear
    curtain_margin: float = 1.0     # curtain distance outside the grid edge
    curtain_h: float = 4.0
    cue_edge: str = "row0"          # row0 | row9 | col0 | col9 (UNVERIFIED in the source text)
    pit_depth: float = 2.0
    tex_amp: float = 0.08           # floor texture amplitude (0 = flat); rotated every 10 trials
    tex_period: float = 0.25        # module-local; the same texture on every module
    lum: dict = field(default_factory=lambda: dict(floor=0.55, pit=0.05, white=0.85, black=0.05, sky=0.6, fog=0.7, side=0.2))


def _cam(view, eye, heading):
    """Unit ray directions (H, W, 3) in world xyz = (col, up, row) for a camera facing heading (d_row, d_col)."""
    f = np.array([heading[1], 0.0, heading[0]], float); right = np.array([f[2], 0.0, -f[0]]) * -1.0
    up = np.array([0.0, 1.0, 0.0]); p = np.deg2rad(view.pitch); f = f * np.cos(p) + up * np.sin(p)
    tx = np.tan(np.deg2rad(view.hfov) / 2); ty = tx * view.height / view.width
    u = (np.arange(view.width) + .5) / view.width * 2 - 1; v = 1 - (np.arange(view.height) + .5) / view.height * 2
    U, V = np.meshgrid(u * tx, v * ty)
    d = f[None, None] + U[..., None] * right[None, None] + V[..., None] * up[None, None]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def render(view, walkable, cell, heading, tex_rot=0):
    """Luminance image and a per-pixel surface id (0 sky, 1 module top, 2 pit, 3 white curtain, 4 black curtain,
    5 module side face: the visible wall of a gap left by a removed module)."""
    H, W = walkable.shape
    eye = np.array([cell[1] + .5, view.eye_h, cell[0] + .5]); d = _cam(view, eye, heading)
    t_best = np.full(d.shape[:2], np.inf); sid = np.zeros(d.shape[:2], int)
    with np.errstate(divide="ignore", invalid="ignore"):
        # module platforms: boxes [c,c+1] x [-pit,0] x [r,r+1] for walkable cells (slab test, all boxes)
        for r, c in np.argwhere(walkable):
            lo = np.array([c, -view.pit_depth, r], float); hi = lo + np.array([1, view.pit_depth, 1])
            t1, t2 = (lo - eye) / d, (hi - eye) / d
            tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
            hit = (tn <= tf) & (tf > 0) & (tn > 1e-9) & (tn < t_best)
            t_best[hit] = tn[hit]; sid[hit] = 1
        # pit bottom (removed modules and the room around the grid), then curtains at the margin
        tp = (-view.pit_depth - eye[1]) / d[..., 1]; hit = (tp > 0) & (tp < t_best); t_best[hit] = tp[hit]; sid[hit] = 2
        m = view.curtain_margin
        for name, axis, pos in (("row0", 2, -m), ("row9", 2, H + m), ("col0", 0, -m), ("col9", 0, W + m)):
            tc = (pos - eye[axis]) / d[..., axis]; y = eye[1] + tc * d[..., 1]
            hit = (tc > 0) & (tc < t_best) & (y < view.curtain_h)
            t_best[hit] = tc[hit]; sid[hit] = 4 if name == view.cue_edge else 3
    with np.errstate(invalid="ignore"):
        side = (sid == 1) & (eye[1] + t_best * d[..., 1] < -1e-6)    # hit below the top surface = side face
    sid[side] = 5
    L = view.lum; img = np.choose(sid, [L["sky"], L["floor"], L["pit"], L["white"], L["black"], L["side"]]).astype(float)
    on_floor = sid == 1
    if view.tex_amp and on_floor.any():                  # identical seamless texture on every module, rotated
        p = eye + d[on_floor] * t_best[on_floor][:, None]; a = np.deg2rad(37.0 * tex_rot)
        q = np.mod(p[:, [0, 2]], 1.0) - 0.5                          # module-local: every module carries the same texture
        img[on_floor] += view.tex_amp * np.sin(2 * np.pi * (np.cos(a) * q[:, 0] + np.sin(a) * q[:, 1]) / view.tex_period)
    # floor-level fog: attenuate by the path length below fog_top, beyond a clear distance
    tt = np.where(np.isfinite(t_best), t_best, 60.0); dy = d[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_cross = np.where(np.abs(dy) > 1e-12, (view.fog_top - eye[1]) / dy, np.inf)
    if eye[1] < view.fog_top:
        fog_len = np.where((dy > 0) & (t_cross < tt), t_cross, tt)
    else:
        fog_len = np.where((dy < 0) & (t_cross < tt), tt - t_cross, 0.0)
    T = np.exp(-view.fog_density * np.maximum(fog_len - view.fog_clear, 0.0))
    return T * img + (1 - T) * L["fog"], sid, T


class VisualTartarus(Local45Environment):
    """Same dynamics/budget/result as Local45Environment; observation = [image.ravel(), contact, reward]."""

    def __init__(self, task, seed=0, view=View(), mode="allocentric", trial_index=0):
        self.view, self.mode, self.tex_rot = view, mode, trial_index // 10
        g = np.array(task.walkable.shape) / 2 - .5; s = np.array(task.spawn)
        to_c = g - s; k = int(np.argmax(np.abs(to_c)))         # initial heading: toward the arena centre (declared)
        self.heading = HEADINGS[[0, 1][int(to_c[k] < 0)] if k == 0 else [2, 3][int(to_c[k] < 0)]]
        super().__init__(task, seed)

    def _sense(self):
        img, _, _ = render(self.view, self.task.walkable, self.position, self.heading, self.tex_rot)
        reward = float(self.first_arrival is not None and self.first_arrival == len(self.actions) and len(self.actions) > 0)
        return np.r_[img.ravel(), float(self.contact), reward]

    def step(self, action):
        if self.mode == "allocentric":
            if action < 4: self.heading = HEADINGS[action]
            return super().step(action)
        if action == 0:                                          # egocentric forward
            return super().step(HEADINGS.index(self.heading))
        if action in (1, 2):                                     # turn left / right: a decision without translation
            dr, dc = self.heading; self.heading = (-dc, dr) if action == 1 else (dc, -dr)
            return super().step(4)
        return super().step(4)


def check():
    tasks = load_tasks(TASKS); view = View(); rng = np.random.default_rng(0); out = {}
    t = tasks[0]; W = t.walkable
    # 1. input access: the observation is only image + contact + reward (no position, goal, layout id, map)
    e = VisualTartarus(t, 0); o = e.observe()
    assert o.shape == (view.width * view.height + 2,), o.shape
    # 2. goal invisible: moving the goal leaves every pre-arrival observation identical (same actions)
    free = [tuple(p) for p in np.argwhere(W) if tuple(p) not in (t.spawn, t.goal)]
    t2 = Task(t.id + "_g", W, t.visual_occlusion, t.spawn, free[17], t.native_cell_m, t.source, t.split)
    for seed in range(5):
        a1, a2 = VisualTartarus(t, seed), VisualTartarus(t2, seed); acts = np.random.default_rng(seed).integers(4, size=45)
        for x in acts:
            if a1.done or a2.done or a1.position in (t.goal, t2.goal): break
            assert np.array_equal(a1.observe(), a2.observe()); a1.step(int(x)); a2.step(int(x))
    # 3. dynamics, budget and termination identical to Local45Environment (allocentric) on every task
    for tk in tasks[::5]:
        a, b = Local45Environment(tk, 0), VisualTartarus(tk, 0); acts = rng.integers(5, size=45)
        for x in acts:
            if a.done: break
            a.step(int(x)); b.step(int(x)); assert a.position == b.position and a.contact == b.contact
        assert a.result()["path"] == b.result()["path"] and b.budget == 45
    # 4. fog: removing a module 3 cells ahead changes the image ~nothing; removing the adjacent one changes it
    cell, head = (5, 2), (0, 1); base = np.ones((10, 10), bool)
    img0, _, _ = render(view, base, cell, head)
    near = base.copy(); near[5, 3] = False; far = base.copy(); far[5, 6] = False
    dn = float(np.abs(render(view, near, cell, head)[0] - img0).max()); df = float(np.abs(render(view, far, cell, head)[0] - img0).max())
    assert dn > 0.1 and df < 0.01 and dn > 50 * df, (dn, df)   # adjacent visible, 3 ahead fogged out
    # 4b. an adjacent removed module is visible as dark gap surfaces (pit or side face) in the lower image
    _, sid_near, _ = render(view, near, cell, head)
    assert np.isin(sid_near[2 * view.height // 3:], [2, 5]).mean() > 0.1
    # 5. the black curtain is a usable landmark: facing it vs facing away differ in the upper image at every cell
    up_rows = slice(0, view.height // 3); seen = []
    for r, c in np.argwhere(base):
        a = render(view, base, (r, c), (-1, 0))[0][up_rows].mean(); b = render(view, base, (r, c), (1, 0))[0][up_rows].mean()
        seen.append(b - a)
    assert min(seen) > 0.05, min(seen)
    # 6. self-localisation is genuinely required: views alias across cells (open arena, one heading)
    imgs = {}
    for r, c in np.argwhere(base):
        imgs[(r, c)] = np.round(render(view, base, (r, c), (0, 1))[0], 3).tobytes()
    alias = 1 - len(set(imgs.values())) / len(imgs)
    # 7. texture rotation changes appearance only (surface ids unchanged)
    i0, s0, _ = render(view, base, cell, head, 0); i1, s1, _ = render(view, base, cell, head, 1)
    assert np.array_equal(s0, s1) and np.abs(i0 - i1).max() > 0
    # 7b. modules are indistinct: in an open arena, translating the camera by one module along a row (away from the
    #     curtains' angular change) changes only far-field pixels; the near floor (lower image) is identical
    lo_rows = slice(2 * view.height // 3, view.height)
    nr = float(np.abs(render(view, base, (5, 3), (1, 0))[0][lo_rows] - render(view, base, (5, 4), (1, 0))[0][lo_rows]).max())
    assert nr < 1e-9, nr
    # 8. speed
    t0 = time.perf_counter(); n = 200
    for k in range(n): render(view, W, (k % 10, (k * 3) % 10), HEADINGS[k % 4])
    fps = n / (time.perf_counter() - t0)
    out = dict(ok=True, obs_len=int(o.shape[0]), fog_adjacent_change=dn, fog_far_change=df, curtain_contrast_min=min(seen), near_floor_translation_change=nr,
               aliased_view_fraction_open_arena_one_heading=alias, frames_per_s_cpu=round(fps, 1), view=view.__dict__)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    {"check": check}[sys.argv[1]]()
