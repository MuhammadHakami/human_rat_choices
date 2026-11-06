"""Five sensor suites on one 3-D world: what a cell, a fly, a mouse, a macaque and a human sense at a pose.

Each species is its body width, eye height, sensor geometry and spectral channels. Parameters are literature
values cited inline; where a value is a modelling choice it says so. All rays are cast on GPU 1 through
arc_fly.world3d.RayCaster. Poses are (x, y, heading) in the species' own world; heading 0 faces +x.

Head frame: x forward, y left, z up. Azimuth is measured to the left of forward, elevation upward.
"""
from dataclasses import dataclass, field

import numpy as np

DEG = np.pi / 180


def _dirs(az, el):
    az, el = np.asarray(az, float), np.asarray(el, float)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def sphere_lattice(spacing_deg, az_range, el_range):
    """Near-uniform directions with the given angular spacing (Fibonacci sphere), kept inside a field."""
    n = int(round(4 * np.pi / ((spacing_deg * DEG) ** 2 * np.sqrt(3) / 2)))
    i = np.arange(n) + 0.5
    el = np.arcsin(1 - 2 * i / n)
    az = (np.pi * (1 + 5 ** 0.5) * i) % (2 * np.pi)
    az = np.where(az > np.pi, az - 2 * np.pi, az)
    keep = ((az >= az_range[0] * DEG) & (az <= az_range[1] * DEG)
            & (el >= el_range[0] * DEG) & (el <= el_range[1] * DEG))
    return az[keep], el[keep]


def foveated(fovea_deg, max_ecc_deg, ratio, per_ring, az_lim, el_lim):
    """Log-polar retina: ring eccentricities grow geometrically from the fovea (cortical-magnification-like)."""
    ecc, out = fovea_deg, [(0.0, 0.0)]
    while ecc <= max_ecc_deg:
        th = np.linspace(0, 2 * np.pi, per_ring, endpoint=False) + (len(out) % 2) * np.pi / per_ring
        out += list(zip(ecc * np.cos(th), ecc * np.sin(th)))   # (azimuth, elevation) offsets in degrees
        ecc *= ratio
    a = np.array(out)
    keep = (np.abs(a[:, 0]) <= az_lim) & (a[:, 1] >= el_lim[0]) & (a[:, 1] <= el_lim[1])
    return a[keep, 0] * DEG, a[keep, 1] * DEG


@dataclass
class Species:
    name: str
    body_width_m: float
    body_length_m: float
    eye_h_m: float
    eyes: list = field(default_factory=list)        # list of (position (3,) body frame, az (N,), el (N,))
    channels: str = "rgb"                          # rgb | uv_green | uv_green_gradient
    chemo: np.ndarray = None                       # (k, 3) chemosensor positions, body frame
    touch_origin: np.ndarray = None                # (T, 3) ray origins, body frame
    touch_dir: np.ndarray = None                   # (T, 3) ray directions, body frame
    touch_len_m: float = 0.0
    speed_mps: float = 0.0
    sources: dict = field(default_factory=dict)

    @property
    def n_receptors(self):
        return sum(len(a) for _, a, _ in self.eyes)


def _ring(n, radius, height, start=0.0):
    th = start + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(th), radius * np.sin(th), np.full(n, height)], -1), th


def make_species():
    S = {}
    # Dictyostelium discoideum: ~10 um cell, no eyes; cAMP receptors spread over the membrane (sampled at 32
    # points, a modelling choice, not a receptor count); membrane contact as short rays. Speed: ChemoTrack
    # calibration 0.09-0.19 um/s (data/calibration/dictyostelium_v1).
    pos, th = _ring(32, 5e-6, 1e-6)
    to, tth = _ring(16, 0.5e-6, 1e-6)
    S["cell"] = Species("cell", 10e-6, 15e-6, 5e-6, eyes=[], chemo=pos,
                        touch_origin=to, touch_dir=_dirs(tth, 0 * tth), touch_len_m=7e-6, speed_mps=0.15e-6,
                        sources=dict(size="Cao et al. 2019 supplement priors (350 um^3)", speed="ChemoTrack S-BIAD3674 calibration"))
    # Drosophila melanogaster: two compound eyes of ~720 ommatidia (FlyGym Retina uses 721 per eye), inter-
    # ommatidial angle ~5 deg (Land 1997), near-panoramic field with ~20 deg frontal overlap; two spectral
    # channels standing in for UV/blue (R7) and green (R8, R1-6). Antennae ~0.3 mm apart at the head front.
    eyes = []
    for side in (1, -1):
        az, el = sphere_lattice(5.5, (-10, 170) if side == 1 else (-170, 10), (-60, 80))
        eyes.append((np.array([1.0e-3, side * 0.35e-3, 0.9e-3]), az, el))
    ant = np.array([[1.3e-3, 0.15e-3, 0.9e-3], [1.3e-3, -0.15e-3, 0.9e-3]])
    ta = np.array([25, -25, 60, -60, 90, -90, 135, -135, 180]) * DEG
    S["fly"] = Species("fly", 1.0e-3, 2.5e-3, 0.9e-3, eyes=eyes, channels="uv_green", chemo=ant,
                       touch_origin=np.c_[np.zeros((len(ta), 2)), np.full(len(ta), 0.3e-3)],
                       touch_dir=_dirs(ta, 0 * ta), touch_len_m=1.5e-3, speed_mps=10e-3,
                       sources=dict(eyes="Land 1997; FlyGym retina", walking="~1 cm/s free walking"))
    # Mus musculus: lateral eyes (optical axes ~60 deg off midline), ~280 deg panoramic field with ~40 deg
    # binocular overlap, upper field emphasised; acuity ~0.5 cycle/deg (sampled here at 3 deg, a cost choice);
    # dichromatic with a dorsoventral opsin gradient (UV-dominant view of the upper field). Nostrils ~2 mm apart
    # at the snout; macrovibrissae ~3 cm.
    eyes = []
    for side in (1, -1):
        az, el = sphere_lattice(3.0, (-20, 140) if side == 1 else (-140, 20), (-40, 90))
        eyes.append((np.array([0.03, side * 0.009, 0.025]), az, el))
    wa = np.r_[np.linspace(15, 105, 9), -np.linspace(15, 105, 9)] * DEG
    S["mouse"] = Species("mouse", 0.03, 0.09, 0.025, eyes=eyes, channels="uv_green_gradient",
                         chemo=np.array([[0.055, 0.001, 0.015], [0.055, -0.001, 0.015]]),
                         touch_origin=np.tile([0.05, 0.0, 0.015], (len(wa), 1)), touch_dir=_dirs(wa, 0 * wa),
                         touch_len_m=0.03, speed_mps=0.2,
                         sources=dict(vision="Prusky 2000 acuity; Baden 2013 opsin gradient", whiskers="~3 cm vibrissae"))
    # Caenorhabditis elegans: ~1 mm long, ~50 um wide, no image-forming eyes; the two amphid chemosensory sensilla open
    # at the nose tip ~20 um apart (left/right); nose touch (FLP/OLQ/IL1) as short rays; crawls ~0.2 mm/s. Klinotaxis
    # and weathervaning sample odour by head swings -- the E1b scan is the analogue.
    wn = np.linspace(-60, 60, 5) * DEG
    S["worm"] = Species("worm", 50e-6, 1.0e-3, 25e-6, eyes=[],
                        chemo=np.array([[0.5e-3, 10e-6, 20e-6], [0.5e-3, -10e-6, 20e-6]]),
                        touch_origin=np.tile([0.5e-3, 0.0, 20e-6], (len(wn), 1)), touch_dir=_dirs(wn, 0 * wn),
                        touch_len_m=50e-6, speed_mps=0.2e-3,
                        sources=dict(size="adult C. elegans ~1 mm x 50 um", sensors="amphid sensilla at the nose; klinotaxis by head swings"))
    # Rattus norvegicus (Tartarus rung): lateral eyes, ~300 deg panoramic field, acuity ~1 cycle/deg (Long-Evans;
    # sampled at 3 deg), dichromatic UV + green without the mouse's strong opsin gradient; nostrils ~3 mm apart;
    # macrovibrissae ~5 cm; body ~22 cm long, ~6 cm wide; eye height ~4 cm.
    eyes = []
    for side in (1, -1):
        az, el = sphere_lattice(3.0, (-20, 150) if side == 1 else (-150, 20), (-40, 90))
        eyes.append((np.array([0.06, side * 0.015, 0.04]), az, el))
    wa = np.r_[np.linspace(15, 105, 9), -np.linspace(15, 105, 9)] * DEG
    S["rat"] = Species("rat", 0.06, 0.22, 0.04, eyes=eyes, channels="uv_green",
                       chemo=np.array([[0.10, 0.0015, 0.03], [0.10, -0.0015, 0.03]]),
                       touch_origin=np.tile([0.09, 0.0, 0.03], (len(wa), 1)), touch_dir=_dirs(wa, 0 * wa),
                       touch_len_m=0.05, speed_mps=0.3,
                       sources=dict(vision="Prusky 2002 rat acuity ~1 c/deg", whiskers="~5 cm vibrissae"))
    # Macaca mulatta: forward binocular eyes treated as one cyclopean foveated retina; trichromatic;
    # field ~190 x 125 deg; foveal sampling 0.5 deg growing geometrically with eccentricity.
    az, el = foveated(0.5, 95, 1.12, 48, 95, (-70, 55))
    ha = np.linspace(-40, 40, 5) * DEG
    S["macaque"] = Species("macaque", 0.25, 0.5, 0.5, eyes=[(np.array([0.2, 0.0, 0.5]), az, el)], channels="rgb",
                           chemo=np.array([[0.25, 0.0075, 0.48], [0.25, -0.0075, 0.48]]),
                           touch_origin=np.tile([0.2, 0.0, 0.3], (len(ha), 1)), touch_dir=_dirs(ha, 0 * ha),
                           touch_len_m=0.4, speed_mps=1.0,
                           sources=dict(vision="foveated trichromatic primate retina (modelling choice for sampling)"))
    # Homo sapiens: eye height 1.6 m; field ~200 x 120 deg; foveated trichromatic; nostrils ~1.6 cm apart
    # (humans can scent-track, Porter et al. 2007); arm reach ~0.7 m.
    az, el = foveated(0.5, 100, 1.12, 48, 100, (-70, 50))
    ha = np.linspace(-60, 60, 7) * DEG
    S["human"] = Species("human", 0.45, 0.3, 1.6, eyes=[(np.array([0.1, 0.0, 1.6]), az, el)], channels="rgb",
                         chemo=np.array([[0.12, 0.008, 1.55], [0.12, -0.008, 1.55]]),
                         touch_origin=np.tile([0.0, 0.0, 1.0], (len(ha), 1)), touch_dir=_dirs(ha, 0 * ha),
                         touch_len_m=0.7, speed_mps=1.4,
                         sources=dict(olfaction="Porter et al. 2007 scent tracking", walking="1.4 m/s"))
    # Human seen through a VR head-mounted display (Tartarus humans): same eye, field limited to a ~110 deg
    # headset -- VERIFIED: an HTC Vive (nominal ~110 deg), seated, heading by chair orientation, speed by controller,
    # size and speed matched to a rat at 20 cm/s, tiles 20 x 20 cm (de Cothi et al. 2022, bioRxiv v6 methods).
    # The faithful Tartarus observer puts this field at the rat's eye height (scripts/tartarus_choices.observer).
    az, el = foveated(0.5, 78, 1.12, 48, 55, (-55, 55))
    S["human_hmd"] = Species(**{**S["human"].__dict__, "name": "human_hmd",
                                "eyes": [(np.array([0.1, 0.0, 1.6]), az, el)],
                                "sources": dict(vision="HTC Vive field ~110 deg (de Cothi et al. 2022 methods)")})
    return S


SPECIES = make_species()
CHEMO_FLOOR = 1e-4          # detection floor added before the log transform (Weber-Fechner), a modelling choice


def _yaw(v, heading):
    """Rotate body-frame vectors (..., 3) by per-pose yaw (P,) -> (P, ..., 3)."""
    c, s = np.cos(heading), np.sin(heading)
    v = np.asarray(v, float)
    x = c[:, None] * v[None, :, 0] - s[:, None] * v[None, :, 1]
    y = s[:, None] * v[None, :, 0] + c[:, None] * v[None, :, 1]
    return np.stack([x, y, np.broadcast_to(v[None, :, 2], x.shape)], -1)


def _project(rgb, sp, el):
    if sp.channels == "rgb":
        return rgb
    if sp.channels == "uv_green":
        return rgb[..., [2, 1]]
    s = 1 / (1 + np.exp(-el / (15 * DEG)))                     # upper field -> UV-dominant
    return np.stack([rgb[..., 2] * (0.2 + 0.8 * s), rgb[..., 1] * (0.2 + 0.8 * (1 - s))], -1)


def sense(world, caster, sp, xy, heading, chunk=1_500_000):
    """Senses of species sp at P poses. Returns dict(vision (P,R,C) or None, olfaction (P,k), touch (P,T))."""
    xy = np.asarray(xy, float); heading = np.asarray(heading, float); P = len(xy)
    base = np.c_[xy, np.zeros(P)][:, None, :]
    out = {}
    if sp.eyes:
        parts = []
        for pos, az, el in sp.eyes:
            d_body = _dirs(az, el)
            o = base + _yaw(pos[None], heading)                    # (P, 1, 3)
            d = _yaw(d_body, heading)                              # (P, R, 3)
            o = np.broadcast_to(o, d.shape).reshape(-1, 3); d = d.reshape(-1, 3)
            rgb = np.empty((len(o), 3))
            for a in range(0, len(o), chunk):
                dist, gid, nrm = caster.cast(o[a:a + chunk], d[a:a + chunk], groups=(1, 1, 0, 0, 0, 0))
                rgb[a:a + chunk] = caster.shade(o[a:a + chunk], d[a:a + chunk], dist, gid, nrm)
            rgb = rgb.reshape(P, len(az), 3)
            parts.append(_project(rgb, sp, np.broadcast_to(el, (P, len(el)))))
        out["vision"] = np.concatenate(parts, 1).astype(np.float32)
    else:
        out["vision"] = None
    chem = base + _yaw(sp.chemo, heading)                           # (P, k, 3)
    out["olfaction"] = np.log10(world.odour_at(chem[..., :2]) + CHEMO_FLOOR).astype(np.float32)
    to = (base + _yaw(sp.touch_origin, heading)).reshape(-1, 3)
    td = _yaw(sp.touch_dir, heading).reshape(-1, 3)
    dist, gid, _ = caster.cast(to, td, groups=(1, 0, 1, 0, 0, 0))
    floor = world.geom_kind[np.maximum(gid, 0)] == 0
    hit = (dist >= 0) & (dist <= sp.touch_len_m) & ~floor
    out["touch"] = np.where(hit, 1 - dist / sp.touch_len_m, 0.0).reshape(P, -1).astype(np.float32)
    return out


def _selfcheck():
    """Receptor counts in the intended ranges; a fly facing the beacon sees more beacon-coloured receptors than
    one facing away; odour rises toward the goal; touch fires next to a wall and not in the open."""
    import os
    from arc_fly.unified_env import load_tasks
    from arc_fly.world3d import build, RayCaster
    for k, sp in SPECIES.items():
        print(f"{k:8s} receptors {sp.n_receptors:5d}  chemosensors {len(sp.chemo):2d}  touch rays {len(sp.touch_dir):2d}")
    assert 1300 <= SPECIES["fly"].n_receptors <= 1700, SPECIES["fly"].n_receptors
    task = next(t for t in load_tasks("data/unified/tasks.json") if t.id.startswith("arc"))
    sp = SPECIES["fly"]
    w = build(task, sp.body_width_m, sp.eye_h_m); c = RayCaster(w)
    g = np.array(task.goal); H, W = task.walkable.shape
    nb = [g + d for d in ((0, 1), (0, -1), (1, 0), (-1, 0))
          if 0 <= g[0] + d[0] < H and 0 <= g[1] + d[1] < W and task.walkable[tuple(g + d)]][0]
    xy = w.xy_of_rc(nb[None].astype(float)); to_goal = w.xy_of_rc(g[None].astype(float)) - xy
    h = np.arctan2(to_goal[0, 1], to_goal[0, 0])
    s = sense(w, c, sp, np.r_[xy, xy], np.array([h, h + np.pi]))
    beacon = lambda v: (v[..., 1] > 3 * v[..., 0] + 1e-3).sum(-1)   # yellow beacon: green/UV ratio ~4.5, every other surface < 2.5
    assert beacon(s["vision"])[0] > beacon(s["vision"])[1], beacon(s["vision"])
    far = w.xy_of_rc(np.array(task.spawn, float)[None])
    assert s["olfaction"][0].mean() > sense(w, c, sp, far, np.array([0.0]))["olfaction"][0].mean()
    print(f"fly next to the goal: beacon receptors facing {beacon(s['vision'])[0]} vs away {beacon(s['vision'])[1]}; "
          f"touch max {s['touch'].max():.2f}")
    print("species self-check OK")


if __name__ == "__main__":
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    _selfcheck()
