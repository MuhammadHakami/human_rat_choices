// The paper's protocol on top of sim.js: the 30-trial frozen probe and the two-phase learning schedule
// (2,048 familiar-goal interactions, then 30,720 with the goal moved; 16 animals in parallel).
import { Env, rng, pick, softmax, Latent } from "./sim.js";

// Frozen probe (wacv_continual_factorial_claude.probe): every start x 3 texture rotations, 45 decisions each, act ~ softmax(logits).
export function probe(E, room, goal, act, seed = 1, { onStep, onDone } = {}) {
  const sched = []; for (let rot = 0; rot < 3; rot++) for (const s of E.starts) sched.push({ spawn: s, rotation_index: rot });
  let k = 0; const env = new Env(E, { walkable: room.walkable, goal }, 1, () => sched[Math.min(k++, sched.length - 1)]); const u = rng(seed);
  let x = env.reset()[0];
  while (env.completed.length < sched.length) {
    const out = act(x); const a = pick(out.p, u()); onStep?.(env, out, a);
    const o = env.step([a]); x = o.obs[0];
    if (o.done[0]) { onDone?.(env.completed.at(-1)); if (onDone) x = env.sense(0, true); }    // context may change: re-sense
  }
  return env.completed.slice(0, sched.length);
}

export const frozenAgent = prior => x => { const b = prior.body(x), hd = prior.heads(b.h); return { ...b, ...hd, p: softmax(hd.lg) }; };
export const circuitAgent = L => x => { const f = L.forward(x); return { ...f, p: softmax(f.lg) }; };   // probes act at tau = 1

// Learning (v4n_brainlike run_cfg): yields progress so the page stays responsive; probes are run by the caller.
export function* learn(E, L, room, familiar, moved, { seed = 1, n = E.n_env, pre = E.pre, budget = E.budget, latent = null } = {}) {
  const u = rng(seed), start = () => ({ spawn: E.starts[Math.floor(u() * E.starts.length)], rotation_index: Math.floor(u() * E.rotations.length) });
  for (const [phase, goal, frames] of [["familiar", familiar, pre], ["moved", moved, budget - pre]]) {
    const env = new Env(E, { walkable: room.walkable, goal }, n, start); let F = env.reset().map(x => L.forward(x)); L.ea = null; let t = 0;
    while (t < frames) {
      const a = F.map(f => pick(f.p, u())), c0 = L.ctx, o = env.step(a); t += n;
      let c1 = c0; if (latent) { for (let i = 0; i < n; i++) if (o.done[i]) latent.update(o.r[i] > .5); c1 = latent.c; }
      L.ctx = c0; const Fn = o.obs.map(x => L.forward(x)); L.step(F, a, o.r, o.done, Fn); L.ctx = c1;
      F = c1 !== c0 ? (L.ea = null, o.obs.map(x => L.forward(x))) : Fn;
      if (!L.Wa.every(isFinite)) throw new Error("diverged");
      yield { phase, frames: (phase === "moved" ? pre : 0) + t, env, F, a, r: o.r, done: o.done, ctx: L.ctx };
    }
  }
}

// Probe with the learned circuit; with the gate, the context follows a copy of the latent switch during the probe
// (v4n_context_gating_claude_20261103.probe).
export function probeCircuit(E, L, room, goal, latent, seed) {
  const lat = latent?.clone(), c0 = L.ctx; if (lat) L.ctx = lat.c;
  const eps = probe(E, room, goal, circuitAgent(L), seed, lat ? { onDone: ep => { lat.update(ep.success); L.ctx = lat.c; } } : {});
  L.ctx = c0; return eps;
}
