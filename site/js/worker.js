// Learning lab, off the main thread: the brain circuit learns a moved goal with its local three-factor rule, and is
// probed (frozen, 30 trials) on the new and the old goal at fixed checkpoints.
import { Prior, Circuit, Latent, prepare } from "./sim.js";
import { learn, probeCircuit } from "./protocol.js";

const get = async (f, kind = "json") => { const r = await fetch(new URL(`../data/${f}`, import.meta.url)); if (!r.ok) throw new Error(`${f}: ${r.status}`); return r[kind](); };
let stop = false;

onmessage = async ({ data: m }) => {
  if (m.stop) { stop = true; return; }
  stop = false;
  try {
    const [E0, C, pw, kw] = await Promise.all([get("env.json"), get("circuits.json"), get(`models/${m.prior}_${m.seed}.bin`, "arrayBuffer"), get(`kc/${m.seed}.bin`, "arrayBuffer")]);
    const E = prepare(E0), room = m.room, unseen = m.unseen, cfg = C.configs[m.gate];
    const L = new Circuit(new Prior(pw), kw, cfg, C.gate_half0), lat = m.gate === "inferred" ? new Latent() : null;
    const checks = [0, E.pre, 8192, 16384, 24576, E.budget]; let next = 0, t0 = performance.now();
    const probeAll = (frames, last) => {
      const score = (rm, g, s) => probeCircuit(E, L, rm, g, lat, s).filter(e => e.success).length * 100 / 30;
      const out = { frames, new: score(room, m.moved, 11 + frames), old: score(room, m.familiar, 12 + frames) };
      if (last) Object.assign(out, { newUnseen: score(unseen, m.movedUnseen, 13), oldUnseen: score(unseen, m.familiar, 14) });
      postMessage({ probe: out });
    };
    probeAll(0); next = 1; let k = 0;
    for (const s of learn(E, L, room, m.familiar, m.moved, { seed: m.runSeed, latent: lat })) {
      if (stop) { postMessage({ stopped: true }); return; }
      if (++k % 3 === 0) {
        const f = s.F[0]; let wa = 0; for (const x of L.Wa) wa += Math.abs(x);
        postMessage({ step: { frames: s.frames, phase: s.phase, ctx: s.ctx, switches: lat?.switches ?? 0, pos: s.env.w.map(w => w.pos), goal: s.phase === "familiar" ? m.familiar : m.moved,
          z: { i: f.z.i, v: f.z.v }, p: f.p, wa, done: s.env.completed.length, success: s.env.completed.filter(e => e.success).length, ms: performance.now() - t0 } });
      }
      while (next < checks.length && s.frames >= checks[next]) { probeAll(checks[next], next === checks.length - 1); next++; }
      if (k % 6 === 0) await new Promise(r => setTimeout(r));          // let stop messages in
    }
    postMessage({ done: true, ms: performance.now() - t0, ctx: L.ctx, Wa: Array.from(L.Wa) });
  } catch (e) { postMessage({ error: String(e) }); }
};
