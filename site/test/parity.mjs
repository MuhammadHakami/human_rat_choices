// Checks site/js/sim.js against the Python reference traces in site/data/golden.json (written by tools/build_site_data.py).
//   node site/test/parity.mjs
import { readFileSync } from "node:fs";
import { Env, Prior, Circuit, prepare } from "../js/sim.js";

const D = new URL("../data/", import.meta.url), J = f => JSON.parse(readFileSync(new URL(f, D)));
const bin = f => { const b = readFileSync(new URL(f, D)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const E = prepare(J("env.json")), G = J("golden.json"), C = J("circuits.json");
const room = cfg => E.rooms.find(r => r.configuration === cfg);
let worst = {}; const note = (k, a, b) => { const d = Math.abs(a - b); worst[k] = Math.max(worst[k] ?? 0, d); };

// 1. frozen probe: observations (renderer + 8-bit pooling + localisation filter) and logits, step by step
{
  const P = G.probe, prior = new Prior(bin(`models/${P.prior}_${P.seed}.bin`)), r = room(P.configuration);
  const env = new Env(E, { walkable: r.walkable, goal: r.moved_goal }, 1, () => P.schedule[0]); let x = env.reset()[0];
  for (const s of P.steps) {
    s.obs.forEach((v, i) => note(i < 144 ? "view" : i < 150 ? "contact/reward/move" : "belief", x[i], v));
    const { lg } = prior.heads(prior.body(x).h); s.logits.forEach((v, i) => note("logits", lg[i], v));
    const o = env.step([s.a]); x = o.obs[0]; if (o.done[0]) break;
  }
}
// 2. brain circuit: 64 learning steps of 16 animals from the recorded actions; final plastic synapses
{
  const Q = G.circuit, prior = new Prior(bin(`models/${Q.prior}_${Q.seed}.bin`)), r = room(100);
  const L = new Circuit(prior, bin(`kc/${Q.seed}.bin`), Q.cfg, C.gate_half0); let k = 0;
  const env = new Env(E, { walkable: r.walkable, goal: r.familiar_goal }, 16, () => Q.worlds[k++]);
  let F = env.reset().map(x => L.forward(x)); Q.p0.forEach((v, i) => note("circuit p", F[0].p[i], v));
  for (const a of Q.actions) { const o = env.step(a), Fn = o.obs.map(x => L.forward(x)); L.step(F, a, o.r, o.done, Fn); F = Fn; }
  let s = 0; Q.Wa.forEach((v, i) => { note("Wa", L.Wa[i], v); s += Math.abs(v); }); Q.Wv.forEach((v, i) => note("Wv", L.Wv[i], v));
  worst["|Wa| (scale)"] = s; worst["worlds used"] = k + "/" + Q.worlds.length;
}
// 3. latent-cause gate: active cells in each context
{
  const Q = G.gate, prior = new Prior(bin("models/human_bc_1000.bin")), L = new Circuit(prior, bin("kc/1000.bin"), C.configs.inferred, C.gate_half0);
  const h = prior.body(Float32Array.from(Q.obs)).h;
  for (const c of ["0", "1"]) { L.ctx = +c; const z = L.code(h).i.sort((a, b) => a - b); worst[`gate ctx${c} same cells`] = JSON.stringify(z) === JSON.stringify(Q.active[c]); }
}
console.log(worst);
const tol = { view: 1e-5, "contact/reward/move": 0, belief: 1e-4, logits: 1e-4, "circuit p": 1e-5, Wa: 1e-3, Wv: 1e-3 };
const bad = Object.entries(tol).filter(([k, t]) => !(worst[k] <= t)).concat(Object.entries(worst).filter(([k, v]) => v === false));
if (bad.length) { console.error("PARITY FAIL", bad); process.exit(1); } else console.log("parity ok");
