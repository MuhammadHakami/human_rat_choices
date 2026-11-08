// Replays the Python reference episodes from tools/golden_many.py in the browser code (sim.js), step by step:
// observations, logits, end of episode and success must match for every shown agent, room and goal.
//   python tools/golden_many.py --cohort ... --learned-root ...   then   node site/test/parity_many.mjs [outputs/golden_many.json]
import { readFileSync } from "node:fs";
import { Env, Prior, Circuit, stage7, prepare } from "../js/sim.js";

const D = new URL("../data/", import.meta.url), J = f => JSON.parse(readFileSync(new URL(f, D)));
const bin = f => { const b = readFileSync(new URL(f, D)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const E = prepare(J("env.json")), C = J("circuits.json"), EPS = JSON.parse(readFileSync(process.argv[2] ?? "outputs/golden_many.json"));

function agent({ kind, prior, seed }) {                              // same construction as site/js/rates.js
  if (kind === "ppo") { const P = new Prior(bin(`learned/ppo_${prior}_${seed}.bin`)); return x => P.heads(P.body(x).h).lg; }
  const P = new Prior(bin(`models/${prior}_${seed}.bin`));
  if (kind === "frozen") return x => P.heads(P.body(x).h).lg;
  if (kind === "bl7") { const L = stage7(P, bin(`kc/${seed}.bin`), C.configs.bl7, C.gate_half0, bin(`learned/bl7_${prior}_${seed}.bin`)); return x => L.forward(x).lg; }
  const L = new Circuit(P, bin(`kc/${seed}.bin`), C.configs.none, C.gate_half0), w = new Float32Array(bin(`learned/bl3_${prior}_${seed}.bin`));
  L.Wa.set(w.subarray(0, 4 * 2048)); L.Wv.set(w.subarray(4 * 2048)); return x => L.forward(x).lg;
}

const worst = {}, bad = [], note = (k, d) => { worst[k] = Math.max(worst[k] ?? 0, d); };
for (const ep of EPS) {
  const act = agent(ep), room = E.rooms.find(r => r.configuration === ep.configuration);
  const env = new Env(E, { walkable: room.walkable, goal: ep.goal }, 1, () => ({ spawn: ep.spawn, rotation_index: ep.rotation_index }));
  let x = env.reset()[0], t = 0, end = null, d = {};
  for (const s of ep.steps) {
    s.obs.forEach((v, i) => { const k = i < 144 ? "view" : i < 150 ? "contact/reward/move" : "belief"; d[k] = Math.max(d[k] ?? 0, Math.abs(x[i] - v)); });
    act(x).forEach((v, i) => { d.logits = Math.max(d.logits ?? 0, Math.abs(v - s.logits[i])); });
    const o = env.step([s.a]); t++; x = o.obs[0]; if (o.done[0]) { end = { t, success: o.r[0] > .5 }; break; }
  }
  const sameEnd = end ? end.t === ep.steps.length && end.success === ep.success : !ep.success;
  for (const [k, v] of Object.entries(d)) note(`${ep.kind} ${k}`, v);
  if (!sameEnd || d.view > 1e-3 || d.belief > 1e-3 || d.logits > 1e-3) bad.push({ kind: ep.kind, seed: ep.seed, cfg: ep.configuration, goal: ep.goal, d, end, py: [ep.steps.length, ep.success] });
}
console.log(EPS.length, "episodes,", EPS.reduce((a, e) => a + e.steps.length, 0), "decisions; worst abs difference:");
for (const [k, v] of Object.entries(worst)) console.log(`  ${k.padEnd(32)} ${v.toExponential(2)}`);
console.log(bad.length ? `MISMATCH in ${bad.length} episodes:\n` + bad.slice(0, 8).map(b => JSON.stringify(b)).join("\n") : "parity_many ok: every episode ends at the same step with the same outcome");
process.exit(bad.length ? 1 : 0);
