// Leakage audit (descriptive, after the test). Uses the site port (step-exact vs Python).
// 1. distal cue (first 64 pooled features, the belief's only input) vs layout: same cell/heading/texture, different rooms.
// 2. goal moved FAR from every remembered goal in the never-seen room: a learner that reads the goal from its input
//    would still succeed; a goal memory cannot. 3. shuffled goal memory (G centred on a random far cell).
import { readFileSync } from "node:fs";
const S = new URL("../", import.meta.url).pathname;
const { Env, Prior, Circuit, stage7, prepare, bfs, MOVES } = await import(S + "js/sim.js");
const { probe, frozenAgent, circuitAgent } = await import(S + "js/protocol.js");
const J = f => JSON.parse(readFileSync(S + "data/" + f)), bin = f => { const b = readFileSync(S + "data/" + f); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const E = prepare(J("env.json")), C = J("circuits.json");
// 1
let dmax = 0, dfloor = 0; const R0 = E.rooms[0], R1 = E.rooms[1];
for (let s = 0; s < 100; s += 7) for (let h = 0; h < 4; h++) {
  const cell = [Math.floor(s / 10), s % 10], e0 = new Env(E, { walkable: R0.walkable, goal: [0, 0] }, 1, null), e1 = new Env(E, { walkable: R1.walkable, goal: [0, 0] }, 1, null);
  const a = e0.frame(cell, MOVES[h], 0).feat, b = e1.frame(cell, MOVES[h], 0).feat;
  for (let i = 0; i < 64; i++) dmax = Math.max(dmax, Math.abs(a[i] - b[i])); for (let i = 64; i < 144; i++) dfloor = Math.max(dfloor, Math.abs(a[i] - b[i]));
}
console.log(`1. distal cue rows (belief input) max |room A - room B| = ${dmax.toExponential(2)}; floor rows (layout visible) = ${dfloor.toFixed(3)}`);
// 2, 3
const man = d => Math.abs(d[0]) + Math.abs(d[1]);
const res = {}; const add = (k, v) => (res[k] ??= []).push(v);
const KIND = process.argv[2] || "bl7"; for (const L of C.learned.filter(l => l.kind === KIND)) {
  const u = E.rooms.find(r => r.configuration === 101 + 2 * L.pair), mem = [u.familiar_goal, E.rooms.find(r => r.configuration === 100 + 2 * L.pair).moved_goal];
  const reach = bfs(u.walkable, E.starts.find(s => u.walkable[s[0]][s[1]]) ?? E.starts[0]);
  const far = []; for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) if (u.walkable[r][c] && reach.has(r * 10 + c) && !E.starts.some(s => s[0] === r && s[1] === c) && mem.every(m => man([m[0] - r, m[1] - c]) >= 4)) far.push([r, c]);
  const goal = far[(L.seed * 7 + L.prior.length) % far.length];
  const mk7 = Gover => { const P = new Prior(bin(`models/${L.prior}_${L.seed}.bin`)), Lc = stage7(P, bin(`kc/${L.seed}.bin`), C.configs[KIND], C.gate_half0, bin(`learned/${KIND}_${L.prior}_${L.seed}.bin`));
    if (Gover) { Lc.hippo.G = Gover; Lc.hippo.reset(); } return circuitAgent(Lc); };
  const bl3 = () => { const c = new Circuit(new Prior(bin(`models/${L.prior}_${L.seed}.bin`)), bin(`kc/${L.seed}.bin`), C.configs.none, C.gate_half0), w = new Float32Array(bin(`learned/bl3_${L.prior}_${L.seed}.bin`)); c.Wa.set(w.subarray(0, 8192)); c.Wv.set(w.subarray(8192)); return circuitAgent(c); };
  const ppo = () => frozenAgent(new Prior(bin(`learned/ppo_${L.prior}_${L.seed}.bin`)));
  const rate = (act, g) => probe(E, u, g, act, 9 + L.seed).filter(e => e.success).length * 100 / 30;
  // G as tested, but centred on the far cell (same amplitude and width): "if the memory pointed there"
  const G0 = Float64Array.from(new Float32Array(bin(`learned/${KIND}_${L.prior}_${L.seed}.bin`)).subarray(8592)), Gf = new Float64Array(100);
  const { eta_g: eta, sigma } = C.configs[KIND]; for (let x = 0; x < 100; x++) Gf[x] = eta * Math.exp(-((Math.floor(x / 10) - goal[0]) ** 2 + (x % 10 - goal[1]) ** 2) / (2 * sigma ** 2));
  add(KIND + " goal map, moved goal (as tested)", rate(mk7(), u.moved_goal));
  add(KIND + " goal map, goal FAR from memory", rate(mk7(), goal));
  add(KIND + " goal map, memory pointed at the far goal", rate(mk7(Gf), goal));
  add(KIND + " goal map, no memory (G = 0)", rate(mk7(new Float64Array(100)), u.moved_goal));
  add("stage 3, goal FAR", rate(bl3(), goal)); add("PPO, goal FAR", rate(ppo(), goal));
  add("PPO, moved goal", rate(ppo(), u.moved_goal));
}
for (const [k, v] of Object.entries(res)) { const m = v.reduce((a, b) => a + b) / v.length, sd = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
  console.log(`${k.padEnd(44)} ${m.toFixed(1).padStart(5)}% ± ${(2.145 * sd / Math.sqrt(v.length)).toFixed(1)} (n=${v.length})`); }
