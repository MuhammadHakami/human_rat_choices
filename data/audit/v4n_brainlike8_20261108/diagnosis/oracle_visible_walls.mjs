// Ceiling check for the root cause: does the stage-7 goal map stop looping if links it can SEE are closed (visual-cliff avoidance)
// before it bumps into them? Oracle: the true walls in the 3 visible directions (ahead, left, right) at the believed cell.
// Variants: base (stage 7 as tested) | cliff3 (see 3 directions) | cliff4 (all 4) | cliff3-noior (cliff3 with inhibition of return off)
//   node oracle.mjs
import { readFileSync } from "node:fs";
const SITE = new URL("../../../../site/", import.meta.url).href;   // run from the repository root: node data/audit/v4n_brainlike8_20261108/diagnosis/<file>.mjs
const { Env, Prior, Circuit, prepare, rng, pick, softmax, MOVES } = await import(SITE + "js/sim.js");
const D = new URL("data/", SITE), J = f => JSON.parse(readFileSync(new URL(f, D)));
const bin = f => { const b = readFileSync(new URL(f, D)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const E = prepare(J("env.json")), C = J("circuits.json");
const argmax = v => { let k = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i; return k; }, mean = v => v.reduce((a, b) => a + b, 0) / v.length;
// ---- stage-7 goal map (copy of sim.js Hippo) with perceived-wall link depression ----
const NXT = Int32Array.from({ length: 400 }, (_, q) => { const s = q >> 2, [dr, dc] = MOVES[q & 3], r = Math.floor(s / 10) + dr, c = s % 10 + dc; return r >= 0 && r < 10 && c >= 0 && c < 10 ? r * 10 + c : s; });
const OPEN0 = Uint8Array.from(NXT, (n, q) => +(n !== q >> 2));
const kern = (c, s) => Float64Array.from({ length: 100 }, (_, x) => Math.exp(-((Math.floor(x / 10) - Math.floor(c / 10)) ** 2 + (x % 10 - c % 10) ** 2) / (2 * s * s)));
function wave(open, Ge, g) { let v = Float64Array.from(Ge); if (Math.max(...Ge) <= 0) return v;
  for (let it = 0; it < 100; it++) { const nv = new Float64Array(100); let same = true;
    for (let s = 0; s < 100; s++) { let m = 0; for (let a = 0; a < 4; a++) m = Math.max(m, open[s * 4 + a] ? v[NXT[s * 4 + a]] : 0); nv[s] = Math.max(Ge[s], g * m); same &&= nv[s] === v[s]; }
    if (same) break; v = nv; } return v; }
class Hippo2 {
  constructor(cfg, G, mode) { this.k = cfg.kappa; this.sig = cfg.sigma; this.gam = cfg.gamma_h; this.thr = cfg.thr; this.stp = cfg.stp; this.ior = mode.includes("noior") ? 0 : cfg.ior; this.see = mode.startsWith("cliff4") ? 4 : mode.startsWith("cliff3") ? 3 : 0; this.G = Float64Array.from(G); this.prev = null; this.reset(); }
  reset() { this.open = Uint8Array.from(OPEN0); this.Ge = Float64Array.from(this.G); this.v = wave(this.open, this.Ge, this.gam); }
  drive(x, walls, heading) {                                         // walls(c, a): oracle, true link closed; heading: current heading (d_row, d_col)
    if (this.prev && this.prev.every((y, i) => y === x[i])) return this.D;
    const b = x.subarray(150, 250); let c = 0; for (let s = 1; s < 100; s++) if (b[s] > b[c]) c = s; let ch = false;
    if (x[146] + x[147] + x[148] + x[149] === 0) this.reset();
    else { const mx = Math.max(...this.Ge);
      if (this.stp && x[144] > .5) { let a = 0; for (let q = 1; q < 4; q++) if (x[146 + q] > x[146 + a]) a = q; if (this.open[c * 4 + a]) { this.open[c * 4 + a] = 0; ch = true; } }
      if (this.ior && mx > 0 && this.Ge[c] > this.thr * mx) { const K = kern(c, this.sig); this.Ge = this.Ge.map((g, s) => g * (1 - K[s])); ch = true; } }
    if (this.see) {                                                  // visual cliff: links seen closed from the believed cell are depressed before any bump
      const dirs = this.see === 4 ? [0, 1, 2, 3] : [heading, [-heading[1], heading[0]], [heading[1], -heading[0]]].map(d => MOVES.findIndex(m => m[0] === d[0] && m[1] === d[1]));
      for (const a of dirs) if (this.open[c * 4 + a] && walls(c, a)) { this.open[c * 4 + a] = 0; ch = true; } }
    if (ch) this.v = wave(this.open, this.Ge, this.gam);
    const v = this.v, Dd = new Float64Array(4);
    if (Math.max(...v) > 0) { const q = [0, 1, 2, 3].map(a => { let s = 0; for (let y = 0; y < 100; y++) s += b[y] * (this.open[y * 4 + a] ? v[NXT[y * 4 + a]] : v[y]); return s; }), qm = Math.max(...q);
      for (let a = 0; a < 4; a++) Dd[a] = Math.max(Math.log(Math.max(q[a], 1e-12) / qm) / -Math.log(this.gam), -10); }
    this.prev = Float32Array.from(x); return this.D = Dd;
  }
}
function build7(prior, seed, mode) {
  const w = new Float32Array(bin(`learned/bl7_${prior}_${seed}.bin`)), L = new Circuit(new Prior(bin(`models/${prior}_${seed}.bin`)), bin(`kc/${seed}.bin`), C.configs.bl7, C.gate_half0); L.Wb = new Float64Array(400);
  for (let a = 0; a < 4; a++) { L.Wa.set(w.subarray(a * 2148, a * 2148 + 2048), a * 2048); L.Wb.set(w.subarray(a * 2148 + 2048, (a + 1) * 2148), a * 100); }
  L.h2 = new Hippo2(C.configs.bl7, w.subarray(8592), mode); return L;
}
function run(L, room, goal, seed) {
  const sched = []; for (let rot = 0; rot < 3; rot++) for (const s of E.starts) sched.push({ spawn: s, rotation_index: rot });
  let k = 0; const env = new Env(E, { walkable: room.walkable, goal }, 1, () => sched[Math.min(k++, sched.length - 1)]), u = rng(seed), W = room.walkable;
  const walls = (c, a) => { const r = Math.floor(c / 10) + MOVES[a][0], q = c % 10 + MOVES[a][1]; return !(r >= 0 && r < 10 && q >= 0 && q < 10 && W[r][q]); };
  let x = env.reset()[0]; const eps = [];
  while (env.completed.length < sched.length) {
    const { h } = L.prior.body(x), lg = L.prior.heads(h).lg, z = L.code(h);
    for (let t = 0; t < z.i.length; t++) { const j = z.i[t], r = z.v[t]; for (let a = 0; a < 4; a++) lg[a] += L.Wa[a * L.K + j] * r; }
    for (let a = 0; a < 4; a++) for (let s = 0; s < 100; s++) lg[a] += L.Wb[a * 100 + s] * L.c.g * x[150 + s];
    const Dd = L.h2.drive(x, walls, env.w[0].heading); for (let a = 0; a < 4; a++) lg[a] += L.h2.k * Dd[a];
    const a = pick(softmax(lg), u()), o = env.step([a]); x = o.obs[0];
    if (o.done[0]) { const r = env.completed.at(-1), p = r.path; let bumps = 0; for (let i = 1; i < p.length; i++) bumps += p[i][0] === p[i - 1][0] && p[i][1] === p[i - 1][1];
      const tail = p.slice(-17); eps.push({ s: +r.success, bumps, trapped: +(!r.success && p.length > 16 && new Set(tail.map(q => q[0] * 10 + q[1])).size <= 4), n: r.decisions }); }
  }
  return eps;
}
const MODES = ["base", "cliff3", "cliff4", "cliff3-noior"], out = {};
for (const seed of [1000, 1001, 1002, 1003, 1004]) for (const prior of ["human_bc", "human_legal_uniform", "cold_ppo"]) for (const mode of MODES) {
  const pair = seed % 8;
  for (const [stage, gk] of [["A", "moved"], ["A", "familiar"], ["B", "moved"], ["B", "familiar"]]) {
    const room = E.rooms.find(r => r.configuration === 100 + 2 * pair + (stage === "B")), eps = run(build7(prior, seed, mode), room, gk === "moved" ? room.moved_goal : room.familiar_goal, 7 + seed);
    (out[`${mode.padEnd(13)} ${stage === "A" ? "trained   " : "never-seen"} ${gk.padEnd(8)}`] ??= []).push(...eps);
  }
}
console.log("variant        room       goal      success  trapped(of failures)  bumps/episode  decisions/success");
for (const [k, e] of Object.entries(out)) { const F = e.filter(q => !q.s), S = e.filter(q => q.s);
  console.log(`${k}  ${(100 * mean(e.map(q => q.s))).toFixed(1).padStart(5)}%   ${F.length ? (100 * mean(F.map(q => q.trapped))).toFixed(0).padStart(3) : "  -"}%   ${mean(e.map(q => q.bumps)).toFixed(1).padStart(8)}      ${S.length ? mean(S.map(q => q.n)).toFixed(1) : "-"}`); }
