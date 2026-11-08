// Loop diagnosis (stage-8 study, 2026-11-08): why do the bio-plausible agents sometimes loop in new scenarios? Replays the page's step-exact port of the paper's agents
// (stage-7 goal map, stage-3 mushroom-body circuit, PPO) through the 30-trial probe in trained and never-seen rooms, moved and
// familiar goals, and decomposes every decision into habit (prior + KC + place synapses) and goal-map drive.
//   node site/test/loops.mjs [kinds=bl8,bl7,bl3,ppo] [seeds=1000,1001,1002,1003,1004]
import { readFileSync } from "node:fs";
const SITE = new URL("../", import.meta.url).href;
const { Env, Prior, Circuit, stage7, prepare, rng, pick, softmax, bfs, MOVES } = await import(SITE + "js/sim.js");
const D = new URL("data/", SITE), J = f => JSON.parse(readFileSync(new URL(f, D)));
const bin = f => { const b = readFileSync(new URL(f, D)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const E = prepare(J("env.json")), C = J("circuits.json");
const KINDS = (process.argv[2] || "bl7,bl3,ppo").split(","), SEEDS = (process.argv[3] || "1000,1001,1002,1003,1004").split(",").map(Number);
const argmax = v => { let k = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i; return k; };
const mean = v => v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;

function build(kind, prior, seed) {
  const P = () => new Prior(bin(`models/${prior}_${seed}.bin`)), kc = () => bin(`kc/${seed}.bin`);
  if (kind === "ppo") return { prior: new Prior(bin(`learned/ppo_${prior}_${seed}.bin`)) };
  if (kind === "frozen") return { prior: P() };
  if (kind === "bl7" || kind === "bl8") return stage7(P(), kc(), C.configs[kind], C.gate_half0, bin(`learned/${kind}_${prior}_${seed}.bin`));
  const L = new Circuit(P(), kc(), C.configs.none, C.gate_half0), w = new Float32Array(bin(`learned/bl3_${prior}_${seed}.bin`));
  L.Wa.set(w.subarray(0, 4 * 2048)); L.Wv.set(w.subarray(4 * 2048)); return L;
}
// one decision, decomposed: habit = prior head + plastic KC->action + place->action synapses; drive = kappa * goal-map D
function parts(L, x) {
  const { h } = L.prior.body(x), lg0 = Array.from(L.prior.heads(h).lg), kc = [0, 0, 0, 0], wb = [0, 0, 0, 0];
  if (L.code) { const z = L.code(h); for (let t = 0; t < z.i.length; t++) { const j = z.i[t], r = z.v[t]; for (let a = 0; a < 4; a++) kc[a] += L.Wa[a * L.K + j] * r; } }
  if (L.Wb) for (let a = 0; a < 4; a++) for (let s = 0; s < 100; s++) wb[a] += L.Wb[a * 100 + s] * L.c.g * x[150 + s];
  const Dr = L.hippo?.k ? Array.from(L.hippo.drive(x), d => L.hippo.k * d) : [0, 0, 0, 0];
  const habit = lg0.map((v, a) => v + kc[a] + wb[a]), lg = habit.map((v, a) => v + Dr[a]);
  return { habit, drive: Dr, p: softmax(lg) };
}
function run(L, room, goal, seed) {
  const sched = []; for (let rot = 0; rot < 3; rot++) for (const s of E.starts) sched.push({ spawn: s, rotation_index: rot });
  let k = 0; const env = new Env(E, { walkable: room.walkable, goal }, 1, () => sched[Math.min(k++, sched.length - 1)]), u = rng(seed);
  const dist = bfs(room.walkable, goal), eps = []; let x = env.reset()[0], log = [];
  while (env.completed.length < sched.length) {
    const w = env.w[0], pos = w.pos.slice(), f = parts(L, x), a = pick(f.p, u());
    const d0 = dist.get(pos[0] * 10 + pos[1]), opt = a => { const q = [pos[0] + MOVES[a][0], pos[1] + MOVES[a][1]]; const d = dist.get(q[0] * 10 + q[1]); return d != null && d === d0 - 1; };
    const bel = argmax(x.subarray(150, 250)), hb = argmax(f.habit), dv = argmax(f.drive), range = Math.max(...f.drive) - Math.min(...f.drive);
    log.push({ pos, a, belOk: bel === pos[0] * 10 + pos[1], aOpt: opt(a), hOpt: opt(hb), dOpt: opt(dv), agree: hb === dv, flat: range < 1, pH: softmax(f.habit)[a], pD: softmax(f.drive)[a], d0 });
    const o = env.step([a]); x = o.obs[0];
    if (o.done[0]) { const r = env.completed.at(-1); eps.push(summar(r, log)); log = []; }
  }
  return eps;
}
function summar(r, log) {
  const path = r.path, n = path.length - 1, key = p => p[0] * 10 + p[1];
  let bumps = 0, cyc = 0, rv = 0; const seen = new Set([key(path[0])]);
  for (let i = 1; i < path.length; i++) { const k = key(path[i]); bumps += k === key(path[i - 1]); rv += seen.has(k); seen.add(k); if (i >= 2 && k === key(path[i - 2]) && k !== key(path[i - 1])) cyc++; }
  const tail = path.slice(-17), trapped = !r.success && n >= 16 && new Set(tail.map(key)).size <= 4;
  const T = log.slice(-16), m = (k, S = log) => mean(S.map(s => +s[k]));
  return { success: r.success, n, bumps, cyc, rv: rv / Math.max(n, 1), trapped, bel: m("belOk"), aOpt: m("aOpt"), dOpt: m("dOpt"), hOpt: m("hOpt"),
    tBel: m("belOk", T), tAOpt: m("aOpt", T), tDOpt: m("dOpt", T), tHOpt: m("hOpt", T), tAgree: m("agree", T), tFlat: m("flat", T), tPH: mean(T.map(s => s.pH)), tPD: mean(T.map(s => s.pD)),
    tBumps: T.filter((s, i) => i && key(s.pos) === key(T[i - 1].pos)).length, d0: log[0].d0 };
}
const out = {};
for (const seed of SEEDS) for (const prior of ["human_bc", "human_legal_uniform", "cold_ppo"]) for (const kind of KINDS) {
  const pair = seed % 8;
  for (const [stage, goalk] of [["A", "moved"], ["B", "moved"], ["B", "familiar"], ["A", "familiar"]]) {
    const room = E.rooms.find(r => r.configuration === 100 + 2 * pair + (stage === "B")), goal = goalk === "moved" ? room.moved_goal : room.familiar_goal;
    const L = build(kind, prior, seed), eps = run(L, room, goal, 7 + seed);          // fresh agent per condition: episode state only
    (out[`${kind} ${stage === "A" ? "trained " : "never-seen"} ${goalk.padEnd(8)}`] ??= []).push(...eps);
  }
  process.stderr.write(`${seed} ${prior} ${kind}\n`);
}
const pct = v => (100 * v).toFixed(0).padStart(3), f2 = v => v.toFixed(2);
console.log("condition                          succ  | failures: trapped  bumps/ep  2cyc/ep  | at trapped steps: belief-ok  act-opt  drive-opt  habit-opt  agree  drive-flat  p_habit(a)  p_drive(a)  bumps/16");
for (const [k, eps] of Object.entries(out)) {
  const F = eps.filter(e => !e.success), T = F.filter(e => e.trapped);
  console.log(`${k}  ${pct(mean(eps.map(e => +e.success)))}% | n=${String(F.length).padStart(3)}  ${pct(mean(F.map(e => +e.trapped)))}%   ${f2(mean(F.map(e => e.bumps))).padStart(6)}   ${f2(mean(F.map(e => e.cyc))).padStart(6)}  |` +
    `  ${pct(mean(T.map(e => e.tBel)))}%      ${pct(mean(T.map(e => e.tAOpt)))}%     ${pct(mean(T.map(e => e.tDOpt)))}%      ${pct(mean(T.map(e => e.tHOpt)))}%     ${pct(mean(T.map(e => e.tAgree)))}%    ${pct(mean(T.map(e => e.tFlat)))}%       ${f2(mean(T.map(e => e.tPH)))}        ${f2(mean(T.map(e => e.tPD)))}      ${f2(mean(T.map(e => e.tBumps)))}`);
}
console.log("\nall steps (successes and failures): belief-ok / action-optimal / drive-optimal / habit-optimal");
for (const [k, eps] of Object.entries(out)) console.log(`${k}  ${pct(mean(eps.map(e => e.bel)))}%  ${pct(mean(eps.map(e => e.aOpt)))}%  ${pct(mean(eps.map(e => e.dOpt)))}%  ${pct(mean(eps.map(e => e.hOpt)))}%   start distance ${f2(mean(eps.map(e => e.d0)))}`);
