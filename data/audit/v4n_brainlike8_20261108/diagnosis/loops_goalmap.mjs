// Stage-7 goal map only: at trapped steps, is the belief spread, is the chosen link already closed, and is the remembered goal
// the drive pulls toward walkable / reachable / the true goal?   node loopdiag2.mjs
import { readFileSync } from "node:fs";
const SITE = new URL("../../../../site/", import.meta.url).href;   // run from the repository root: node data/audit/v4n_brainlike8_20261108/diagnosis/<file>.mjs
const { Env, Prior, Circuit, stage7, prepare, rng, pick, softmax, bfs, MOVES } = await import(SITE + "js/sim.js");
const D = new URL("data/", SITE), J = f => JSON.parse(readFileSync(new URL(f, D)));
const bin = f => { const b = readFileSync(new URL(f, D)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const E = prepare(J("env.json")), C = J("circuits.json");
const argmax = v => { let k = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i; return k; }, mean = v => v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
const build7 = (prior, seed) => stage7(new Prior(bin(`models/${prior}_${seed}.bin`)), bin(`kc/${seed}.bin`), C.configs.bl7, C.gate_half0, bin(`learned/bl7_${prior}_${seed}.bin`));
function run(L, room, goal, seed) {
  const sched = []; for (let rot = 0; rot < 3; rot++) for (const s of E.starts) sched.push({ spawn: s, rotation_index: rot });
  let k = 0; const env = new Env(E, { walkable: room.walkable, goal }, 1, () => sched[Math.min(k++, sched.length - 1)]), u = rng(seed), W = room.walkable, gi = goal[0] * 10 + goal[1];
  let x = env.reset()[0]; const eps = []; let log = [];
  while (env.completed.length < sched.length) {
    const pos = env.w[0].pos.slice(), f = L.forward(x), p = softmax(f.lg), a = pick(p, u()), H = L.hippo, b = x.subarray(150, 250), c = argmax(b), bmax = Math.max(...b);
    const g = argmax(H.Ge), gmax = H.Ge[g], gw = !!W[Math.floor(g / 10)][g % 10], greach = bfs(W, pos).has(g), reach = bfs(W, pos);
    // how many remembered-goal cells (Ge > thr * max) are unwalkable or unreachable from here
    let nG = 0, nBad = 0; for (let s = 0; s < 100; s++) if (H.Ge[s] > H.thr * gmax) { nG++; nBad += !reach.has(s); }
    log.push({ pos, a, bmax, closed: !H.open[c * 4 + a], Da: H.D[a], Dgap: Math.max(...H.D) - H.D[a], gw, greach, gTrue: g === gi, badFrac: nG ? nBad / nG : 0, pD: softmax(H.D.map(d => H.k * d))[a] });
    const o = env.step([a]); x = o.obs[0];
    if (o.done[0]) { const r = env.completed.at(-1), P = r.path, key = q => q[0] * 10 + q[1]; let bumps = 0; for (let i = 1; i < P.length; i++) bumps += key(P[i]) === key(P[i - 1]);
      const trapped = !r.success && P.length > 16 && new Set(P.slice(-17).map(key)).size <= 4, T = log.slice(-16), m = (q, S = log) => mean(S.map(s => +s[q]));
      eps.push({ s: +r.success, trapped, bumps, bmax: m("bmax", T), closed: m("closed", T), Da: m("Da", T), Dgap: m("Dgap", T), gw: m("gw", T), greach: m("greach", T), gTrue: m("gTrue", T), bad: m("badFrac", T), pD: m("pD", T),
        aBmax: m("bmax"), aClosed: m("closed"), aGTrue: m("gTrue") }); log = []; }
  }
  return eps;
}
const out = {};
for (const seed of [1000, 1001, 1002, 1003, 1004]) for (const prior of ["human_bc", "human_legal_uniform", "cold_ppo"]) {
  const pair = seed % 8;
  for (const [stage, gk] of [["A", "moved"], ["A", "familiar"], ["B", "moved"], ["B", "familiar"]]) {
    const room = E.rooms.find(r => r.configuration === 100 + 2 * pair + (stage === "B"));
    (out[`${stage === "A" ? "trained   " : "never-seen"} ${gk.padEnd(8)}`] ??= []).push(...run(build7(prior, seed), room, gk === "moved" ? room.moved_goal : room.familiar_goal, 7 + seed));
  }
}
const pc = v => (100 * v).toFixed(0).padStart(3), f2 = v => v.toFixed(2);
console.log("goal map, at TRAPPED steps (last 16 of a trapped failure):");
console.log("condition            success  trapped-eps | belief max  chosen link closed  D[a]   D gap   p_drive(a) | Ge peak: walkable  reachable  = true goal   remembered cells unreachable");
for (const [k, e] of Object.entries(out)) { const T = e.filter(q => q.trapped);
  console.log(`${k}   ${pc(mean(e.map(q => q.s)))}%    n=${String(T.length).padStart(3)}    |   ${f2(mean(T.map(q => q.bmax)))}        ${pc(mean(T.map(q => q.closed)))}%        ${f2(mean(T.map(q => q.Da))).padStart(6)}  ${f2(mean(T.map(q => q.Dgap))).padStart(5)}    ${f2(mean(T.map(q => q.pD)))}    |          ${pc(mean(T.map(q => q.gw)))}%      ${pc(mean(T.map(q => q.greach)))}%       ${pc(mean(T.map(q => q.gTrue)))}%            ${pc(mean(T.map(q => q.bad)))}%`); }
console.log("\nall steps: belief max / chosen link closed / Ge peak = true goal");
for (const [k, e] of Object.entries(out)) console.log(`${k}   ${f2(mean(e.map(q => q.aBmax)))}   ${pc(mean(e.map(q => q.aClosed)))}%   ${pc(mean(e.map(q => q.aGTrue)))}%`);
