// Path efficiency audit (stage-8 study, 2026-11-08): node site/test/efficiency.mjs
// Path efficiency of the goal maps in never-seen rooms: decisions / shortest path, bumps, and the 45-move budget vs the longest shortest path.
import { readFileSync } from "node:fs";
const S = new URL("../", import.meta.url).href;
const { Env, Prior, stage7, prepare, rng, pick, softmax, bfs } = await import(S + "js/sim.js");
const D = new URL("data/", S), J = f => JSON.parse(readFileSync(new URL(f, D))), bin = f => { const b = readFileSync(new URL(f, D)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const E = prepare(J("env.json")), C = J("circuits.json"), mean = v => v.reduce((a, b) => a + b, 0) / v.length;
let maxShort = 0; for (const r of E.rooms) for (const g of [r.familiar_goal, r.moved_goal]) for (const s of E.starts) { const d = bfs(r.walkable, s).get(g[0] * 10 + g[1]); if (d == null) console.log("UNREACHABLE", r.configuration, g, s); else maxShort = Math.max(maxShort, d); }
console.log("longest shortest path over all rooms, goals, starts:", maxShort, "(budget 45)");
for (const kind of ["bl8", "bl7"]) for (const [stage, gk] of [["B", "familiar"], ["B", "moved"], ["A", "moved"]]) {
  const ex = [], bumps = [], succ = []; let n = 0;
  for (const L of C.learned.filter(l => l.kind === kind)) {
    const room = E.rooms.find(r => r.configuration === 100 + 2 * L.pair + (stage === "B")), goal = gk === "moved" ? room.moved_goal : room.familiar_goal;
    const A = stage7(new Prior(bin(`models/${L.prior}_${L.seed}.bin`)), bin(`kc/${L.seed}.bin`), C.configs[kind], C.gate_half0, bin(`learned/${kind}_${L.prior}_${L.seed}.bin`));
    const sched = []; for (let rot = 0; rot < 3; rot++) for (const s of E.starts) sched.push({ spawn: s, rotation_index: rot });
    let k = 0; const env = new Env(E, { walkable: room.walkable, goal }, 1, () => sched[Math.min(k++, sched.length - 1)]), u = rng(11 + L.seed); let x = env.reset()[0];
    while (env.completed.length < sched.length) { const f = A.forward(x), a = pick(softmax(f.lg), u()), o = env.step([a]); x = o.obs[0];
      if (o.done[0]) { const r = env.completed.at(-1); n++; succ.push(+r.success); let b = 0; for (let i = 1; i < r.path.length; i++) b += String(r.path[i]) === String(r.path[i - 1]); bumps.push(b);
        if (r.success) ex.push(r.decisions / bfs(room.walkable, r.spawn).get(goal[0] * 10 + goal[1])); } }
  }
  console.log(`${kind} ${stage === "B" ? "never-seen" : "trained   "} ${gk.padEnd(8)} success ${(100 * mean(succ)).toFixed(1)}%  decisions/shortest (successes) mean ${mean(ex).toFixed(2)}, median ${ex.sort((a, b) => a - b)[ex.length >> 1].toFixed(2)}, share at exactly 1.0: ${(100 * ex.filter(v => v === 1).length / ex.length).toFixed(0)}%  bumps/episode ${mean(bumps).toFixed(1)}  (n=${n})`);
}
