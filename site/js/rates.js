// Stable success rates for the cards: every agent in the current room and goal, 5 seeds x 30 frozen trials (the paper's
// probe), off the main thread. A newer request cancels an older one between seeds.
import { Prior, Circuit, prepare } from "./sim.js";
import { probe, frozenAgent, circuitAgent } from "./protocol.js";

const cache = {}, get = (f, kind = "json") => cache[f] ??= fetch(new URL(`../data/${f}`, import.meta.url)).then(r => { if (!r.ok) throw new Error(`${f}: ${r.status}`); return r[kind](); });
let latest = 0, E = null, C = null;

async function agent(key, seed, learnedWa) {                     // same construction as the page (app.js)
  const [kind, base] = key.includes("@") ? key.split("@") : [null, key];
  if (kind === "ppo") return frozenAgent(new Prior(await get(`learned/ppo_${base}_${seed}.bin`, "arrayBuffer")));
  const P = new Prior(await get(`models/${base}_${seed}.bin`, "arrayBuffer"));
  if (kind !== "bl3" && !learnedWa) return frozenAgent(P);       // frozen prior (= the circuit before learning)
  const L = new Circuit(P, await get(`kc/${seed}.bin`, "arrayBuffer"), C.configs[learnedWa?.gate ?? "none"], C.gate_half0);
  if (learnedWa) { L.Wa.set(learnedWa.Wa); L.ctx = learnedWa.ctx; }
  else { const w = new Float32Array(await get(`learned/bl3_${base}_${seed}.bin`, "arrayBuffer")); L.Wa.set(w.subarray(0, 4 * 2048)); L.Wv.set(w.subarray(4 * 2048)); }
  return circuitAgent(L);
}

onmessage = async ({ data: m }) => {
  const id = latest = m.id;
  try {
    E ??= prepare(await get("env.json")); C ??= await get("circuits.json");
    for (const [j, key] of m.models.entries()) {
      const jobs = m.jobs[j], rates = [];                          // [{seed, walk, goal}]: where each seed's agent is measured
      for (const { seed: s, walk, goal } of jobs) {
        if (id !== latest) return;                                 // superseded
        const eps = probe(E, { walkable: walk }, goal, await agent(key, s, m.learned?.[j]), 7 + s);
        rates.push(100 * eps.filter(e => e.success).length / eps.length);
        const mu = rates.reduce((a, b) => a + b) / rates.length, sd = rates.length > 1 ? Math.sqrt(rates.reduce((a, b) => a + (b - mu) ** 2, 0) / (rates.length - 1)) : 0;
        const t = [0, 12.71, 4.30, 3.18, 2.78, 2.57][rates.length - 1] ?? 2;
        postMessage({ id, j, mean: mu, ci: rates.length > 1 ? t * sd / Math.sqrt(rates.length) : null, n: rates.length, trials: 30 * rates.length, done: rates.length === jobs.length });
        await new Promise(r => setTimeout(r));
      }
    }
  } catch (e) { postMessage({ id, error: String(e) }); }
};
