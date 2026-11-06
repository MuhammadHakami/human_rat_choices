// Time one full learning run of the brain circuit and its four probes (node site/test/speed.mjs [prior] [seed] [gate]).
import { readFileSync } from "node:fs";
import { Prior, Circuit, Latent, prepare } from "../js/sim.js";
import { learn, probeCircuit, probe, frozenAgent } from "../js/protocol.js";
const D = new URL("../data/", import.meta.url), J = f => JSON.parse(readFileSync(new URL(f, D)));
const bin = f => { const b = readFileSync(new URL(f, D)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const [prior = "human_bc", seed = "1000", gate = "none"] = process.argv.slice(2), E = prepare(J("env.json")), C = J("circuits.json");
const pair = +seed % 8, A = E.rooms.find(r => r.configuration === 100 + 2 * pair), B = E.rooms.find(r => r.configuration === 101 + 2 * pair);
const P = new Prior(bin(`models/${prior}_${seed}.bin`)), L = new Circuit(P, bin(`kc/${seed}.bin`), C.configs[gate], C.gate_half0), lat = gate === "inferred" ? new Latent() : null;
const goals = eps => eps.filter(e => e.success).length;
let t0 = performance.now(); console.log("frozen familiar/trained", goals(probe(E, A, A.familiar_goal, frozenAgent(P), 3)), (performance.now() - t0 | 0) + " ms");
t0 = performance.now(); for (const s of learn(E, L, A, A.familiar_goal, A.moved_goal, { seed: +(process.env.RUN_SEED ?? 7), latent: lat })) {}
console.log("learn", (performance.now() - t0 | 0) + " ms", "switches", lat?.switches);
t0 = performance.now();
console.log({ moved_trained: goals(probeCircuit(E, L, A, A.moved_goal, lat, 11)), familiar_trained: goals(probeCircuit(E, L, A, A.familiar_goal, lat, 12)),
  moved_unseen: goals(probeCircuit(E, L, B, B.moved_goal, lat, 13)), familiar_unseen: goals(probeCircuit(E, L, B, B.familiar_goal, lat, 14)) }, (performance.now() - t0 | 0) + " ms");
