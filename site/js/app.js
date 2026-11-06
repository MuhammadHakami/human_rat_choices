// Project page: live agents, architecture, learning lab, animals and results. Data: site/data (tools/build_site_data.py).
import { Env, Prior, Circuit, prepare, rng, pick, softmax, movedGoal, bfs, MOVE_NAMES } from "./sim.js";
import { probe, frozenAgent, circuitAgent } from "./protocol.js";
import { fit, drawView, drawPooled, drawRoom, drawStrip, drawKC, drawPolicy, PRIOR_INFO, COL } from "./viz.js";
import { bars, lines, scatter, heat, mean, minibars } from "./charts.js";

const REPO = "MuhammadHakami/human_rat_choices";                                       // set to the GitHub owner/repository
const LINKS = { paper: "paper/vision4neuro_camera.pdf", supp: "paper/vision4neuro_camera_supp.pdf", code: `https://github.com/${REPO}`,
  models: `https://github.com/${REPO}/releases/tag/v1.0` };
document.querySelectorAll("[data-link]").forEach(a => { a.href = LINKS[a.dataset.link]; if (a.dataset.link !== "paper" && a.dataset.link !== "supp") a.target = "_blank"; });
document.getElementById("copy-bib").onclick = () => navigator.clipboard?.writeText(document.getElementById("bibtex").textContent);

const get = (f, kind = "json") => fetch(`data/${f}`).then(r => { if (!r.ok) throw new Error(`${f}: ${r.status}`); return r[kind](); });
const $ = id => document.getElementById(id);
const h = (tag, at = {}, ...kids) => { const e = document.createElement(tag); for (const k in at) k.startsWith("on") ? e[k] = at[k] : e.setAttribute(k, at[k]); e.append(...kids); return e; };
const sel = (label, opts, value, onchange) => { const s = h("select", { onchange: () => onchange(s.value) }); for (const [v, t] of opts) s.append(h("option", { value: v }, t)); s.value = value; return h("label", {}, label, s); };
const segment = (host, opts, value, onchange) => { host.innerHTML = ""; for (const [v, t] of opts) { const b = h("button", { class: v === value ? "on" : "", onclick: () => { host.querySelectorAll("button").forEach(x => x.classList.remove("on")); b.classList.add("on"); onchange(v); } }, t); host.append(b); } onchange(value); };

const D = {}; const priors = {}, kcs = {};
async function prior(p, s) { const k = `${p}_${s}`; return priors[k] ??= new Prior(await get(`models/${k}.bin`, "arrayBuffer")); }
async function kc(s) { return kcs[s] ??= await get(`kc/${s}.bin`, "arrayBuffer"); }
const roomName = r => `room ${r.pair + 1}${r.stage}`;
// Agents: the six frozen priors, plus the paper's agents after learning the moved goal (PPO; stage-3 brain circuit).
const LEARN_PRIORS = ["human_bc", "human_legal_uniform", "cold_ppo"];
const INFO = k => { if (!k.includes("@")) return PRIOR_INFO[k]; const [kind, p] = k.split("@"), from = PRIOR_INFO[p].short;
  if (kind === "ppo") return { label: "PPO (backpropagation), learned the moved goal", short: "PPO", col: "#33383f", from };
  if (kind === "bc0") return { label: "mushroom-body circuit, before learning", short: "mushroom body", col: COL.bio, from };
  return { label: "mushroom-body circuit (local plasticity), learned the moved goal", short: "mushroom body", col: COL.bio, from }; };
const MODEL_OPTS = () => D.C.priors.map(p => [p, `${PRIOR_INFO[p].label} (frozen)`]).concat(LEARN_PRIORS.flatMap(p => [[`bl3@${p}`, `mushroom-body circuit after learning, from ${PRIOR_INFO[p].short}`], [`ppo@${p}`, `PPO after learning, from ${PRIOR_INFO[p].short}`], [`bc0@${p}`, `mushroom-body circuit before learning, from ${PRIOR_INFO[p].short}`]]));
const learnedBins = {};
const learnedBin = (k, p, s) => learnedBins[`${k}${p}${s}`] ??= get(`learned/${k}_${p}_${s}.bin`, "arrayBuffer");
// Autoplay episodes: trained and never-seen rooms, moved and familiar goals, several seeds and starts.
const HERO_EPISODES = [
  { seed: 1002, cfg: 104, goal: "moved", start: 1, rot: 0 }, { seed: 1000, cfg: 101, goal: "familiar", start: 0, rot: 1 },
  { seed: 1004, cfg: 108, goal: "moved", start: 3, rot: 2 }, { seed: 1001, cfg: 103, goal: "moved", start: 0, rot: 0 },
  { seed: 1002, cfg: 105, goal: "familiar", start: 4, rot: 1 }, { seed: 1003, cfg: 106, goal: "moved", start: 6, rot: 2 },
  { seed: 1004, cfg: 109, goal: "familiar", start: 8, rot: 0 }, { seed: 1000, cfg: 100, goal: "moved", start: 5, rot: 1 } ];
const LINEUP = ["human_bc", "rat_bc", "bl3@human_bc", "ppo@human_bc"];
const SCENARIOS = {
  learn: { label: "Goal moved, trained room", models: LINEUP, goal: "moved", room: s => D.E.rooms.findIndex(r => r.configuration === 100 + 2 * (s % 8)), start: 1 },
  unseen: { label: "Goal moved, never-seen room", models: LINEUP, goal: "moved", room: s => D.E.rooms.findIndex(r => r.configuration === 101 + 2 * (s % 8)), start: 0 },
  transfer: { label: "Familiar goal, never-seen room", models: LINEUP, goal: "familiar", room: s => D.E.rooms.findIndex(r => r.configuration === 101 + 2 * (s % 8)), start: 0 } };

// ---------------- Arena: agents acting in one room ----------------
class Arena {
  // default seed 1002: one seed with the typical gap (PPO fails from 7 of 10 starts); the page states the 30-seed mean
  constructor(host, { controls = null, autoplay = false, onSelect = null } = {}) {
    this.host = host; this.autoplay = autoplay; this.onSelect = onSelect; this.sel = 0; this.trial = 0; this.learned = null;
    this.s = { scenario: "learn", room: SCENARIOS.learn.room(1002), goal: "moved", start: 1, rot: 0, seed: 1002, models: LINEUP.slice(), greedy: false, speed: 6, belief: true, edit: false };
    if (controls) this.controls(controls);
  }
  room() { return D.E.rooms[this.s.room]; }
  nextEpisode() {                                                   // autoplay: the next of the diverse episodes
    const ep = HERO_EPISODES[this.trial % HERO_EPISODES.length], S = this.s;
    Object.assign(S, { seed: ep.seed, room: D.E.rooms.findIndex(r => r.configuration === ep.cfg), goal: ep.goal, start: ep.start, rot: ep.rot }); this.walk = null;
    const r = this.room(), cap = document.querySelector(".hero-cap");
    if (cap) cap.innerHTML = `episode ${this.trial % HERO_EPISODES.length + 1}/${HERO_EPISODES.length} · seed ${ep.seed} · ${roomName(r)} ${r.stage === "A" ? "(where the learners trained)" : "(never seen by any agent)"} · ` +
      `${ep.goal === "moved" ? "goal moved" : "familiar goal (3,3)"} · start (${D.E.starts[ep.start]}) · ★ goal`;
  }
  renderSlots() {
    if (!this.slots) return; const S = this.s; this.slots.innerHTML = "";
    S.models.forEach((m, j) => this.slots.append(sel(`Agent ${j + 1}`, MODEL_OPTS(), m, v => { S.models[j] = v; this.reset(); })));
    if (S.models.length < 4) this.slots.append(h("button", { class: "btn ghost", onclick: () => { S.models.push("human_legal_uniform"); this.renderSlots(); this.reset(); } }, "+ agent"));
    if (S.models.length > 1) this.slots.append(h("button", { class: "btn ghost", onclick: () => { S.models.pop(); this.renderSlots(); this.reset(); } }, "− agent"));
  }
  controls(c) {
    const S = this.s, re = () => this.reset();
    const rooms = D.E.rooms.map((r, i) => [i, `${roomName(r)}${r.stage === "B" ? " (never used for learning)" : ""}`]);
    const pills = h("span", { class: "pills" }), sync = () => { roomSel.querySelector("select").value = S.room; goalSel.querySelector("select").value = S.goal; startSel.querySelector("select").value = S.start; };
    for (const [k, sc] of Object.entries(SCENARIOS)) { const b = h("button", { class: "pill" + (k === S.scenario ? " on" : ""), style: k === S.scenario ? "background:var(--fg)" : "", onclick: () => {
      S.scenario = k; Object.assign(S, { models: sc.models.slice(), goal: sc.goal, room: sc.room(S.seed), start: sc.start }); this.walk = null; this.clickGoal = null;
      pills.querySelectorAll(".pill").forEach(x => { x.classList.remove("on"); x.style.background = ""; }); b.classList.add("on"); b.style.background = "var(--fg)"; this.renderSlots(); sync(); re(); } }, sc.label); pills.append(b); }
    const roomSel = sel("Room", rooms, S.room, v => { S.room = +v; this.walk = null; re(); }); let seedSel;
    const goalSel = sel("Goal", [["familiar", "familiar (3,3)"], ["moved", "moved"], ["click", "click a cell"]], S.goal, v => { S.goal = v; re(); });
    const startSel = sel("Start", [[-1, "random"]].concat(D.E.starts.map((s, i) => [i, `(${s})`])), S.start, v => { S.start = +v; re(); });
    c.append(pills, h("span", { style: "flex-basis:100%;height:0" }), roomSel,
      goalSel, startSel,
      sel("Floor texture", [0, 1, 2, 3].map(i => [i, `rotation ${i}`]), S.rot, v => { S.rot = +v; re(); }),
      seedSel = sel("Seed", D.C.seeds.map(s => [s, s]), S.seed, v => { S.seed = +v; if (SCENARIOS[S.scenario]) { S.room = SCENARIOS[S.scenario].room(S.seed); this.walk = null; sync(); } re(); }),
      sel("Actions", [["sample", "sampled (as tested)"], ["greedy", "greedy"]], "sample", v => { S.greedy = v === "greedy"; re(); }));
    this.slots = h("span", { class: "seg" }); this.renderSlots();
    const speed = h("input", { type: "range", min: 1, max: 30, value: S.speed, oninput: () => S.speed = +speed.value });
    const editB = h("button", { class: "btn ghost", onclick: () => { S.edit = !S.edit; editB.textContent = S.edit ? "✓ done editing" : "✎ edit walls"; } }, "✎ edit walls");
    const beliefB = h("label", {}, h("input", { type: "checkbox", checked: "", onchange: e => { S.belief = e.target.checked; this.draw(); } }), "show place belief");
    c.append(this.slots, h("label", {}, "Speed", speed), beliefB,
      h("button", { class: "btn", onclick: () => this.playing = !this.playing }, "▶ / ❚❚"),
      h("button", { class: "btn ghost", onclick: () => { this.playing = false; this.tick(); } }, "step"),
      h("button", { class: "btn ghost", onclick: () => { this.trial++; re(); } }, "↺ new trial"),
      h("button", { class: "btn ghost", onclick: () => { const u = Math.random, seed = D.C.seeds[Math.floor(u() * D.C.seeds.length)], pair = seed % 8, unseen = u() < .5;
        Object.assign(S, { scenario: "random", seed, room: D.E.rooms.findIndex(r => r.configuration === 100 + 2 * pair + unseen), goal: u() < .6 ? "moved" : "familiar", start: Math.floor(u() * 10), rot: Math.floor(u() * 3) });
        this.walk = null; this.renderSlots(); sync(); seedSel.querySelector("select").value = seed; re(); } }, "🎲 random episode"), editB,
      h("button", { class: "btn ghost", onclick: () => this.runProbe() }, "Run 30-trial probe"));
  }
  goal() {
    const r = this.room(), w = this.walk;
    if (this.s.goal === "click" && this.clickGoal) return this.clickGoal;
    if (this.s.goal === "moved") return this.walk ? movedGoal(w, D.E.target, D.E.starts, r.familiar_goal) ?? r.moved_goal : r.moved_goal;
    return r.familiar_goal;
  }
  async reset() {
    this.resets = (this.resets || 0) + 1;
    const S = this.s; this.walk ??= this.room().walkable.map(x => x.slice()); const goal = this.goal(), start = D.E.starts[S.start >= 0 ? S.start : (this.trial * 7 + 3) % D.E.starts.length];
    if (!this.walk[start[0]][start[1]] || !this.walk[goal[0]][goal[1]] || !bfs(this.walk, start).has(goal[0] * 10 + goal[1])) { this.status = "start and goal must be open and connected"; }
    const half = D.C.gate_half0, seed = S.seed, mine = m => this.learned && this.learned.prior === m && this.learned.seed === seed;
    this.agents = await Promise.all(S.models.map(async m => {
      const [kind, base] = m.includes("@") ? m.split("@") : [null, m];
      const P = kind === "ppo" ? new Prior(await learnedBin("ppo", base, seed)) : await prior(base, seed);
      const L = new Circuit(P, await kc(seed), D.C.configs[mine(m) ? this.learned.gate : "none"], half);
      if (kind === "bl3") { const w = new Float32Array(await learnedBin("bl3", base, seed)); L.Wa.set(w.subarray(0, 4 * 2048)); L.Wv.set(w.subarray(4 * 2048)); }
      if (mine(m)) { L.Wa.set(this.learned.Wa); L.ctx = this.learned.ctx; }        // the learned synapses, in the final inferred context
      const env = new Env(D.E, { walkable: this.walk, goal }, 1, () => ({ spawn: start, rotation_index: S.rot }));
      return { m, L, env, x: env.reset()[0], u: rng(1 + this.trial * 7919), done: false, out: null, a: null, learned: !!mine(m) };
    }));
    this.agents.forEach(A => A.out = circuitAgent(A.L)(A.x)); this.build(); this.draw(); this.onSelect?.(this.agents[this.sel]);
  }
  build() {
    this.host.innerHTML = "";
    this.cards = this.agents.map((A, j) => {
      const info = INFO(A.m), card = h("div", { class: "agent" + (j === this.sel && !this.autoplay ? " sel" : "") });
      const view = h("canvas", { class: "view" }), room = h("canvas", { class: "room" }), st = h("div", { class: "status" });
      card.append(h("h3", { style: `color:${info.col}` }, A.learned ? "mushroom-body circuit, learned here in your browser" : (this.autoplay ? info.short : info.label), h("span", { class: "mini" }, INFO(A.m).from ? `from ${INFO(A.m).from} · seed ${this.s.seed}` : `seed ${this.s.seed}`)), view, room, st);
      card.onclick = () => { if (this.autoplay) return; this.sel = j; this.cards.forEach((c, k) => c.card.classList.toggle("sel", k === j)); this.onSelect?.(A); };
      room.onclick = e => { if (this.autoplay) return; const b = room.getBoundingClientRect(), c = Math.floor((e.clientX - b.left) / b.width * 10), r = Math.floor((e.clientY - b.top) / b.height * 10);
        if (this.s.edit) { this.walk[r][c] = 1 - this.walk[r][c]; this.reset(); } else if (this.s.goal === "click") { this.clickGoal = [r, c]; this.reset(); } };
      this.host.append(card); return { card, view, room, st };
    });
    const refit = () => { this.cards.forEach(c => { fit(c.view, c.view.clientWidth, c.view.clientWidth * 36 / 64); fit(c.room, c.room.clientWidth, c.room.clientWidth); }); this.draw(); };
    requestAnimationFrame(refit); this.ro?.disconnect(); this.ro = new ResizeObserver(() => requestAnimationFrame(refit)); this.ro.observe(this.host);
  }
  tick() {
    let any = false;
    for (const A of this.agents) {
      if (A.done) continue; any = true;
      const p = A.out.p, a = this.s.greedy ? p.indexOf(Math.max(...p)) : pick(p, A.u()); A.a = a;
      const o = A.env.step([a]); if (o.done[0]) { A.done = true; A.result = A.env.completed.at(-1); } else { A.x = o.obs[0]; A.out = circuitAgent(A.L)(A.x); }
    }
    this.draw(); this.onSelect?.(this.agents[this.sel]);
    if (!any && this.autoplay && !this.pending) {                  // once: the loop keeps ticking while the restart waits
      this.pending = true; setTimeout(() => { this.pending = false; this.trial++; this.nextEpisode(); this.reset(); }, 1800); }
    return any;
  }
  draw() {
    if (!this.cards) return;
    this.agents.forEach((A, j) => {
      const c = this.cards[j], w = A.done ? A.result : A.env.w[0], info = INFO(A.m), E = D.E;
      const fr = A.env.frame(w.pos, w.heading, w.rot); if (c.view.width) drawView(c.view, fr.img, E.view);
      drawRoom(c.room, this.walk, { goal: A.env.goal, alt: this.s.goal !== "familiar" ? this.room().familiar_goal : null, start: w.spawn, path: w.path, pos: w.pos, heading: w.heading, col: info.col,
        belief: this.s.belief && !A.done ? A.x.subarray(150, 250) : null });
      c.st.innerHTML = A.done ? (A.result.success ? `<span class="ok">reached the goal in ${A.result.decisions} moves</span>` : `<span class="bad">time out (45 moves)</span>`) +
        ` · revisits ${revisits(A.result.path)}%` : `move ${w.decisions}<div class="pbar" style="color:${info.col}">${A.out.p.map(p => `<i style="--w:${(100 * p).toFixed(0)}%"></i>`).join("")}</div><div class="pbar-l"><span>S</span><span>N</span><span>E</span><span>W</span></div>`;
    });
  }
  loop() { let acc = 0, last = performance.now();
    const f = t => { acc += (t - last) / 1000 * this.s.speed; last = t; if ((this.playing ?? this.autoplay) && this.agents) while (acc >= 1) { acc -= 1; this.tick(); } else acc = Math.min(acc, 1); requestAnimationFrame(f); };
    requestAnimationFrame(f); }
  async runProbe() {
    const out = [], goal = this.goal(), room = { walkable: this.walk };
    $("play-probe-note").textContent = "running 30 trials per agent (every start × 3 floor textures, sampled actions)…";
    for (const A of this.agents) { await new Promise(r => setTimeout(r, 20)); const eps = probe(D.E, room, goal, circuitAgent(A.L), 1 + this.trial);
      out.push({ label: INFO(A.m).short + (A.learned ? "+L" : ""), col: INFO(A.m).col, m: eps.filter(e => e.success).length * 100 / 30, h: 0, rv: mean(eps.map(e => revisits(e.path))) }); }
    bars($("play-probe"), out, { ylabel: "success (%)", ymax: 100, ref: 100 * 4 / 30, refLabel: "" });
    $("play-probe-note").innerHTML = `Success over 30 trials in ${roomName(this.room())}${this.walk.flat().join() !== this.room().walkable.flat().join() ? " (edited)" : ""}, one seed. ` +
      `Revisits: ${out.map(o => `${o.label} ${o.rv.toFixed(0)}%`).join(", ")}. One seed and one room are noisy. Paper, 30 seeds: mushroom-body circuit +12.4 pp over PPO on the new goal; human choices +9.9 pp over legal moves in unseen rooms (Evidence).`;
  }
}
const revisits = path => { const s = new Set([String(path[0])]); let r = 0; for (const c of path.slice(1)) { const k = String(c); r += s.has(k); s.add(k); } return Math.round(100 * r / Math.max(path.length - 1, 1)); };

// ---------------- Architecture panel ----------------
const ARCH = [
  { id: "in", title: "Input · 252", sub: "what the agent senses", tags: [["sensor", ""]], detail: `<b>Exact input, 252 numbers per decision.</b> 144: the first-person camera (64×36 luminance, 8-bit, mean-pooled to 16×9, scaled to [−1, 1]);
    1: contact (bumped into a gap); 1: reward; 4: last move (one-hot); 100: a place belief over the 10×10 grid from a Bayes filter that compares the view with a fixed atlas of views (it never receives the true position);
    2: the belief's centre. No goal, map, room id or position is given. The view reproduces the human VR set-up of de Cothi et al.: fog over the floor shows only adjacent modules and the distal black curtain.` },
  { id: "bb", title: "Backbone · cortex", sub: "252 → 64 → 64, tanh", tags: [["frozen", "frozen"], ["BC + PPO", ""]], detail: `<b>Backbone: the prior policy network</b> (20.7k weights). Trained by behaviour cloning on one label set
    (human choices, rat choices, legal moves at the same states, shuffled rat choices, or none), then by PPO with equal budgets. Frozen here.
    The six priors differ <i>only</i> in these weights, so any difference between agents is the effect of the choices they learned from.` },
  { id: "neck", title: "Neck · Kenyon cells", sub: "64 → 2,048, sparse, k-WTA", tags: [["fixed random", "fixed"], ["~2% active", ""]], detail: `<b>Neck: a mushroom-body / cerebellar granule-layer expansion.</b> Each of 2,048 cells sums 7 random hidden units
    with fixed random weights (never trained, like a reservoir without recurrence). Global inhibition keeps the top 2.2% active (k-WTA, an APL/Golgi analogue), and the active rates are divisively normalised.
    Sparse, high-dimensional codes separate similar views, so new learning interferes less. With the latent-cause gate, each inferred context uses its own half of the cells.` },
  { id: "head", title: "Heads", sub: "prior π, V + plastic KC→π, KC→V", tags: [["plastic", "plastic"], ["three-factor", ""]], detail: `<b>Heads.</b> The prior's policy and value heads stay as they are. Plastic synapses from the Kenyon cells (4×2,048 to actions, 2,048 to value) start at zero, so the untrained circuit
    acts exactly like the prior. They learn with a <b>three-factor rule</b>: ΔW = η · δ · e, where the eligibility e = γλe + (onehot(a) − π) ⊗ z is pre- and post-synaptic and δ = r + γV(s′) − V(s) is a dopamine-like TD error.
    It adds metaplasticity (synapses that changed a lot become stable), adaptive dopamine (δ divided by its running RMS) and awake replay of the last 16 steps.
    There is no backpropagation, no optimiser and no batch statistics. With 30 seeds this circuit learns the new goal +12.4 pp better than PPO (Results).` },
  { id: "out", title: "Output", sub: "4 moves + value", tags: [["softmax", ""]], detail: `<b>Output:</b> a probability for each of the four moves (south, north, east, west) and a value estimate. During learning, moves are sampled at temperature τ ≈ 3.4 (exploration);
    probes and this page sample at τ = 1, as in the paper's tests.` },
];
function architecture() {
  const host = $("arch"), cv = {};
  for (const b of ARCH) {
    const el = h("div", { class: "block", onclick: () => { host.querySelectorAll(".block").forEach(x => x.classList.remove("on")); el.classList.add("on"); $("arch-detail").innerHTML = b.detail; } },
      h("h4", {}, b.title), h("div", { class: "sub" }, b.sub));
    const c = (k, w, hh, cap) => { const x = h("canvas", { width: w, height: hh }); cv[k] = x; el.append(x); if (cap) el.append(h("div", { class: "mini" }, cap)); };
    if (b.id === "in") { c("view", 256, 144, "camera 64×36"); c("pool", 256, 144, "pooled 16×9 = 144"); c("belief", 120, 120, "place belief 10×10"); cv.belief.classList.add("belief"); el.append(cv.chips = h("div", { class: "chips" })); }
    if (b.id === "bb") { c("h1", 160, 160, "layer 1 · 64"); c("h2", 160, 160, "layer 2 · 64"); }
    if (b.id === "neck") { c("kc", 320, 160, ""); el.append(cv.kcn = h("div", { class: "mini" })); }
    if (b.id === "head") { c("logit", 240, 110, "logits: prior (light) + plastic (dark)"); el.append(cv.val = h("div", { class: "mini" })); }
    if (b.id === "out") { c("pol", 140, 140, ""); el.append(cv.move = h("div", { class: "mini" })); }
    for (const [t, k] of b.tags) el.append(h("span", { class: `tag ${k}` }, t));
    host.append(el);
  }
  $("arch-detail").innerHTML = ARCH[2].detail;
  return A => {
    if (!A?.out) return; const x = A.x, o = A.out, E = D.E, w = A.done ? A.result : A.env.w[0];
    drawView(cv.view, A.env.frame(w.pos, w.heading, w.rot).img, E.view); drawPooled(cv.pool, x.subarray(0, 144), ...E.pooled);
    const bel = x.subarray(150, 250), bm = Math.max(...bel); drawStrip(cv.belief, Array.from(bel, v => v / bm), 10);
    cv.chips.innerHTML = ""; cv.chips.append(h("span", { class: "chip" + (x[144] ? " hot" : "") }, "contact " + x[144]), h("span", { class: "chip" + (x[145] ? " hot" : "") }, "reward " + x[145]),
      ...MOVE_NAMES.map((n, i) => h("span", { class: "chip" + (x[146 + i] ? " hot" : "") }, n)));
    drawStrip(cv.h1, o.h1, 8); drawStrip(cv.h2, o.h, 8);
    drawKC(cv.kc, A.L.K, o.z); cv.kcn.textContent = `${o.z.i.length} of ${A.L.K} cells active`;
    const g = cv.logit.getContext("2d"), W = cv.logit.width, H = cv.logit.height, base = A.L.prior.heads(o.h).lg; g.clearRect(0, 0, W, H);
    MOVE_NAMES.forEach((n, i) => { const y = 8 + i * 25, z0 = 46 + (W - 46) / 2, sc = 12, b = base[i], tot = o.lg[i];
      g.fillStyle = "rgba(125,60,152,.28)"; g.fillRect(z0, y, b * sc, 16); g.fillStyle = "rgba(125,60,152,.9)"; g.fillRect(z0 + b * sc, y + 4, (tot - b) * sc, 8);
      g.fillStyle = getComputedStyle(document.body).color; g.font = "11px Inter"; g.fillText(n, 2, y + 12); g.fillStyle = "rgba(128,128,128,.5)"; g.fillRect(z0, y - 2, 1, 20); });
    cv.val.textContent = `value V = ${o.v.toFixed(3)} · plastic part ${(o.lg.reduce((s, v, i) => s + Math.abs(v - base[i]), 0)).toFixed(3)}`;
    drawPolicy(cv.pol, o.p, A.a, INFO(A.m).col); const nx = o.p.indexOf(Math.max(...o.p)); cv.move.textContent = `most likely next: ${MOVE_NAMES[nx]} (${(100 * o.p[nx]).toFixed(0)}%)`;
  };
}

// ---------------- Learn & forget: the same circuit, without and with the latent-cause gate, side by side ----------------
function learningLab(play) {
  const S = { prior: "human_bc", seed: 1000 }, c = $("learn-controls"), lanes = {}, probes = { none: [], inferred: [] }, learned = {};
  const room = () => D.E.rooms.find(r => r.configuration === 100 + 2 * (S.seed % 8)), unseen = () => D.E.rooms.find(r => r.configuration === 101 + 2 * (S.seed % 8));
  const mask = (() => { const m = [new Uint8Array(2048), new Uint8Array(2048)]; for (const j of D.C.gate_half0) m[0][j] = 1; for (let j = 0; j < 2048; j++) m[1][j] = 1 - m[0][j]; return m; })();
  for (const [g, title] of [["none", "mushroom-body circuit"], ["inferred", "mushroom-body circuit + latent-cause gate"]]) {
    const L = lanes[g] = { room: h("canvas", { class: "lroom" }), kc: h("canvas", { class: "lkc" }), badge: h("span", { class: "ctxbadge" }, g === "none" ? "no context" : "context 0"),
      stat: h("div", { class: "lstat" }), big: h("div", { class: "big" }, "–"), W: null };
    $("duel").append(h("div", { class: "lane" }, h("h3", { style: `color:${g === "none" ? "var(--fg)" : "var(--bio)"}` }, title, L.badge), L.room,
      h("div", {}, h("div", { class: "mini" }, "Kenyon cells, animal 1"), L.kc, h("div", { class: "mini" }, "old goal kept"), L.big, L.stat)));
  }
  const idle = () => { for (const L of Object.values(lanes)) drawRoom(L.room, room().walkable, { goal: room().familiar_goal, alt: room().moved_goal, starts: D.E.starts }); };
  const refit = () => { for (const L of Object.values(lanes)) { fit(L.room, L.room.clientWidth, L.room.clientWidth); fit(L.kc, L.kc.clientWidth, L.kc.clientWidth / 2); } idle(); };
  requestAnimationFrame(refit);
  const runB = h("button", { class: "btn", onclick: () => run() }, "▶ move the goal and learn"), stopB = h("button", { class: "btn ghost", onclick: () => Object.values(lanes).forEach(L => L.W?.postMessage({ stop: true })) }, "stop");
  const loadB = h("button", { class: "btn ghost", disabled: "", onclick: () => { const g = learned.inferred ? "inferred" : "none", x = learned[g];
    play.learned = { prior: S.prior, seed: S.seed, Wa: x.Wa, gate: g, ctx: x.ctx }; Object.assign(play.s, { seed: S.seed, room: D.E.rooms.indexOf(room()), goal: "moved", scenario: "lab" });
    play.s.models = LEARN_PRIORS.includes(S.prior) ? [S.prior, `ppo@${S.prior}`] : [S.prior]; play.renderSlots(); play.walk = null; play.reset(); location.hash = "play"; } }, "try the learned agent →");
  c.append(sel("Prior", D.C.priors.map(p => [p, PRIOR_INFO[p].label]), S.prior, v => S.prior = v), sel("Seed", D.C.seeds.map(s => [s, s]), S.seed, v => { S.seed = +v; idle(); }), runB, stopB, loadB);
  const chart = () => {
    const ser = [];
    for (const [g, col] of [["none", "#8c8c8c"], ["inferred", COL.bio]]) for (const [k, l, dash] of [["new", "new goal", ""], ["old", "old goal", "5 3"]])
      ser.push({ label: `${l}${g === "inferred" ? ", gated" : ""}`, col, dash, x: probes[g].map(p => p.frames || 1), y: probes[g].map(p => [p[k], 0]) });
    const pc = D.R.curves[`ppo/${S.prior}`], po = D.R.kpi[`ppo/${S.prior}/familiar_trained_room`];
    if (pc) ser.push({ label: "PPO new goal (paper, 30 seeds)", col: "#33383f", dash: "2 2", x: pc.x.map(x => x || 1), y: pc.y });
    if (po) ser.push({ label: "PPO old goal kept (paper)", col: "#33383f", open: true, x: [D.E.budget], y: [po] });
    lines($("learn-chart"), ser, { xmin: 0, xmax: D.E.budget, ymin: 0, ymax: 100, ylabel: "probe success (%)", xlabel: "interactions (16 animals); the goal moves at 2,048", w: 640, h: 300 });
  };
  const ref = g => { const K = D.R.kpi, l = g === "none" ? "bl3_none" : "bl2_inferred", m = k => mean(["cold_ppo", "human_legal_uniform", "human_bc"].map(p => K[`${l}/${p}/${k}`][0]));
    return `paper, 30 seeds × 3 priors: new goal ${m("new").toFixed(0)}%, old goal kept ${m("familiar_trained_room").toFixed(0)}%`; };
  for (const g of ["none", "inferred"]) lanes[g].stat.textContent = ref(g);
  chart(); $("learn-note").innerHTML = "Both circuits start from the same prior and see the same rooms; probes are frozen 30-trial tests. One run varies a lot, so press ▶ again or change the seed. On average the gate keeps more of the old goal and learns the new one more slowly. Dark dotted line: PPO (backprop) learning the same goal, from the paper's 30 seeds.";
  function run() {
    const r = room(), u = unseen(), runSeed = Date.now() % 100000; loadB.disabled = true;
    for (const g of ["none", "inferred"]) {
      const L = lanes[g]; L.W?.terminate(); probes[g] = []; delete learned[g]; L.big.textContent = "…";
      L.W = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
      L.W.postMessage({ prior: S.prior, seed: S.seed, gate: g, room: r, unseen: u, familiar: r.familiar_goal, moved: r.moved_goal, movedUnseen: u.moved_goal, runSeed });
      L.W.onmessage = ({ data: m }) => {
        if (m.step) { const s = m.step, moved = s.phase === "moved";
          drawRoom(L.room, r.walkable, { goal: moved ? r.moved_goal : r.familiar_goal, alt: moved ? r.familiar_goal : null, dots: s.pos, dotCol: g === "none" ? "#666" : COL.bio });
          drawKC(L.kc, 2048, s.z, g === "inferred" ? mask : null, s.ctx);
          if (g === "inferred") { L.badge.textContent = `context ${s.ctx}${s.switches ? ` · ${s.switches} switch${s.switches > 1 ? "es" : ""}` : ""}`; L.badge.classList.toggle("on", s.ctx === 1); }
          L.stat.innerHTML = `${moved ? "goal moved" : "familiar goal"} · ${(s.frames / 1000).toFixed(1)}k / ${D.E.budget / 1000 | 0}k · ${(s.ms / 1000).toFixed(0)} s<br>${ref(g)}`; }
        if (m.probe) { probes[g].push(m.probe); chart(); L.big.textContent = `${m.probe.old.toFixed(0)}%`; L.big.style.color = m.probe.old < 15 ? "var(--rat)" : "var(--ok)"; }
        if (m.done) { learned[g] = { Wa: Float64Array.from(m.Wa), ctx: m.ctx }; loadB.disabled = false; }
        if (m.error) L.stat.textContent = "error: " + m.error;
      };
    }
  }
  return refit;
}

// ---------------- evidence tiles ----------------
function tiles() {
  const R = D.R, ood = R.ood.registered_sample_source, m = (a, k) => mean(ood[a][k]), K = (l, k) => mean(["cold_ppo", "human_legal_uniform", "human_bc"].map(p => R.kpi[`${l}/${p}/${k}`][0]));
  const T = [
    ["+9.9 pp", "success in unseen rooms, human choices vs legal moves", "#results", [["human", m("human_bc", "success"), COL.human], ["legal", m("human_legal_uniform", "success"), COL.legal_h]], 60, "%"],
    ["−8.2 pp", "fewer revisits: the human habit of not going back", "#animals", [["human", m("human_bc", "revisit"), COL.human], ["legal", m("human_legal_uniform", "revisit"), COL.legal_h]], 80, "%"],
    ["+12.4 pp", "moved goal: mushroom-body circuit vs PPO, no backpropagation", "#learn", [["MB circuit", K("bl3_none", "new"), COL.bio], ["PPO", K("ppo", "new"), "#8c8c8c"]], 100, "%"],
    ["43%", "of the old goal kept with the latent-cause gate (4.7% without it)", "#results", [["gate", 43, COL.bio], ["none", 4.7, "#8c8c8c"]], 100, "%"],
    ["75%", "of next moves predicted from one view (majority 47%), yet the route is not", "#results", [["view", 75, COL.human], ["majority", 47, "#8c8c8c"]], 100, "%"],
    ["0.06", "unpaired alignment of networks trained on the same choices (chance 0.5)", "#results", [["matched", .06, COL.human], ["chance", .5, "#8c8c8c"]], .6, ""],
  ];
  for (const [big, txt, href, rows, max, unit] of T) {
    const spark = h("div", { class: "spark" }); minibars(spark, rows, max, unit);
    $("tiles").append(h("a", { class: "tile", href }, h("b", {}, big), h("span", {}, txt), spark));
  }
}

// ---------------- the mushroom-body circuit, in brain terms (each term is searchable) ----------------
const GLOSSARY = [
  ["Mushroom body · Kenyon cells", "The insect brain's learning centre. Thousands of Kenyon cells each listen to a few random inputs, so similar scenes get distinct codes.", "2,048 cells, each wired to 7 random units of the backbone; the wiring never changes.", "mushroom body Kenyon cells sparse coding"],
  ["Cerebellar granule cells", "The same motif in mammals: a huge layer of small cells re-codes the input before a learning stage.", "the neck is this expansion (64 → 2,048), like a fixed reservoir without recurrence.", "cerebellum granule cell expansion recoding"],
  ["Winner-take-all inhibition", "One inhibitory neuron (APL in flies, Golgi cells in the cerebellum) silences all but the most active cells.", "only the top 2% of Kenyon cells fire at each step.", "APL neuron global inhibition sparse Kenyon cells"],
  ["Divisive normalisation", "Neurons divide their activity by the pooled activity of their neighbours, a canonical neural computation.", "the active cells' rates are scaled to unit length.", "divisive normalization canonical neural computation"],
  ["Dopamine prediction error", "Dopamine neurons fire for 'better than expected' and pause for 'worse than expected'.", "δ = reward + discounted next value − current value, broadcast to every plastic synapse.", "dopamine reward prediction error Schultz"],
  ["Three-factor learning rule", "A synapse changes only if its two neurons were recently active together (an eligibility trace) and dopamine then arrives.", "ΔW = η · δ · trace; no backpropagation and no optimiser.", "three-factor learning rule eligibility trace"],
  ["Metaplasticity", "Synapses that have changed a lot become harder to change again, which protects old memories.", "each synapse's learning rate falls with the change it has accumulated.", "metaplasticity synaptic consolidation Benna Fusi"],
  ["Awake replay", "After an experience the hippocampus quickly replays it in sharp-wave ripples, which speeds up learning.", "the last 16 steps are replayed 4 times with the same rule.", "hippocampal awake replay sharp wave ripples"],
  ["Adaptive dopamine and exploration", "Dopamine responses rescale to the range of recent rewards, and noradrenaline sets how exploratory behaviour is.", "δ is divided by its running size; moves are sampled at a temperature while learning.", "adaptive coding dopamine Tobler locus coeruleus exploration"],
  ["Latent-cause inference", "When outcomes suddenly stop matching expectations, the brain infers that the world has changed and starts a new memory.", "a change detector on successes switches to the other half of the Kenyon cells, with no cue.", "latent cause inference Gershman"],
  ["Behavioural cloning", "Learning a policy by imitating recorded choices.", "the backbone (252 → 64 → 64) was trained on human or rat choices, then frozen.", "behavioral cloning imitation learning"],
];
function glossary() {
  const host = $("glossary"); if (!host) return;
  for (const [term, plain, ours, q] of GLOSSARY) host.append(h("div", { class: "gl" }, h("b", {}, term), h("span", {}, plain), h("i", {}, "In the model: " + ours),
    h("a", { href: `https://www.google.com/search?q=${encodeURIComponent(q)}`, target: "_blank", rel: "noopener" }, "search ↗")));
}

// ---------------- Lean ----------------
function proofList() {
  const P = [["trapped_forever", "a frozen agent that revisits a state loops forever"], ["relabel_keeps_states", "the legal-label control changes only actions"],
    ["species_gap_decomposition", "human − rat = coverage gap + choice gap"], ["trace_bounded", "the circuit's eligibility trace stays bounded"],
    ["softmax_logit_grad", "the local rule is the policy gradient (Mathlib)"]];
  for (const [n, t] of P) { const e = h("div", { class: "proof", onclick: () => e.classList.toggle("on") }, h("code", {}, n), h("span", {}, t), h("pre", {}, D.R.proofs[n])); $("proof-list").append(e); }
}

function reveal() {
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: .08 });
  document.querySelectorAll("section figure, .tile, .lane, .block, .proof").forEach(e => { e.classList.add("reveal"); io.observe(e); });
}

// ---------------- Humans vs rats ----------------
function animals() {
  const T = D.A.tasks, ids = Object.keys(T), cfgs = [...new Set(ids.map(k => +k.split("/")[1]))].sort((a, b) => a - b), S = { cfg: 16, start: 9 }, c = $("animal-controls");
  const cvH = $("an-a"), cvR = $("an-b"), capId = { human: "an-a-cap", rat: "an-b-cap" }; let anim = null;
  const starts = () => ids.filter(k => +k.split("/")[1] === S.cfg).map(k => +k.split("/")[2]).sort((a, b) => a - b);
  const startSel = sel("Start", [], "", v => { S.start = +v; draw(); });
  const fillStarts = () => { const s = startSel.querySelector("select"); s.innerHTML = ""; for (const v of starts()) s.append(h("option", { value: v }, v)); if (!starts().includes(S.start)) S.start = starts()[0]; s.value = S.start; };
  c.append(sel("Maze configuration", cfgs.map(v => [v, v]), S.cfg, v => { S.cfg = +v; fillStarts(); draw(); }), startSel);
  fillStarts();
  function draw() {
    cancelAnimationFrame(anim); const t = T[`tartarus/${S.cfg}/${S.start}`];
    const one = (cv, sp, col, path) => { const a = t.species[sp]; drawRoom(cv, t.walkable, { goal: t.goal, start: t.spawn, occ: a?.occ, occCol: col, path, col });
      $(capId[sp]).innerHTML = a ? `<b style="color:${col}">${sp}s</b> · ${a.success}/${a.n} reached · ${a.revisit}% revisits` : `no ${sp} trials from this start`; };
    const ex = `tartarus/${S.cfg}/${S.start}` === D.A.example.task ? D.A.example.paths : null; let k = 0;
    const step = () => { one(cvH, "human", COL.human, ex?.human.slice(0, k)); one(cvR, "rat", COL.rat, ex?.rat.slice(0, k)); if (ex && k++ < Math.max(ex.human.length, ex.rat.length) + 8) anim = setTimeout(() => requestAnimationFrame(step), 160); else if (ex) { k = 0; anim = setTimeout(step, 1200); } };
    step();
  }
  requestAnimationFrame(() => { fit(cvH, cvH.clientWidth, cvH.clientWidth); fit(cvR, cvR.clientWidth, cvR.clientWidth); draw(); });
  bars($("an-revisit"), [{ label: `humans\n(n=${D.R.native.human.length})`, col: COL.human, values: D.R.native.human }, { label: `rats\n(n=${D.R.native.rat.length})`, col: COL.rat, values: D.R.native.rat }],
    { ylabel: "revisits (%)", ymax: 30, w: 260, h: 250 });
}

// ---------------- Results ----------------
const NICE2 = s => s.replace("brain circuit", "mushroom-body circuit");
function results() {
  const R = D.R, ARMS = ["human_bc", "human_legal_uniform", "rat_bc", "rat_legal_uniform", "rat_sham", "cold_ppo"], lab = a => PRIOR_INFO[a].short.replace(" ", "\n");
  segment($("r-ood-seg"), [["success|registered_sample_source", "success, familiar goal"], ["revisit|registered_sample_source", "revisits"], ["success|sample_moved", "success, moved goal"]], "success|registered_sample_source", v => {
    const [k, fam] = v.split("|"); bars($("r-ood"), ARMS.filter(a => R.ood[fam]?.[a]).map(a => ({ label: lab(a), col: PRIOR_INFO[a].col, values: R.ood[fam][a][k] })),
      { ylabel: k === "revisit" ? "moves revisiting a cell (%)" : "success, frozen (%)", ymin: k === "revisit" ? 40 : 0, ymax: k === "revisit" ? 80 : 60, ref: k === "success" ? 13.3 : null, refLabel: "uniform\npolicy", mr: k === "success" ? 52 : 10 }); });
  const L = Object.keys(R.nice).filter(l => l !== "before"), PR = ["cold_ppo", "human_legal_uniform", "human_bc"], bio = l => l.startsWith("bl");
  const lm = (l, key) => { const v = PR.map(p => R.kpi[`${l}/${p}/${key}`]?.[0]).filter(x => x != null); return v.length ? mean(v) : null; };
  scatter($("r-keep"), L.filter(l => lm(l, "new") != null).map(l => ({ x: lm(l, "new"), y: lm(l, "familiar_trained_room"), col: bio(l) ? COL.bio : l === "G2" ? COL.rat : "#7a7a7a", label: NICE2(R.nice[l]), text: NICE2(R.nice[l]), dy: l === "cs_dfa" ? 13 : l === "bl3_none" ? -6 : 3 })),
    { xmin: 0, xmax: 100, ymin: 0, ymax: 100, xlabel: "new goal, trained room (%)", ylabel: "old goal kept (%)" });
  segment($("r-forget-seg"), [["familiar_trained_room", "old goal, trained room"], ["familiar_heldout_room", "old goal, unseen room"], ["moved_heldout_room", "new goal, unseen room"]], "familiar_trained_room", key => {
    const G = []; for (const l of ["before", ...L]) for (const p of PR) { const v = R.kpi[`${l}/${p}/${key}`]; if (v) G.push({ label: p === "human_legal_uniform" ? NICE2(R.nice[l]).replace(/ /g, "\n").replace("+", "+\n").slice(0, 26) : "", col: PRIOR_INFO[p].col, m: v[0], h: v[1] }); }
    bars($("r-forget"), G, { ylabel: "success (%)", ymax: 100, w: 520, mb: 64 }); });
  segment($("r-curve-seg"), [["ppo", "PPO"], ["td_head", "TD-readout"], ["td_fa", "TD-DFA"]], "ppo", l => {
    lines($("r-curves"), PR.filter(p => R.curves[`${l}/${p}`]).map(p => ({ label: PRIOR_INFO[p].label, col: PRIOR_INFO[p].col, ...R.curves[`${l}/${p}`] }))
      .concat([{ label: "◯ mushroom-body circuit, end", col: COL.bio, open: true, x: [32768], y: [[mean(PR.map(p => R.kpi[`bl3_none/${p}/new`][0])), 0]] }]),
      { xmin: 0, xmax: 32768, ymin: 0, ymax: 100, ylabel: "moved-goal success (%)", xlabel: "interactions" }); });
  segment($("r-snn-seg"), [["priors", "frozen priors"], ["learned", "after learning: PPO vs local rules"]], "priors", v => {
    if (v === "priors") return lines($("r-snn"), PR.map(p => { const T = [1, 2, 4, 8, 16, 32].filter(t => R.snn[`${p}/T${t}`]); return { label: PRIOR_INFO[p].label, col: PRIOR_INFO[p].col, x: T.map(t => R.snn[`${p}/T${t}`].synops / 1000), y: T.map(t => R.snn[`${p}/T${t}`].success) }; }),
      { xmin: 1, xmax: 1000, xlog: true, ymin: 0, ymax: 100, ylabel: "success (%)", xlabel: "synaptic operations per decision (thousands)" });
    const SL = R.snn_learned, L2 = [["ppo", "PPO (backprop)", "#33383f"], ["td_head", "TD-readout (local)", "#9b6fbf"], ["s2_ed", "local + replay", COL.bio]];
    lines($("r-snn"), L2.filter(([k]) => SL[`${k}/T8`]).map(([k, l, col]) => ({ label: l, col, x: [8, 16, 32].map(t => SL[`${k}/T${t}`].synops / 1000), y: [8, 16, 32].map(t => [SL[`${k}/T${t}`].spiking, 0]) })),
      { xmin: 1, xmax: 10000, xlog: true, ymin: 0, ymax: 105, ylabel: "spiking success (%)", xlabel: "synaptic operations per decision (thousands)" }); });
  const VA = ["human_bc", "human_legal_uniform", "rat_bc", "cold_ppo"];
  bars($("r-vla"), VA.map(a => ({ label: "move\n" + PRIOR_INFO[a].short, col: PRIOR_INFO[a].col, m: 100 * R.text_to_move[a], h: 0 })).concat(
    VA.map(a => ({ label: "route\n" + PRIOR_INFO[a].short, col: PRIOR_INFO[a].col, alpha: .45, m: R.closed_loop[a][0], h: R.closed_loop[a][1] }))), { ylabel: "%", ymax: 100, ref: 13.3, refLabel: "uniform\npolicy", w: 520, mr: 52 });
  const kinds = ["image", "text", "cortex", "hid", "act"], names = { image: "image", text: "caption", cortex: "mouse cortex", hid: "agent hidden", act: "action" };
  segment($("r-cka-seg"), VA.map(a => [a, PRIOR_INFO[a].short]), "human_bc", a => {
    const v = (x, y) => { if (x === y) return 1; const k = [`${x}-${y}`, `${y}-${x}`]; for (const kk of k) { if (R.cka[`${kk}|${a}`]) return R.cka[`${kk}|${a}`][0]; if (R.cka[`${kk}|pooled`]) return R.cka[`${kk}|pooled`][0]; } return null; };
    heat($("r-cka"), kinds.map(x => kinds.map(y => v(x, y))), kinds.map(k => names[k]), kinds.map(k => names[k])); });
  bars($("r-brain"), VA.map(a => ({ label: lab(a), col: PRIOR_INFO[a].col, m: R.cka[`cortexfull-hid|${a}`][0], h: R.cka[`cortexfull-hid|${a}`][1] })), { ylabel: "CKA to mouse visual cortex", ymin: 0, ymax: .7 });
}

// ---------------- boot ----------------
(async () => {
  const [E, C, R, A] = await Promise.all([get("env.json"), get("circuits.json"), get("results.json"), get("animals.json")]);
  Object.assign(D, { E: prepare(E), C, R, A });
  results(); animals();
  tiles(); proofList();
  const hero = new Arena($("hero"), { autoplay: true }); hero.s.speed = 5; hero.nextEpisode(); await hero.reset(); hero.loop(); glossary(); window.__arenas = { hero };   // test handle (site/test/browser.mjs)
  const show = architecture(); const play = new Arena($("play-agents"), { controls: $("play-controls"), onSelect: show }); await play.reset(); play.playing = true; play.loop();
  const refitLab = learningLab(play); addEventListener("resize", () => requestAnimationFrame(refitLab)); reveal();
})().catch(e => { console.error(e); document.body.insertAdjacentHTML("afterbegin", `<p style="padding:12px;background:#c0392b;color:#fff">Could not load the page data: ${e.message}</p>`); });
