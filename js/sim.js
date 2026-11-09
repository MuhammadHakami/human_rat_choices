// The paper's agents, environment and brain-like circuit, ported line by line from the registered Python
// (scripts/tartarus_visual_env.py render, wacv_embodied_source_cache.pool_pixels, wacv_native_prior_confirm_env.LogBelief,
//  wacv_continual_geometry_env_20261012.ContinuousRooms/VisualRooms, v4n_brainlike*_claude Circuit/Circuit2,
//  v4n_context_gating_claude_20261103.Latent). site/test/parity.mjs checks it against Python traces.
// Runs in the browser and in Node. Only the random number generator differs from Python.

export const MOVES = [[1, 0], [-1, 0], [0, 1], [0, -1]];             // action index -> (d_row, d_col)
export const MOVE_NAMES = ["south", "north", "east", "west"];

// ---------- random numbers (mulberry32; Python uses numpy / torch generators) ----------
export function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export const pick = (p, u) => { let c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (u < c) return i; } return p.length - 1; };

// ---------- renderer: luminance image of the first-person view ----------
export function render(view, walk, cell, heading, texRot) {
  const W = view.width, H = view.height, L = view.lum, n = W * H;
  const eye = [cell[1] + .5, view.eye_h, cell[0] + .5];
  const f = [heading[1], 0, heading[0]], right = [-f[2], 0, f[0]];
  const tx = Math.tan(view.hfov * Math.PI / 360), ty = tx * H / W;
  const D = new Float64Array(n * 3), tb = new Float64Array(n).fill(Infinity), sid = new Uint8Array(n);
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    const U = ((i + .5) / W * 2 - 1) * tx, V = (1 - (j + .5) / H * 2) * ty, k = (j * W + i) * 3;
    const x = f[0] + U * right[0], y = f[1] + V, z = f[2] + U * right[2], s = Math.hypot(x, y, z);
    D[k] = x / s; D[k + 1] = y / s; D[k + 2] = z / s;
  }
  const lo = [0, -view.pit_depth, 0], hi = [0, 0, 0];
  for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
    if (!walk[r][c]) continue;
    lo[0] = c; lo[2] = r; hi[0] = c + 1; hi[2] = r + 1;
    for (let p = 0; p < n; p++) {
      let tn = -Infinity, tf = Infinity;
      for (let a = 0; a < 3; a++) {
        const d = D[p * 3 + a], t1 = (lo[a] - eye[a]) / d, t2 = (hi[a] - eye[a]) / d;
        tn = Math.max(tn, Math.min(t1, t2)); tf = Math.min(tf, Math.max(t1, t2));
      }
      if (tn <= tf && tf > 0 && tn > 1e-9 && tn < tb[p]) { tb[p] = tn; sid[p] = 1; }
    }
  }
  const m = view.curtain_margin, curt = [["row0", 2, -m], ["row9", 2, 10 + m], ["col0", 0, -m], ["col9", 0, 10 + m]];
  for (let p = 0; p < n; p++) {
    const dy = D[p * 3 + 1], tp = (-view.pit_depth - eye[1]) / dy;
    if (tp > 0 && tp < tb[p]) { tb[p] = tp; sid[p] = 2; }
    for (const [name, ax, pos] of curt) {
      const tc = (pos - eye[ax]) / D[p * 3 + ax], y = eye[1] + tc * dy;
      if (tc > 0 && tc < tb[p] && y < view.curtain_h) { tb[p] = tc; sid[p] = name === view.cue_edge ? 4 : 3; }
    }
    if (sid[p] === 1 && eye[1] + tb[p] * dy < -1e-6) sid[p] = 5;
  }
  const lum = [L.sky, L.floor, L.pit, L.white, L.black, L.side], img = new Float64Array(n), ang = 37 * texRot * Math.PI / 180;
  const mod1 = x => x - Math.floor(x);
  for (let p = 0; p < n; p++) {
    let v = lum[sid[p]];
    if (sid[p] === 1 && view.tex_amp) {
      const q0 = mod1(eye[0] + D[p * 3] * tb[p]) - .5, q1 = mod1(eye[2] + D[p * 3 + 2] * tb[p]) - .5;
      v += view.tex_amp * Math.sin(2 * Math.PI * (Math.cos(ang) * q0 + Math.sin(ang) * q1) / view.tex_period);
    }
    const tt = isFinite(tb[p]) ? tb[p] : 60, dy = D[p * 3 + 1];
    const tcr = Math.abs(dy) > 1e-12 ? (view.fog_top - eye[1]) / dy : Infinity;
    const fl = eye[1] < view.fog_top ? ((dy > 0 && tcr < tt) ? tcr : tt) : ((dy < 0 && tcr < tt) ? tt - tcr : 0);
    const T = Math.exp(-view.fog_density * Math.max(fl - view.fog_clear, 0));
    img[p] = T * v + (1 - T) * L.fog;
  }
  return img;
}

const rint = x => { const r = Math.round(x); return (Math.abs(x % 1) === .5 && r % 2) ? r - 1 : r; };   // numpy rint: ties to even
export function pool(img, view, oh, ow) {                            // 8-bit camera, mean-pooled to oh x ow, mapped to [-1, 1]
  const W = view.width, H = view.height, bh = H / oh, bw = W / ow, pix = new Uint8Array(W * H), out = new Float32Array(oh * ow);
  for (let p = 0; p < pix.length; p++) pix[p] = rint(Math.min(Math.max(img[p], 0), 1) * 255);
  for (let a = 0; a < oh; a++) for (let b = 0; b < ow; b++) {
    let s = 0;
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) s += Math.fround(pix[(a * bh + y) * W + b * bw + x] / 255);
    out[a * ow + b] = 2 * Math.fround(s / (bh * bw)) - 1;
  }
  return { pix, feat: out };
}

// ---------- localisation filter (Bayes, log domain) ----------
const lse = a => { let m = -Infinity; for (const x of a) m = Math.max(m, x); if (m === -Infinity) return m; let s = 0; for (const x of a) s += Math.exp(x - m); return m + Math.log(s); };
export class Belief {
  constructor(atlas, sigma) { this.atlas = atlas; this.s2 = 2 * sigma * sigma; this.reset(); }   // atlas: 100 cells x 4 headings x 64
  reset() { this.logp = new Float64Array(100).fill(-Math.log(100)); }
  observe(cue, action, contact) {
    if (action != null && !contact) {
      const [dr, dc] = MOVES[action], nl = new Float64Array(100).fill(-Infinity);
      for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
        const r2 = r + dr, c2 = c + dc;
        if (r2 >= 0 && r2 < 10 && c2 >= 0 && c2 < 10) nl[r2 * 10 + c2] = this.logp[r * 10 + c];
      }
      this.logp = nl;
    }
    const lp = new Float64Array(100);
    for (let s = 0; s < 100; s++) {
      const ll = [];
      for (let h = 0; h < 4; h++) {
        if (action != null && h !== action) continue;
        let e = 0; const o = (s * 4 + h) * 64;
        for (let i = 0; i < 64; i++) { const d = this.atlas[o + i] - cue[i]; e += d * d; }
        ll.push(-e / 64 / this.s2);
      }
      lp[s] = this.logp[s] + (action == null ? lse(ll) - Math.log(4) : ll[0]);
    }
    const z = lse(lp); if (!isFinite(z)) throw new Error("belief lost support");
    for (let s = 0; s < 100; s++) this.logp[s] = lp[s] - z;
    return Float32Array.from(this.logp, Math.exp);
  }
}

// ---------- the world: one room, many parallel animals ----------
export const inwardHeading = ([r, c]) => { const dr = 4.5 - r, dc = 4.5 - c; return Math.abs(dr) >= Math.abs(dc) ? (dr >= 0 ? 0 : 1) : (dc >= 0 ? 2 : 3); };
export function bfs(walk, from) {
  const d = new Map([[from[0] * 10 + from[1], 0]]), q = [from];
  while (q.length) { const [r, c] = q.shift(), k = d.get(r * 10 + c);
    for (const [dr, dc] of MOVES) { const a = r + dr, b = c + dc; if (a >= 0 && a < 10 && b >= 0 && b < 10 && walk[a][b] && !d.has(a * 10 + b)) { d.set(a * 10 + b, k + 1); q.push([a, b]); } } }
  return d;
}

const FRAMES = new Map(), FRAME_CAP = 4000;                              // views shared across episodes (least recently used out)
export class Env {
  // E: env.json; room: {walkable, goal}; spawner(): {spawn, rotation_index}
  constructor(E, room, n, spawner) {
    this.E = E; this.walk = room.walkable; this.goal = room.goal; this.n = n; this.spawner = spawner; this.wkey = room.walkable.flat().join("");
    this.belief = Array.from({ length: n }, () => new Belief(E.atlasF, E.sigma)); this.completed = [];
  }
  frame(cell, heading, rot) {
    const k = `${this.wkey}|${cell}|${heading}|${rot}`; let f = FRAMES.get(k);
    if (f) { FRAMES.delete(k); FRAMES.set(k, f); return f; }
    const img = render(this.E.view, this.walk, cell, heading, this.E.rotations[rot]); f = { img: Float32Array.from(img), ...pool(img, this.E.view, ...this.E.pooled) };
    FRAMES.set(k, f); if (FRAMES.size > FRAME_CAP) FRAMES.delete(FRAMES.keys().next().value); return f;
  }
  newWorld() {
    const { spawn, rotation_index } = this.spawner();
    return { spawn, pos: spawn, heading: MOVES[inwardHeading(spawn)], rot: rotation_index, decisions: 0, travelled: 0, contact: false, last: null, reward: 0, path: [spawn], actions: [] };
  }
  sense(i, reset, action) {
    const w = this.w[i], x = new Float32Array(252), f = this.frame(w.pos, w.heading, w.rot).feat;
    x.set(f); x[144] = +w.contact; x[145] = w.reward; if (w.last != null) x[146 + w.last] = 1;
    if (reset) this.belief[i].reset();
    const p = this.belief[i].observe(f.subarray(0, 64), reset ? null : action, w.contact);
    x.set(p, 150); let a = 0, b = 0; for (let s = 0; s < 100; s++) { a += p[s] * this.E.coord[s][0]; b += p[s] * this.E.coord[s][1]; }
    x[250] = a; x[251] = b; return x;
  }
  reset() { this.w = Array.from({ length: this.n }, () => this.newWorld()); return this.w.map((_, i) => this.sense(i, true)); }
  step(actions) {
    const obs = [], r = new Float32Array(this.n), done = new Uint8Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const w = this.w[i], a = actions[i], q = [w.pos[0] + MOVES[a][0], w.pos[1] + MOVES[a][1]];
      const ok = q[0] >= 0 && q[0] < 10 && q[1] >= 0 && q[1] < 10 && !!this.walk[q[0]][q[1]];
      w.pos = ok ? q : w.pos; w.contact = !ok; w.heading = MOVES[a]; w.last = a; w.decisions++; w.travelled += +ok; w.path.push(w.pos); w.actions.push(a);
      const success = w.pos[0] === this.goal[0] && w.pos[1] === this.goal[1]; w.reward = r[i] = +success;
      done[i] = +(success || w.decisions === this.E.max_decisions);
      if (done[i]) { this.completed.push({ ...w, success }); this.w[i] = this.newWorld(); obs.push(this.sense(i, true)); }
      else obs.push(this.sense(i, false, a));
    }
    return { obs, r, done };
  }
}

// ---------- networks ----------
export class Prior {                                                  // frozen cortex: 252 -> 64 tanh -> 64 tanh; heads pi (4), v (1)
  constructor(buf) {
    const a = new Float32Array(buf); let o = 0; const take = k => a.subarray(o, o += k);
    this.W1 = take(64 * 252); this.b1 = take(64); this.W2 = take(64 * 64); this.b2 = take(64); this.Wp = take(256); this.bp = take(4); this.Wv = take(64); this.bv = take(1);
  }
  body(x) {
    const h1 = new Float32Array(64), h = new Float32Array(64);
    for (let j = 0; j < 64; j++) { let s = this.b1[j]; for (let i = 0; i < 252; i++) s += this.W1[j * 252 + i] * x[i]; h1[j] = Math.tanh(s); }
    for (let j = 0; j < 64; j++) { let s = this.b2[j]; for (let i = 0; i < 64; i++) s += this.W2[j * 64 + i] * h1[i]; h[j] = Math.tanh(s); }
    return { h1, h };
  }
  heads(h) {
    const lg = new Float64Array(4); let v = this.bv[0];
    for (let a = 0; a < 4; a++) { let s = this.bp[a]; for (let i = 0; i < 64; i++) s += this.Wp[a * 64 + i] * h[i]; lg[a] = s; }
    for (let i = 0; i < 64; i++) v += this.Wv[i] * h[i];
    return { lg, v };
  }
}
export const softmax = (lg, tau = 1) => { const m = Math.max(...lg); const e = Array.from(lg, x => Math.exp((x - m) / tau)); const s = e.reduce((a, b) => a + b); return e.map(x => x / s); };

export class Circuit {
  // Mushroom-body / cerebellum-like circuit on the frozen prior: K granule (Kenyon) cells, each wired to c hidden units,
  // k-WTA + divisive normalisation; plastic KC->action and KC->value synapses (start at zero), three-factor TD rule with
  // eligibility traces, metaplasticity, adaptive dopamine, awake replay of the last T steps (E times); optional latent-cause gate.
  constructor(prior, kcBuf, cfg, half0) {
    this.prior = prior; this.c = cfg; const K = this.K = cfg.K, cc = cfg.c;
    this.idx = new Int32Array(kcBuf, 0, K * cc); this.w = new Float32Array(kcBuf, K * cc * 4, K * cc);
    this.gated = cfg.gate === "inferred"; const k = Math.max(1, Math.round(cfg.f * K)); this.k = this.gated ? Math.max(1, Math.floor(k / 2)) : k;
    this.mask = [new Uint8Array(K), new Uint8Array(K)]; for (const j of half0) this.mask[0][j] = 1; for (let j = 0; j < K; j++) this.mask[1][j] = 1 - this.mask[0][j];
    this.Wa = new Float64Array(4 * K); this.Wv = new Float64Array(K); this.Ma = new Float64Array(4 * K); this.Mv = new Float64Array(K);
    this.ctx = 0; this.ea = null; this.buf = []; this.rms = null; this.gamma = .95;
  }
  code(h) {                                                          // sparse code: {i: active cell ids, v: rates}
    const K = this.K, cc = this.c.c, pre = new Float64Array(K);
    for (let j = 0; j < K; j++) { let s = 0; for (let q = 0; q < cc; q++) s += h[this.idx[j * cc + q]] * this.w[j * cc + q]; pre[j] = s; }
    const cand = this.gated ? pre.map((v, j) => this.mask[this.ctx][j] ? v : -Infinity) : pre;   // k-WTA within the active context
    const thr = Float64Array.from(cand).sort()[K - this.k], top = [];                               // k-th largest, no full argsort
    for (let j = 0; j < K && top.length < this.k; j++) if (cand[j] >= thr && cand[j] > 0) top.push(j);
    let n = 0; for (const j of top) n += pre[j] * pre[j]; n = Math.sqrt(n) + 1e-6;
    return { i: top, v: top.map(j => pre[j] / n), pre };
  }
  forward(x) {
    const { h1, h } = this.prior.body(x), z = this.code(h), { lg, v } = this.prior.heads(h); let vv = v;
    for (let t = 0; t < z.i.length; t++) { const j = z.i[t], r = z.v[t]; for (let a = 0; a < 4; a++) lg[a] += this.Wa[a * this.K + j] * r; vv += this.Wv[j] * r; }
    if (this.Wb) for (let a = 0; a < 4; a++) for (let s = 0; s < 100; s++) lg[a] += this.Wb[a * 100 + s] * this.c.g * x[150 + s];   // stage 6: place-belief synapses
    if (this.hippo?.k) { const D = this.hippo.drive(x); for (let a = 0; a < 4; a++) lg[a] += this.hippo.k * D[a]; }              // stage 7: goal-map drive
    return { h1, h, z, lg, p: softmax(lg, this.c.tau), v: vv };
  }
  apply(da, dv) {
    const c = this.c;
    for (let q = 0; q < da.length; q++) { const u = c.eta / (1 + c.m * this.Ma[q]) * da[q]; this.Wa[q] += u; this.Ma[q] += Math.abs(u); }
    for (let q = 0; q < dv.length; q++) { const u = c.eta_v / (1 + c.m * this.Mv[q]) * dv[q]; this.Wv[q] += u; this.Mv[q] += Math.abs(u); }
  }
  dopamine(d) {
    if (!this.c.A) return d;
    if (!this.rms) this.rms = new Float64Array(d.length).fill(1);
    return d.map((x, i) => { this.rms[i] = .999 * this.rms[i] + .001 * x * x; return x / Math.max(Math.sqrt(this.rms[i]), 1e-3); });
  }
  dz(z, W) { let s = 0; for (let t = 0; t < z.i.length; t++) s += W[z.i[t]] * z.v[t]; return s; }
  traces(ea, ev, kp, Z, G) {                                         // e = gamma lambda e (kept within an episode) + post x pre
    const n = Z.length, K = this.K, gl = this.gamma * this.c.lam;
    for (let i = 0; i < n; i++) {
      const s = gl * kp[i], A = ea[i], V = ev[i];
      for (let q = 0; q < 4 * K; q++) A[q] *= s; for (let q = 0; q < K; q++) V[q] *= s;
      const z = Z[i]; for (let t = 0; t < z.i.length; t++) { const j = z.i[t], r = z.v[t]; for (let a = 0; a < 4; a++) A[a * K + j] += G[i][a] * r; V[j] += r; }
    }
  }
  learnFrom(ea, ev, delta) {
    const n = ea.length, K = this.K, da = new Float64Array(4 * K), dv = new Float64Array(K);
    for (let i = 0; i < n; i++) { const d = delta[i] / n; if (!d) continue; for (let q = 0; q < 4 * K; q++) da[q] += d * ea[i][q]; for (let q = 0; q < K; q++) dv[q] += d * ev[i][q]; }
    this.apply(da, dv);
  }
  step(F, a, r, done, Fn) {                                          // F, Fn: forward() of each animal before / after the move
    const n = F.length, K = this.K;
    if (!this.ea) { this.ea = Array.from({ length: n }, () => new Float64Array(4 * K)); this.ev = Array.from({ length: n }, () => new Float64Array(K)); this.kp = new Float64Array(n).fill(1); }
    const G = F.map((f, i) => f.p.map((p, q) => (q === a[i] ? 1 : 0) - p)), keep = Array.from(done, d => 1 - d);
    this.traces(this.ea, this.ev, this.kp, F.map(f => f.z), G); this.kp = Float64Array.from(keep);
    const delta = this.dopamine(F.map((f, i) => r[i] + this.gamma * Fn[i].v * keep[i] - f.v)); this.learnFrom(this.ea, this.ev, delta);
    if (this.c.E) {
      this.buf.push(F.map((f, i) => ({ z: f.z, lb: f.p.map((p, q) => this.c.tau * Math.log(Math.max(p, 1e-8)) - this.dz(f.z, this.Wa.subarray(q * K, (q + 1) * K))),
        vb: f.v - this.dz(f.z, this.Wv), a: a[i], r: r[i], keep: keep[i], vn: Fn[i].v })));
      if (this.buf.length === this.c.T) { this.replay(); this.buf = []; }
    }
    return delta;
  }
  replay() {                                                         // awake replay of the last T steps, E times, same rule
    const n = this.buf[0].length, K = this.K;
    for (let e = 0; e < this.c.E; e++) {
      const ea = Array.from({ length: n }, () => new Float64Array(4 * K)), ev = Array.from({ length: n }, () => new Float64Array(K)); let kp = new Float64Array(n).fill(1);
      for (let t = 0; t < this.buf.length; t++) {
        const S = this.buf[t], G = [], delta = [];
        for (let i = 0; i < n; i++) {
          const s = S[i], lg = s.lb.map((l, q) => (l + this.dz(s.z, this.Wa.subarray(q * K, (q + 1) * K))) / this.c.tau), pc = softmax(lg);
          G.push(pc.map((p, q) => (q === s.a ? 1 : 0) - p));
          const v = s.vb + this.dz(s.z, this.Wv), vn = t + 1 < this.buf.length ? this.buf[t + 1][i].vb + this.dz(this.buf[t + 1][i].z, this.Wv) : s.vn;
          delta.push(s.r + this.gamma * vn * s.keep - v);
        }
        this.traces(ea, ev, kp, S.map(s => s.z), G); kp = Float64Array.from(S.map(s => s.keep));
        this.learnFrom(ea, ev, this.dopamine(delta));
      }
      if (!this.Wa.every(isFinite) || !this.Wv.every(isFinite)) return;
    }
  }
}

export class Latent {                                                // cue-free latent-cause switch on episode outcomes (CUSUM)
  constructor() { this.e = [0, 0]; this.n = [0, 0]; this.c = 0; this.S = 0; this.switches = 0; this.dwell = 0; }
  update(success) {
    const c = this.c; this.n[c]++; this.dwell++;
    if (this.S > 0 || (this.n[c] > 20 && this.e[c] >= .25 && this.dwell > 20)) {
      if (this.S === 0) this.ref = Math.min(this.e[c], .95);
      const e = this.ref, q = e / 4; this.S = Math.max(0, this.S + (success ? Math.log(q / e) : Math.log((1 - q) / (1 - e))));
      if (this.S > 12) { this.c = 1 - c; this.S = 0; this.switches++; this.dwell = 0; return; }
    }
    this.e[c] += (+success - this.e[c]) / Math.min(this.n[c], 20);
  }
  clone() { return Object.assign(new Latent(), structuredClone(this)); }
}

// ---------- the registered moved-goal rule ----------
export function movedGoal(walk, target, starts, familiar) {
  const ok = (r, c) => walk[r][c] && !starts.some(s => s[0] === r && s[1] === c) && !(r === familiar[0] && c === familiar[1]) && starts.every(s => bfs(walk, [r, c]).has(s[0] * 10 + s[1]));
  const cells = []; for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) cells.push([Math.abs(r - target[0]) + Math.abs(c - target[1]), r, c]);
  cells.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const f = cells.find(([, r, c]) => ok(r, c)); return f ? [f[1], f[2]] : null;
}

export function prepare(E) { E.atlasF = Float64Array.from(E.atlas); return E; }

// ---------- stage 7: hippocampal goal map (v4n_brainlike7_claude_20261107.Hippo) ----------
// Goal memory G over the 100 place cells (learned at reward, frozen here); per episode: lattice links that a bump depresses,
// an inhibited copy of G at visited empty goals, the wavefront value and a drive (extra steps to the goal) added to the logits.
const NXT = Int32Array.from({ length: 400 }, (_, q) => { const s = q >> 2, [dr, dc] = MOVES[q & 3], r = Math.floor(s / 10) + dr, c = s % 10 + dc;
  return r >= 0 && r < 10 && c >= 0 && c < 10 ? r * 10 + c : s; });
const OPEN0 = Uint8Array.from(NXT, (n, q) => +(n !== q >> 2));
const kern = (c, s) => Float64Array.from({ length: 100 }, (_, x) => Math.exp(-((Math.floor(x / 10) - Math.floor(c / 10)) ** 2 + (x % 10 - c % 10) ** 2) / (2 * s * s)));
function wave(open, Ge, g) {
  let v = Float64Array.from(Ge); if (Math.max(...Ge) <= 0) return v;
  for (let it = 0; it < 100; it++) {
    const nv = new Float64Array(100); let same = true;
    for (let s = 0; s < 100; s++) { let m = 0; for (let a = 0; a < 4; a++) m = Math.max(m, open[s * 4 + a] ? v[NXT[s * 4 + a]] : 0); nv[s] = Math.max(Ge[s], g * m); same &&= nv[s] === v[s]; }
    if (same) break; v = nv;
  }
  return v;
}
export class Hippo {
  constructor(cfg, G) { this.k = cfg.kappa; this.sig = cfg.sigma; this.gam = cfg.gamma_h; this.thr = cfg.thr; this.stp = cfg.stp; this.ior = cfg.ior; this.G = Float64Array.from(G); this.prev = null; this.reset(); }
  reset() { this.open = Uint8Array.from(OPEN0); this.Ge = Float64Array.from(this.G); this.v = wave(this.open, this.Ge, this.gam); }
  drive(x) {
    if (this.prev && this.prev.every((y, i) => y === x[i])) return this.D;                  // same observation again: same drive
    const b = x.subarray(150, 250); let c = 0; for (let s = 1; s < 100; s++) if (b[s] > b[c]) c = s;
    if (x[146] + x[147] + x[148] + x[149] === 0) this.reset();                               // episode start (no previous move)
    else {
      let ch = false; const mx = Math.max(...this.Ge);
      if (this.stp && x[144] > .5) { let a = 0; for (let q = 1; q < 4; q++) if (x[146 + q] > x[146 + a]) a = q; if (this.open[c * 4 + a]) { this.open[c * 4 + a] = 0; ch = true; } }
      if (this.ior && mx > 0 && this.Ge[c] > this.thr * mx) { const K = kern(c, this.sig); this.Ge = this.Ge.map((g, s) => g * (1 - K[s])); ch = true; }
      if (ch) this.v = wave(this.open, this.Ge, this.gam);
    }
    const v = this.v, D = new Float64Array(4);
    if (Math.max(...v) > 0) {
      const q = [0, 1, 2, 3].map(a => { let s = 0; for (let y = 0; y < 100; y++) s += b[y] * (this.open[y * 4 + a] ? v[NXT[y * 4 + a]] : v[y]); return s; }), qm = Math.max(...q);
      for (let a = 0; a < 4; a++) D[a] = Math.max(Math.log(Math.max(q[a], 1e-12) / qm) / -Math.log(this.gam), -10);
    }
    this.prev = Float32Array.from(x); return this.D = D;
  }
}
// learned/bl7_<prior>_<seed>.bin: KC + place -> action synapses (4 x 2148, rows [2048 KC | 100 place]), then the goal map G (100).
export function stage7(prior, kcBuf, cfg, half0, buf) {
  const w = new Float32Array(buf), L = new Circuit(prior, kcBuf, cfg, half0); L.Wb = new Float64Array(400);
  for (let a = 0; a < 4; a++) { L.Wa.set(w.subarray(a * 2148, a * 2148 + 2048), a * 2048); L.Wb.set(w.subarray(a * 2148 + 2048, (a + 1) * 2148), a * 100); }
  L.hippo = new Hippo(cfg, w.subarray(8592)); return L;
}
