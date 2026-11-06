// Canvas drawing: first-person view, top-down room, belief, layer activity, Kenyon-cell grid.
export const COL = { human: "#1b6ca8", rat: "#c0392b", legal_h: "#7fa6c6", legal_r: "#d9928a", sham: "#a0522d", none: "#7a7a7a", bio: "#7d3c98", gold: "#f2b705" };
export const PRIOR_INFO = {
  human_bc: { label: "human choices", short: "human", col: COL.human },
  rat_bc: { label: "rat choices", short: "rat", col: COL.rat },
  human_legal_uniform: { label: "legal moves at human states", short: "legal (H)", col: COL.legal_h },
  rat_legal_uniform: { label: "legal moves at rat states", short: "legal (R)", col: COL.legal_r },
  rat_sham: { label: "rat states, shuffled choices", short: "sham", col: COL.sham },
  cold_ppo: { label: "no demonstrator (PPO)", short: "scratch", col: COL.none },
};
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

export function fit(cv, w, h) {
  const r = devicePixelRatio || 1; cv.width = Math.round(w * r); cv.height = Math.round(h * r);      // CSS keeps the layout width
  const g = cv.getContext("2d"); g.setTransform(r, 0, 0, r, 0, 0); return g;
}

export function drawView(cv, img, view) {                          // 64 x 36 luminance image, nearest-neighbour upscale
  const W = view.width, H = view.height, off = new OffscreenCanvas(W, H), o = off.getContext("2d"), d = o.createImageData(W, H);
  for (let p = 0; p < W * H; p++) { const v = Math.max(0, Math.min(255, Math.round(img[p] * 255))); d.data.set([v, v, v * .97 + 6, 255], p * 4); }
  o.putImageData(d, 0, 0); const g = cv.getContext("2d"); g.imageSmoothingEnabled = false; g.drawImage(off, 0, 0, cv.width, cv.height);
}

export function drawPooled(cv, feat, oh, ow) {
  const g = cv.getContext("2d"), cw = cv.width / ow, ch = cv.height / oh;
  for (let a = 0; a < oh; a++) for (let b = 0; b < ow; b++) { const v = Math.round((feat[a * ow + b] + 1) / 2 * 255); g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(b * cw, a * ch, cw + .5, ch + .5); }
}

// Top-down room. opts: {goal, alt, path, pos, heading, col, belief, occ, occCol, edit, start, starts}
export function drawRoom(cv, walk, o = {}) {
  const g = cv.getContext("2d"), S = cv.width / (g.getTransform().a || 1), c = S / 10, dark = matchMedia("(prefers-color-scheme: dark)").matches && document.documentElement.dataset.theme !== "light";
  g.fillStyle = dark ? "#2b2f35" : "#e4dfd4"; g.fillRect(0, 0, S, S);             // soft grid lines between cells
  for (let r = 0; r < 10; r++) for (let q = 0; q < 10; q++) {
    g.fillStyle = walk[r][q] ? (dark ? "#3a3f47" : "#f4f1ea") : (dark ? "#101214" : "#2b2b2b"); g.fillRect(q * c, r * c, c - 1, c - 1);
    if (o.occ && walk[r][q] && o.occ[r * 10 + q] > 0) { g.globalAlpha = Math.min(.9, o.occ[r * 10 + q] * .9 + .05); g.fillStyle = o.occCol; g.fillRect(q * c, r * c, c - 1, c - 1); g.globalAlpha = 1; }
    if (o.belief) { const b = o.belief[r * 10 + q]; if (b > .01) { g.globalAlpha = Math.min(.85, b * 1.6); g.fillStyle = COL.gold; g.beginPath(); g.arc(q * c + c / 2, r * c + c / 2, c * .42, 0, 7); g.fill(); g.globalAlpha = 1; } }
  }
  g.fillStyle = "#111"; g.fillRect(0, 0, S, 3);                        // the black curtain (distal cue) along row 0
  (o.starts || []).forEach(([r, q]) => { g.strokeStyle = dark ? "#777" : "#bbb"; g.strokeRect(q * c + c * .3, r * c + c * .3, c * .4, c * .4); });
  const star = ([r, q], fill) => { g.save(); g.translate(q * c + c / 2, r * c + c / 2); g.beginPath(); for (let i = 0; i < 10; i++) { const a = i * Math.PI / 5 - Math.PI / 2, rr = i % 2 ? c * .18 : c * .4; g.lineTo(Math.cos(a) * rr, Math.sin(a) * rr); } g.closePath(); g.fillStyle = fill; g.fill(); g.strokeStyle = "#000"; g.lineWidth = .8; g.stroke(); g.restore(); };
  if (o.alt) { g.globalAlpha = .35; star(o.alt, COL.gold); g.globalAlpha = 1; }
  if (o.goal) star(o.goal, COL.gold);
  if (o.start) { g.fillStyle = dark ? "#eee" : "#111"; g.fillRect(o.start[1] * c + c * .28, o.start[0] * c + c * .28, c * .44, c * .44); }
  if (o.path?.length) {
    g.strokeStyle = o.col || "#1b6ca8"; g.lineWidth = Math.max(2, c * .14); g.lineJoin = g.lineCap = "round"; g.globalAlpha = .8; g.beginPath();
    o.path.forEach(([r, q], i) => i ? g.lineTo(q * c + c / 2, r * c + c / 2) : g.moveTo(q * c + c / 2, r * c + c / 2)); g.stroke(); g.globalAlpha = 1;
  }
  if (o.pos) {
    const [r, q] = o.pos, x = q * c + c / 2, y = r * c + c / 2; g.fillStyle = o.col || "#1b6ca8"; g.strokeStyle = "#000"; g.lineWidth = 1;
    g.beginPath(); g.arc(x, y, c * .3, 0, 7); g.fill(); g.stroke();
    if (o.heading) { g.beginPath(); g.moveTo(x + o.heading[1] * c * .5, y + o.heading[0] * c * .5); g.lineTo(x - o.heading[0] * c * .2, y + o.heading[1] * c * .2); g.lineTo(x + o.heading[0] * c * .2, y - o.heading[1] * c * .2); g.closePath(); g.fillStyle = "#fff"; g.fill(); g.stroke(); }
  }
  (o.dots || []).forEach(([r, q], i) => { g.fillStyle = o.dotCol || COL.bio; g.globalAlpha = .75; g.beginPath(); g.arc(q * c + c / 2 + (i % 4 - 1.5) * c * .12, r * c + c / 2 + ((i >> 2) % 4 - 1.5) * c * .12, c * .12, 0, 7); g.fill(); g.globalAlpha = 1; });
}

// Activity strip: values in [-1, 1] as a row of cells (blue negative, orange positive).
export function drawStrip(cv, vals, rows = 1) {
  const g = cv.getContext("2d"), n = vals.length, cols = Math.ceil(n / rows), w = cv.width / cols, h = cv.height / rows;
  g.clearRect(0, 0, cv.width, cv.height);
  for (let i = 0; i < n; i++) { const v = Math.max(-1, Math.min(1, vals[i])); g.fillStyle = v >= 0 ? `rgba(230,126,34,${.08 + .92 * v})` : `rgba(41,128,185,${.08 - .92 * v})`; g.fillRect((i % cols) * w, Math.floor(i / cols) * h, w - .4, h - .4); }
}

// Kenyon cells: K cells on a grid, active ones lit by rate; gated circuits tint the two halves.
export function drawKC(cv, K, z, mask = null, ctx = 0) {
  const g = cv.getContext("2d"), cols = 64, rows = K / cols, w = cv.width / cols, h = cv.height / rows, on = new Map(z.i.map((j, t) => [j, z.v[t]]));
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  g.fillStyle = dark ? "#16181b" : "#f3f3f3"; g.fillRect(0, 0, cv.width, cv.height);
  if (mask) for (let j = 0; j < K; j++) if (mask[ctx][j]) { g.fillStyle = dark ? "rgba(125,60,152,.18)" : "rgba(125,60,152,.10)"; g.fillRect((j % cols) * w, Math.floor(j / cols) * h, w, h); }
  for (const [j, v] of on) { g.fillStyle = `rgba(125,60,152,${.35 + .65 * Math.min(1, v * 4)})`; g.fillRect((j % cols) * w - .5, Math.floor(j / cols) * h - .5, w + 1, h + 1); }
}

// Probability arrows (south, north, east, west) around a centre.
export function drawPolicy(cv, p, chosen, col = "#1b6ca8") {
  const g = cv.getContext("2d"), S = cv.width, c = S / 2; g.clearRect(0, 0, S, S);
  const dir = [[0, 1], [0, -1], [1, 0], [-1, 0]];
  dir.forEach(([dx, dy], a) => { const L = 8 + p[a] * (c - 12); g.strokeStyle = a === chosen ? col : "rgba(128,128,128,.55)"; g.lineWidth = 3 + p[a] * 7; g.lineCap = "round";
    g.beginPath(); g.moveTo(c, c); g.lineTo(c + dx * L, c + dy * L); g.stroke(); });
  g.fillStyle = "#888"; g.beginPath(); g.arc(c, c, 4, 0, 7); g.fill();
}
export { css };
