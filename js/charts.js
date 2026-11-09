// Small SVG charts (no dependencies): bars with 95% CI and per-seed dots, scatter, lines, heatmap. Hover shows values.
const NS = "http://www.w3.org/2000/svg";
const el = (tag, at = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); parent?.appendChild(e); return e; };
const tip = (e, s) => { el("title", {}, e).textContent = s; return e; };
const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
export const mean = v => v.reduce((a, b) => a + b, 0) / v.length;
export const ci95 = v => { if (v.length < 2) return 0; const m = mean(v), s = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)); return 2.045 * s / Math.sqrt(v.length); };

function frame(host, { w = 420, h = 240, ml = 46, mb = 46, mt = 12, mr = 10, ymin = 0, ymax = 100, ylabel = "", xlog = false, xmin, xmax, xlabel = "" }) {
  host.innerHTML = ""; const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, class: "chart", role: "img" }, host);
  const Y = v => mt + (h - mt - mb) * (1 - (v - ymin) / (ymax - ymin));
  const X = xmin == null ? null : v => ml + (w - ml - mr) * (xlog ? (Math.log10(v) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin)) : (v - xmin) / (xmax - xmin));
  const ticks = (a, b) => { const s = (b - a) / 4; const p = 10 ** Math.floor(Math.log10(s)), q = [1, 2, 2.5, 5, 10].find(m => m * p >= s) * p; const t = []; for (let v = Math.ceil(a / q) * q; v <= b + 1e-9; v += q) t.push(+v.toFixed(6)); return t; };
  for (const t of ticks(ymin, ymax)) { el("line", { x1: ml, x2: w - mr, y1: Y(t), y2: Y(t), class: "grid" }, svg); el("text", { x: ml - 5, y: Y(t) + 3, class: "tick", "text-anchor": "end" }, svg).textContent = t; }
  el("text", { x: 11, y: mt + (h - mt - mb) / 2, class: "axis", transform: `rotate(-90 11 ${mt + (h - mt - mb) / 2})`, "text-anchor": "middle" }, svg).textContent = ylabel;
  if (X) {
    const xt = xlog ? Array.from({ length: Math.round(Math.log10(xmax / xmin)) + 1 }, (_, i) => xmin * 10 ** i) : ticks(xmin, xmax);
    for (const t of xt) { el("line", { x1: X(t), x2: X(t), y1: mt, y2: h - mb, class: "grid" }, svg); el("text", { x: X(t), y: h - mb + 13, class: "tick", "text-anchor": "middle" }, svg).textContent = t >= 1000 ? t / 1000 + "k" : t; }
    el("text", { x: ml + (w - ml - mr) / 2, y: h - 8, class: "axis", "text-anchor": "middle" }, svg).textContent = xlabel;
  }
  el("line", { x1: ml, x2: w - mr, y1: h - mb, y2: h - mb, class: "base" }, svg);
  return { svg, X, Y, w, h, ml, mb, mt, mr };
}

// groups: [{label, col, values?: [...], m?, h?, alpha?}], optional ref line
export function bars(host, groups, o = {}) {
  const F = frame(host, o), n = groups.length, bw = (F.w - F.ml - F.mr) / n;
  groups.forEach((gp, i) => {
    const m = gp.m ?? mean(gp.values), h = gp.h ?? (gp.values ? ci95(gp.values) : 0), x = F.ml + bw * i + bw * .18, wd = bw * .64;
    tip(el("rect", { x, y: F.Y(Math.max(m, o.ymin ?? 0)), width: wd, height: Math.abs(F.Y(o.ymin ?? 0) - F.Y(m)), fill: gp.col, opacity: gp.alpha ?? .9, rx: 2 }, F.svg),
      `${gp.label.replace(/\n/g, " ")}: ${fmt(m)} ± ${fmt(h)}${gp.values ? ` (n = ${gp.values.length})` : ""}`);
    (gp.values || []).forEach((v, k) => el("circle", { cx: x + wd / 2 + ((k * 37) % 17 - 8) / 17 * wd * .7, cy: F.Y(v), r: 1.7, class: "dot" }, F.svg));
    el("line", { x1: x + wd / 2, x2: x + wd / 2, y1: F.Y(m - h), y2: F.Y(m + h), class: "err" }, F.svg);
    gp.label.split("\n").forEach((s, k) => { el("text", { x: x + wd / 2, y: F.h - F.mb + 13 + 11 * k, class: "tick", "text-anchor": "middle" }, F.svg).textContent = s; });
  });
  if (o.ref != null) {                                                 // the label sits in the right margin, never over the bars
    el("line", { x1: F.ml, x2: F.w - F.mr, y1: F.Y(o.ref), y2: F.Y(o.ref), class: "ref" }, F.svg);
    (o.refLabel || "").split("\n").forEach((s, k) => { el("text", { x: F.w - F.mr + 4, y: F.Y(o.ref) + 3 + 11 * k, class: "tick", "text-anchor": "start" }, F.svg).textContent = s; }); }
  return F;
}

// series: [{label, col, x: [...], y: [[m, h], ...], dash?}]
export function lines(host, series, o = {}) {
  const F = frame(host, o);
  for (const s of series) {
    const pts = s.x.map((x, i) => [F.X(x), F.Y(s.y[i][0])]);
    const band = s.x.map((x, i) => `${F.X(x)},${F.Y(s.y[i][0] + s.y[i][1])}`).concat(s.x.map((x, i) => `${F.X(x)},${F.Y(s.y[i][0] - s.y[i][1])}`).reverse());
    el("polygon", { points: band.join(" "), fill: s.col, opacity: .13 }, F.svg);
    el("polyline", { points: pts.map(p => p.join(",")).join(" "), fill: "none", stroke: s.col, "stroke-width": 2, "stroke-dasharray": s.dash || "" }, F.svg);
    pts.forEach((p, i) => tip(el("circle", { cx: p[0], cy: p[1], r: s.open ? 3 : 2.6, fill: s.open ? "var(--bg)" : s.col, stroke: s.col, "stroke-width": 1.4 }, F.svg), `${s.label}: ${fmt(s.y[i][0])} ± ${fmt(s.y[i][1])} at ${s.x[i]}`));
  }
  legend(F, series);
  return F;
}

// points: [{x, y, col, label, r?, shape?}]
export function scatter(host, points, o = {}) {
  const F = frame(host, o);
  for (const p of points) {
    const e = p.shape === "diamond" ? el("rect", { x: F.X(p.x) - 4, y: F.Y(p.y) - 4, width: 8, height: 8, transform: `rotate(45 ${F.X(p.x)} ${F.Y(p.y)})`, fill: "var(--bg)", stroke: p.col, "stroke-width": 1.5 }, F.svg)
      : el("circle", { cx: F.X(p.x), cy: F.Y(p.y), r: p.r || 6, fill: p.col, stroke: "var(--bg)", "stroke-width": 1.5 }, F.svg);
    tip(e, `${p.label}: x ${fmt(p.x)}, y ${fmt(p.y)}`);
    if (p.text) { const right = F.X(p.x) > F.ml + .72 * (F.w - F.ml - F.mr);         // near the right edge: label to the left
      el("text", { x: F.X(p.x) + (right ? -9 : 8), y: F.Y(p.y) + (p.dy ?? 3), class: "lab", "text-anchor": right ? "end" : "start" }, F.svg).textContent = p.text; }
  }
  return F;
}

function legend(F, series) {
  series.filter(s => s.label && !s.nolegend).forEach((s, i) => {
    const y = F.mt + 6 + 13 * i; el("line", { x1: F.ml + 8, x2: F.ml + 24, y1: y, y2: y, stroke: s.col, "stroke-width": 2.4, "stroke-dasharray": s.dash || "" }, F.svg);
    el("text", { x: F.ml + 28, y: y + 3.5, class: "tick" }, F.svg).textContent = s.label;
  });
}

// matrix[i][j] in [0, 1]
export function heat(host, M, rows, cols, o = {}) {
  host.innerHTML = ""; const cw = o.cell || 46, ml = o.ml || 96, mt = o.mt || 58, w = ml + cw * cols.length + 8, h = mt + cw * rows.length + 8;
  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, class: "chart heat" }, host);
  cols.forEach((c, j) => el("text", { x: ml + cw * j + cw / 2, y: mt - 8, class: "tick", "text-anchor": "start", transform: `rotate(-35 ${ml + cw * j + cw / 2} ${mt - 8})` }, svg).textContent = c);
  rows.forEach((r, i) => {
    el("text", { x: ml - 6, y: mt + cw * i + cw / 2 + 3, class: "tick", "text-anchor": "end" }, svg).textContent = r;
    cols.forEach((_, j) => { const v = M[i][j]; if (v == null) return;
      tip(el("rect", { x: ml + cw * j, y: mt + cw * i, width: cw - 2, height: cw - 2, rx: 3, fill: `rgba(125,60,152,${.08 + .92 * Math.max(0, Math.min(1, v / (o.max || 1)))})` }, svg), `${r} × ${cols[j]}: ${v.toFixed(o.dp || 2)}`);
      el("text", { x: ml + cw * j + cw / 2 - 1, y: mt + cw * i + cw / 2 + 3, class: "cellv", "text-anchor": "middle", fill: v / (o.max || 1) > .5 ? "#fff" : "var(--fg)" }, svg).textContent = v.toFixed(o.dp || 2); });
  });
}

// Map of representations: points placed so that distance ~ 1 - similarity (classical SMACOF stress majorisation, uniform weights);
// edges drawn for every pair, thicker and darker the more geometry two representations share. pts: [{label, col, open?}], S: similarity matrix.
export function geomap(host, S, pts, o = {}) {
  host.innerHTML = ""; const n = pts.length, w = o.w || 420, h = o.h || 300, svg = el("svg", { viewBox: `0 0 ${w} ${h}`, class: "chart" }, host);
  const D = S.map((r, i) => r.map((s, j) => i === j ? 0 : Math.max(.03, 1 - s)));
  let X = pts.map((_, i) => [Math.cos(2 * Math.PI * i / n + .3), Math.sin(2 * Math.PI * i / n + .3)]);
  for (let it = 0; it < 400; it++) X = X.map((xi, i) => { let ax = 0, ay = 0; for (let j = 0; j < n; j++) if (j !== i) { const dx = xi[0] - X[j][0], dy = xi[1] - X[j][1], d = Math.hypot(dx, dy) || 1e-9, b = D[i][j] / d; ax += b * dx; ay += b * dy; } return [ax / n, ay / n]; });
  if (o.anchor) {                                                      // stable orientation: anchor[0] left, anchor[1] right, anchor[2] above
    const [a, b, c] = o.anchor.map(l => pts.findIndex(p => p.label === l)), ang = Math.atan2(X[b][1] - X[a][1], X[b][0] - X[a][0]);
    X = X.map(([x, y]) => [x * Math.cos(-ang) - y * Math.sin(-ang), x * Math.sin(-ang) + y * Math.cos(-ang)]);
    if (c >= 0 && X[c][1] > (X[a][1] + X[b][1]) / 2) X = X.map(([x, y]) => [x, -y]);
  }
  const xs = X.map(p => p[0]), ys = X.map(p => p[1]), sc = Math.min((w - 150) / (Math.max(...xs) - Math.min(...xs) || 1), (h - 70) / (Math.max(...ys) - Math.min(...ys) || 1));
  const P = X.map(([x, y]) => [75 + (x - Math.min(...xs)) * sc + ((w - 150) - (Math.max(...xs) - Math.min(...xs)) * sc) / 2, 35 + (y - Math.min(...ys)) * sc + ((h - 70) - (Math.max(...ys) - Math.min(...ys)) * sc) / 2]);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const s = S[i][j];
    tip(el("line", { x1: P[i][0], y1: P[i][1], x2: P[j][0], y2: P[j][1], stroke: "var(--bio)", "stroke-width": .5 + 7 * s * s, opacity: .1 + .8 * s }, svg), `${pts[i].label} ↔ ${pts[j].label}: ${s.toFixed(2)}`);
    el("text", { x: (P[i][0] + P[j][0]) / 2, y: (P[i][1] + P[j][1]) / 2 + 3, class: "lab", "text-anchor": "middle", opacity: .9 }, svg).textContent = s.toFixed(2); }
  pts.forEach((p, i) => { tip(el("circle", { cx: P[i][0], cy: P[i][1], r: 8, fill: p.open ? "var(--panel)" : p.col, stroke: p.col, "stroke-width": 2.5 }, svg), p.label);
    el("text", { x: P[i][0], y: P[i][1] - 13, class: "tick", "text-anchor": "middle", "font-weight": 600 }, svg).textContent = p.label; });
}

// Small multiples of horizontal dot rows (one panel per condition, shared x axis). panels: [{title, rows: [{label, pts: [{x, col, open?, shape?}]}]}]
export function dotpanels(host, panels, o = {}) {
  host.innerHTML = ""; const ml = o.ml || 92, pw = o.pw || 176, rh = o.rh || 17, mt = 34, rows = panels[0].rows.length, w = ml + pw * panels.length + 8, h = mt + rh * rows + 26;
  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, class: "chart" }, host), xmin = o.xmin ?? -.05, xmax = o.xmax ?? .15;
  panels[0].rows.forEach((r, i) => el("text", { x: ml - 6, y: mt + rh * i + 12, class: "tick", "text-anchor": "end" }, svg).textContent = r.label);
  panels.forEach((pn, k) => {
    const x0 = ml + pw * k, X = v => x0 + 8 + (pw - 20) * (Math.max(xmin, Math.min(xmax, v)) - xmin) / (xmax - xmin);
    el("text", { x: x0 + pw / 2, y: 12, class: "tick", "text-anchor": "middle", "font-weight": 600 }, svg).textContent = pn.title;
    if (pn.sub) el("text", { x: x0 + pw / 2, y: 24, class: "lab", "text-anchor": "middle" }, svg).textContent = pn.sub;
    el("line", { x1: X(0), x2: X(0), y1: mt - 4, y2: mt + rh * rows, class: "base" }, svg);
    for (const t of (k ? [0, xmax] : [xmin, 0, xmax])) el("text", { x: X(t), y: mt + rh * rows + 12, class: "tick", "text-anchor": t === xmax ? "end" : t === xmin ? "start" : "middle" }, svg).textContent = t;
    pn.rows.forEach((r, i) => { const y = mt + rh * i + 8; el("line", { x1: x0 + 8, x2: x0 + pw - 12, y1: y, y2: y, class: "grid" }, svg);
      if (r.band) el("rect", { x: X(Math.min(...r.band)), y: y - 5, width: Math.max(1, X(Math.max(...r.band)) - X(Math.min(...r.band))), height: 10, fill: r.col || "var(--muted)", opacity: .18 }, svg);
      for (const p of r.pts) { const s = p.shape === "square" ? el("rect", { x: X(p.x) - 4, y: y - 4, width: 8, height: 8, fill: p.col }, svg)
        : p.shape === "triangle" ? el("polygon", { points: `${X(p.x)},${y - 5} ${X(p.x) - 5},${y + 4} ${X(p.x) + 5},${y + 4}`, fill: p.col }, svg)
        : el("circle", { cx: X(p.x), cy: y, r: 4.2, fill: p.open ? "var(--panel)" : p.col, stroke: p.col, "stroke-width": 1.6 }, svg);
        tip(s, `${pn.title} · ${r.label}${p.name ? " " + p.name : ""}: ${p.x.toFixed(3)}`); } });
  });
  if (o.legend) o.legend.forEach((t, i) => el("text", { x: w - 8 - 118 * (o.legend.length - 1 - i), y: h - 2, class: "lab", "text-anchor": "end" }, svg).textContent = t);
}

// Tile sparkline: labelled horizontal bars.
export function minibars(host, rows, max, unit = "") {
  host.innerHTML = ""; const w = 200, rh = 17, svg = el("svg", { viewBox: `0 0 ${w} ${rows.length * rh}`, class: "chart" }, host);
  rows.forEach(([lab, v, col], i) => { const y = i * rh;
    el("text", { x: 0, y: y + 11.5, class: "tick" }, svg).textContent = lab;
    el("rect", { x: 62, y: y + 3, width: w - 100, height: 10, rx: 5, fill: "var(--soft)" }, svg);
    tip(el("rect", { x: 62, y: y + 3, width: Math.max(2, (w - 100) * v / max), height: 10, rx: 5, fill: col }, svg), `${lab}: ${unit ? fmt(v) + unit : v}`);
    el("text", { x: w, y: y + 11.5, class: "tick", "text-anchor": "end" }, svg).textContent = unit ? `${fmt(v)}${unit}` : v; });
}
