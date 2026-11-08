// Your turn: one episode of the agents' task, played by the visitor with the agents' information.
// Same renderer, environment and 45-move budget as the agents (sim.js). You see the camera, the bump and reward signals,
// your last move and the place belief (the 252-number input, drawn); the goal is shown only where the agents learned it.
import { Env, MOVE_NAMES, inwardHeading, rng } from "./sim.js";
import { revisits, bumps } from "./protocol.js";
import { fit, drawView, drawRoom, COL } from "./viz.js";
import { h, segment } from "./dom.js";

const COND = {                                                      // the paper's four tests; what the agent learned = what you are shown
  tf: { label: "trained room · familiar goal", stage: "A", goal: "familiar", know: "You trained here: you know the room and the goal." },
  tm: { label: "trained room · goal moved", stage: "A", goal: "moved", know: "You learned the new goal in this room (the old one is faint)." },
  uf: { label: "never-seen room · old goal", stage: "B", goal: "familiar", know: "New walls, goal not shown: only your habits and the agent's place belief." },
  um: { label: "never-seen room · goal moved", stage: "B", goal: "moved", know: "New walls, and this goal was never seen: find it." } };
const KEYS = { arrowdown: 0, arrowup: 1, arrowright: 2, arrowleft: 3, s: 0, w: 1, d: 2, a: 3 };
const PAD = [[1, "▲", "north"], [3, "◀", "west"], [2, "▶", "east"], [0, "▼", "south"]];      // action index, glyph, name
const OPEN = Array.from({ length: 10 }, () => Array(10).fill(1));  // a never-seen room is drawn without its walls

export function playYourself(host, D, { ref, watch }) {
  const E = D.E, u = rng((Date.now() % 1e9) >>> 0), S = { cond: "tf" }, tally = {};
  let env, x, ep, done, result;
  const view = h("canvas", { class: "view" }), map = h("canvas", { class: "room" }), chips = h("div", { class: "chips" }), status = h("div", { class: "status" });
  const know = h("p", { class: "note know" }), score = h("div", { class: "score" }), pad = h("div", { class: "dpad" }), seg = h("div", { class: "seg" }), btn = {};
  PAD.forEach(([a, g, n]) => pad.append(btn[a] = h("button", { class: `k${a}`, "aria-label": n, onclick: () => act(a) }, g)));
  const centre = h("div", { class: "centre" }); pad.append(centre);
  const again = h("button", { class: "btn ghost", onclick: () => newEpisode() }, "↺ new room"), see = h("button", { class: "btn ghost", onclick: () => watch(ep) }, "watch the agents from here →");
  host.append(h("div", { class: "you-l" }, view, chips, pad), h("div", { class: "you-r" }, seg, know, map, status, score, h("div", { class: "row" }, again, see)));
  segment(seg, Object.entries(COND).map(([k, c]) => [k, c.label]), S.cond, v => { S.cond = v; newEpisode(); });

  function newEpisode() {
    const c = COND[S.cond], rooms = E.rooms.filter(r => r.stage === c.stage), room = rooms[Math.floor(u() * rooms.length)], si = Math.floor(u() * E.starts.length), rot = Math.floor(u() * E.rotations.length);
    const goal = c.goal === "moved" ? room.moved_goal : room.familiar_goal;
    ep = { room, goal, cfg: room.configuration, cond: c.goal, start: si, rot };
    env = new Env(E, { walkable: room.walkable, goal }, 1, () => ({ spawn: E.starts[si], rotation_index: rot })); x = env.reset()[0]; done = false; result = null;
    know.textContent = c.know; see.hidden = true; draw();
  }
  function act(a) {
    if (done) return newEpisode();                                   // any move after the reveal starts the next room
    const o = env.step([a]);
    if (o.done[0]) { done = true; result = env.completed.at(-1); const t = tally[S.cond] ??= { n: 0, ok: 0, rv: 0 }; t.n++; t.ok += +result.success; t.rv += revisits(result.path); see.hidden = false; }
    else x = o.obs[0];
    draw();
  }
  function draw() {
    if (!view.width || !ep) return;
    const w = done ? result : env.w[0], c = COND[S.cond], known = c.stage === "A", facing = MOVE_NAMES[w.last ?? inwardHeading(w.spawn)];
    drawView(view, env.frame(w.pos, w.heading, w.rot).img, E.view);
    chips.replaceChildren(h("span", { class: "chip" }, `move ${w.decisions} / ${E.max_decisions}`), h("span", { class: "chip" }, `facing ${facing}`),
      h("span", { class: "chip" + (w.contact ? " hot" : "") }, "bump"), h("span", { class: "chip" + (w.reward ? " hot" : "") }, "reward"));
    PAD.forEach(([a]) => btn[a].classList.toggle("facing", MOVE_NAMES[a] === facing));
    centre.textContent = done ? "↺" : `${w.decisions}/${E.max_decisions}`;
    if (done) drawRoom(map, ep.room.walkable, { goal: ep.goal, alt: c.goal === "moved" ? ep.room.familiar_goal : null, start: w.spawn, path: w.path, pos: w.pos, heading: w.heading, col: COL.human });
    else drawRoom(map, known ? ep.room.walkable : OPEN, { goal: known ? ep.goal : null, alt: known && c.goal === "moved" ? ep.room.familiar_goal : null, starts: known ? E.starts : null, belief: x.subarray(150, 250) });
    status.innerHTML = done ? (result.success ? `<span class="ok">reached the goal in ${result.decisions} moves</span>` : `<span class="bad">time out (${E.max_decisions} moves)</span>`) + ` · revisits ${revisits(result.path)}% · ${bumps(result.path)} wall bumps`
      : `${known ? "your memory of the room" : "walls unknown"} · gold: the agent's place belief, never its true position`;
    const t = tally[S.cond]; score.innerHTML = `<b>${t ? `you, this test: ${t.ok} of ${t.n} reached · revisits ${Math.round(t.rv / t.n)}%` : "you, this test: no episode finished yet"}</b><span>agents (paper): ${ref(ep.room, c.goal)}</span>`;
  }
  const refit = () => { fit(view, view.clientWidth, view.clientWidth * 36 / 64); fit(map, map.clientWidth, map.clientWidth); draw(); };
  requestAnimationFrame(refit); new ResizeObserver(() => requestAnimationFrame(refit)).observe(host);
  let inView = false; new IntersectionObserver(es => { inView = es[0].isIntersecting; }, { threshold: .3 }).observe(host);
  addEventListener("keydown", e => { const a = KEYS[e.key.toLowerCase()]; if (a == null || !inView || e.altKey || e.ctrlKey || e.metaKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return; e.preventDefault(); act(a); });
  return { act, state: () => ({ ep, done, result, pos: (done ? result : env.w[0]).pos }) };   // test handle (site/test/browser.mjs, mobile.mjs)
}
