// Drives the built page in headless Chrome over the DevTools protocol: learning lab, probe, wall editing, learned agent.
//   bash tools/build_site.sh && python -m http.server -d _site 8791 &  node site/test/browser.mjs http://127.0.0.1:8791/ OUT_DIR
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
const [url, out = "."] = process.argv.slice(2), port = 9333;
const chrome = spawn("google-chrome-stable", ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(tmpdir() + "/cdp")}`, "--window-size=1400,1000", "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
let ws, id = 0; const pending = new Map(), errors = [];
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async expr => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
const shot = async (name, sel) => { await ev(`document.querySelector(${JSON.stringify(sel)}).scrollIntoView()`); await sleep(400);
  const r = await send("Page.captureScreenshot", { format: "png" }); writeFileSync(`${out}/${name}.png`, Buffer.from(r.result.data, "base64")); };
try {
  for (let k = 0; k < 40; k++) { try { const t = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); const p = t.find(x => x.type === "page"); if (p) { ws = new WebSocket(p.webSocketDebuggerUrl); break; } } catch {} await sleep(250); }
  await new Promise(r => ws.onopen = r);
  ws.onmessage = m => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)({ result: d.result }); pending.delete(d.id); }
    if (d.method === "Runtime.exceptionThrown") errors.push(d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text);
    if (d.method === "Runtime.consoleAPICalled" && d.params.type === "error") errors.push(d.params.args.map(a => a.value ?? a.description).join(" ")); };
  await send("Runtime.enable"); await send("Page.enable"); await send("Page.navigate", { url }); await sleep(5000);
  console.log("hero cards:", await ev(`document.querySelectorAll('#hero .agent').length`), "play cards:", await ev(`document.querySelectorAll('#play-agents .agent').length`));
  await shot("0_hero", "#top");
  const r0 = await ev(`window.__arenas.hero.resets`); await sleep(12000);
  console.log("play tick (4 agents + model panel):", await ev(`(() => { const A = window.__arenas.play; const t0 = performance.now(); let n = 0; for (; n < 120; n++) { if (!A.tick()) A.reset(); } return ((performance.now() - t0) / n).toFixed(2) + " ms"; })()`));
  console.log("hero after the curated 8:", await ev(`(() => { const H = window.__arenas.hero; const caps = []; for (let k = 0; k < 3; k++) { H.trial = 8 + k; H.nextEpisode(); caps.push(document.querySelector('.hero-cap').textContent.split(' · ').slice(0, 5).join(' · ')); } H.trial = 1; H.reset(); return caps.join(' | '); })()`));
  console.log("play auto-cycle:", await ev(`(async () => { const P = window.__arenas.play, r0 = P.resets; P.agents.forEach(A => { A.done = true; A.result = A.env.w[0]; }); P.tick();
    await new Promise(r => setTimeout(r, 3500)); return \`resets +\${P.resets - r0}, now seed \${P.s.seed}, \${D_room()}\`; function D_room() { return document.querySelector('#play-controls label select').selectedOptions[0].textContent + ', goal ' + P.s.goal + ', start ' + P.s.start; } })()`));
  console.log("hero resets in 12 s:", (await ev(`window.__arenas.hero.resets`)) - r0, "episode caption:", await ev(`document.querySelector('.hero-cap').textContent`));
  await sleep(15000); console.log("rates:", await ev(`[...document.querySelectorAll('#play-agents .agent')].map(c => c.querySelector('h3').textContent + ': ' + c.querySelector('.rate').innerText.replace(/\\n/g, ' | ')).join(' || ')`));
  await shot("1_play", "#play");
  await ev(`window.__arenas.play.s.cycle = false; document.querySelector('#play .pill[data-k=goalmap]').click()`); await sleep(20000);   // stage-7 goal map, never-seen room (no auto-cycle)
  console.log("goal map rates:", await ev(`[...document.querySelectorAll('#play-agents .agent')].map(c => c.querySelector('h3').textContent + ': ' + c.querySelector('.rate').innerText.replace(/\\n/g, ' | ')).join(' || ')`));
  await shot("1b_goalmap", "#play");
  await ev(`[...document.querySelectorAll('#play button')].find(b => b.textContent.includes('probe')).click()`); await sleep(6000); await shot("2_probe", "#play-probe");
  console.log("probe note:", await ev(`document.getElementById('play-probe-note').textContent`));
  await ev(`[...document.querySelectorAll('#learn-controls button')].find(b => b.textContent.includes('learn')).click()`);
  await sleep(9000); await shot("3_learning", "#learn");
  for (let k = 0; k < 90 && (await ev(`document.querySelector('#learn-controls button[disabled]') !== null`)); k++) await sleep(1000);
  await sleep(1500); await shot("4_learned", "#learn");
  console.log("old goal kept (ungated, gated):", await ev(`[...document.querySelectorAll('.lane .big')].map(e => e.textContent).join(", ")`));
  console.log("gate badge:", await ev(`document.querySelectorAll('.ctxbadge')[1].textContent`));
  await ev(`[...document.querySelectorAll('#learn-controls button')].find(b => b.textContent.includes('learned agent')).click()`); await sleep(5000); await shot("5_play_learned", "#play-agents");
  console.log("agent 1:", await ev(`document.querySelector('#play-agents .agent h3').textContent`));
  await ev(`[...document.querySelectorAll('#play button')].find(b => b.textContent.includes('edit walls')).click()`);
  await ev(`(() => { const c = document.querySelectorAll('#play-agents .room')[1], b = c.getBoundingClientRect(); c.dispatchEvent(new MouseEvent('click', { clientX: b.left + b.width * .45, clientY: b.top + b.height * .45 })); })()`);
  await sleep(2500); await shot("6_edited", "#play-agents");
  await shot("7_you", "#you");                                        // Your turn: keys move, the reveal shows the room, the hand-off reaches Play
  console.log("your turn, keys:", await ev(`(() => { for (const key of ['ArrowUp', 'a', 'ArrowRight']) dispatchEvent(new KeyboardEvent('keydown', { key })); return document.querySelector('.dpad .centre').textContent + ' · ' + document.querySelector('#you-box .know').textContent; })()`));
  console.log("your turn, never-seen room:", await ev(`(async () => { document.querySelectorAll('#you-box .seg button')[3].click(); const S = await import(new URL('js/sim.js', location.href).href), Y = window.__arenas.you, { ep } = Y.state(), d = S.bfs(ep.room.walkable, ep.goal); let n = 0;
    while (!Y.state().done && n++ < 60) { const p = Y.state().pos, k = d.get(p[0] * 10 + p[1]); Y.act(S.MOVES.findIndex(([dr, dc]) => d.get((p[0] + dr) * 10 + p[1] + dc) === k - 1)); }
    const st = Y.state(); return JSON.stringify({ stage: ep.room.stage, success: st.result.success, moves: st.result.decisions, score: document.querySelector('#you-box .score').innerText.replace(/\\n/g, ' | ') }); })()`));
  await shot("7b_you_done", "#you");
  await ev(`[...document.querySelectorAll('#you-box .row button')].find(b => b.textContent.includes('watch')).click()`); await sleep(2500);
  console.log("hand-off to Play:", await ev(`(() => { const P = window.__arenas.play, Y = window.__arenas.you.state().ep; return JSON.stringify({ sameRoom: P.room().configuration === Y.cfg, goal: P.s.goal === Y.cond, start: P.s.start === Y.start, hash: location.hash }); })()`));
  await ev(`document.querySelector('[data-link=code]').href`).then(h => console.log("code link:", h));
} finally { console.log("errors:", JSON.stringify(errors)); ws?.close(); chrome.kill(); }
