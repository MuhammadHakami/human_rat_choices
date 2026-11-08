// The page at phone width (390 x 844) in headless Chrome: no horizontal overflow, and "Your turn" is playable by taps.
//   bash tools/build_site.sh && python -m http.server -d _site 8791 &  node site/test/mobile.mjs http://127.0.0.1:8791/ OUT_DIR
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
const [url, out = "."] = process.argv.slice(2), port = 9334;
const chrome = spawn("google-chrome-stable", ["--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(tmpdir() + "/cdp")}`, "--window-size=390,844", "about:blank"], { stdio: "ignore" });
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
  await send("Runtime.enable"); await send("Page.enable"); await send("Emulation.setTouchEmulationEnabled", { enabled: true });
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });   // headless windows cannot be narrower than 500 px
  await send("Page.navigate", { url }); await sleep(6000);
  console.log("viewport:", await ev(`innerWidth + "x" + innerHeight`), "page width:", await ev(`document.documentElement.scrollWidth`));
  console.log("overflowing elements:", await ev(`JSON.stringify([...document.querySelectorAll('body *')].filter(e => e.getBoundingClientRect().right > innerWidth + 1 && getComputedStyle(e).visibility !== 'hidden' && !e.closest('.navlinks, pre, select')).slice(0, 8).map(e => e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + (e.className && typeof e.className === 'string' ? '.' + e.className.split(' ')[0] : '') + ' ' + Math.round(e.getBoundingClientRect().right)))`));
  await shot("m0_hero", "#top"); await shot("m1_play", "#play");
  console.log("pad buttons ≥ 44 px:", await ev(`[...document.querySelectorAll('.dpad button')].every(b => b.getBoundingClientRect().width >= 44 && b.getBoundingClientRect().height >= 44)`));
  await shot("m2_you", "#you");
  console.log("three taps:", await ev(`(() => { for (const k of ['k1', 'k3', 'k2']) document.querySelector('.dpad .' + k).click(); return document.querySelector('.dpad .centre').textContent; })()`));
  console.log("shortest path by taps:", await ev(`(async () => { const S = await import(new URL('js/sim.js', location.href).href), Y = window.__arenas.you; Y.act(0); if (Y.state().done) Y.act(0);
    const { ep } = Y.state(), d = S.bfs(ep.room.walkable, ep.goal); let n = 0;
    while (!Y.state().done && n++ < 60) { const p = Y.state().pos, k = d.get(p[0] * 10 + p[1]); Y.act(S.MOVES.findIndex(([dr, dc]) => d.get((p[0] + dr) * 10 + p[1] + dc) === k - 1)); }
    const st = Y.state(); return JSON.stringify({ success: st.result.success, moves: st.result.decisions, shortest: d.get(st.result.spawn[0] * 10 + st.result.spawn[1]), status: document.querySelector('#you-box .status').textContent }); })()`));
  await shot("m3_you_done", "#you");
  await shot("m4_model", "#architecture"); await shot("m5_evidence", "#results");
} finally { console.log("errors:", JSON.stringify(errors)); ws?.close(); chrome.kill(); }
