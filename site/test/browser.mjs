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
  console.log("hero resets in 12 s:", (await ev(`window.__arenas.hero.resets`)) - r0, "episode caption:", await ev(`document.querySelector('.hero-cap').textContent`));
  await shot("1_play", "#play");
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
  await ev(`document.querySelector('[data-link=code]').href`).then(h => console.log("code link:", h));
} finally { console.log("errors:", JSON.stringify(errors)); ws?.close(); chrome.kill(); }
