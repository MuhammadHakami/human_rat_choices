// Tiny DOM helpers shared by the page modules (app.js, you.js).
export const $ = id => document.getElementById(id);
export const h = (tag, at = {}, ...kids) => { const e = document.createElement(tag); for (const k in at) k.startsWith("on") ? e[k] = at[k] : e.setAttribute(k, at[k]); e.append(...kids); return e; };
export const sel = (label, opts, value, onchange) => { const s = h("select", { onchange: () => onchange(s.value) }); for (const [v, t] of opts) s.append(h("option", { value: v }, t)); s.value = value; return h("label", {}, label, s); };
export const segment = (host, opts, value, onchange) => { host.innerHTML = ""; for (const [v, t] of opts) { const b = h("button", { class: v === value ? "on" : "", onclick: () => { host.querySelectorAll("button").forEach(x => x.classList.remove("on")); b.classList.add("on"); onchange(v); } }, t); host.append(b); } onchange(value); };
