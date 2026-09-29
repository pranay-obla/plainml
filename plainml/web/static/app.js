/* plainml web: a small single-page app on top of the JSON API in server.py. No build step. */
"use strict";

// ---------- helpers ----------------------------------------------------------------------------

const $ = (selector, root = document) => root.querySelector(selector);

function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "html") node.innerHTML = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const ICONS = {
  upload: "M12 16V4M7 9l5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3",
  file: "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5M9 13h6M9 17h6",
  target: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2z",
  trend: "M3 17l6-6 4 4 8-8M15 7h6v6",
  groups: "M7 10a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM17 20a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM17 10a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM7 20a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
  alert: "M12 9v4M12 17h.01M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z",
  bars: "M4 20h16M7 16V9M12 16V4M17 16v-5",
  drift: "M3 12h4l3-8 4 16 3-8h4",
  table: "M3 5h18v14H3zM3 10h18M3 15h18M9 5v14",
  sparkle: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 17l.7 2.3L22 20l-2.3.7L19 23l-.7-2.3L16 20l2.3-.7z",
  download: "M12 4v12M7 11l5 5 5-5M4 20h16",
  eye: "M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
  external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  play: "M7 4l13 8-13 8z",
  back: "M15 18l-6-6 6-6",
  x: "M18 6L6 18M6 6l12 12",
  model: "M12 2l9 5v10l-9 5-9-5V7zM12 22V12M21 7l-9 5-9-5",
  sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
  auto: "M12 21a9 9 0 1 0 0-18v18z",
  clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v5l3 2",
  broom: "M19 3l-7 7M12 10l-6 6 2 2 6-6zM4 20l2-2M14 12l4 4-3 5-9-9 5-3z",
  scan: "M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2M7 12h10",
};

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "i");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", ICONS[name] || ICONS.file);
  svg.append(path);
  return svg;
}

function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function num(value) {
  if (value === null || value === undefined || value === "") return "";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (Number.isInteger(n)) return n.toLocaleString();
  const size = Math.abs(n);
  const digits = size >= 1000 ? 0 : size >= 100 ? 1 : size >= 1 ? 2 : 4;
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function duration(seconds) {
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

function toast(message, hint, bad = false) {
  const node = h("div", { class: `toast${bad ? " bad" : ""}`, role: bad ? "alert" : "status" }, message, hint ? h("small", {}, hint) : null);
  $("#toasts").append(node);
  setTimeout(() => node.remove(), bad ? 9000 : 4500);
}

const store = {
  get(key) { try { return sessionStorage.getItem(key); } catch (e) { return null; } },
  set(key, value) { try { value === null ? sessionStorage.removeItem(key) : sessionStorage.setItem(key, value); } catch (e) { /* private mode */ } },
};

// ---------- API ----------------------------------------------------------------------------------

class ApiError extends Error {
  constructor(message, hint, status) { super(message); this.hint = hint; this.status = status; }
}

function apiError(status, body) {
  const detail = body && body.detail;
  if (detail && typeof detail === "object" && !Array.isArray(detail)) return new ApiError(detail.message || "Something went wrong.", detail.hint, status);
  if (Array.isArray(detail)) return new ApiError("The request wasn't valid.", detail.map((d) => d.msg).join("; "), status);
  return new ApiError(typeof detail === "string" ? detail : `Request failed (${status}).`, null, status);
}

// ---------- in-browser mode --------------------------------------------------------------------
// The static site (`plainml web --export`) has no server: worker.js runs plainml with Pyodide in
// this browser, and the bridge sends it the same requests the server would get.

const BROWSER = document.querySelector('meta[name="plainml-mode"]')?.content === "browser";
const bridge = BROWSER ? makeBridge() : null;

function makeBridge() {
  const worker = new Worker("worker.js", { type: "module" });
  const waiting = new Map();
  const jobs = new Map();
  const bootListeners = [];
  let next = 1;
  worker.onmessage = (event) => {
    const message = event.data;
    if (message.kind === "boot") return bootListeners.forEach((listener) => listener(message));
    if (message.kind === "job") return void jobs.set(message.job.id, message.job);
    const pending = waiting.get(message.id);
    if (!pending) return;
    waiting.delete(message.id);
    message.ok ? pending.resolve(message.result) : pending.reject(new ApiError("Python in this page hit an error.", message.error));
  };
  worker.onerror = (event) => {
    for (const pending of waiting.values()) pending.reject(new ApiError("Couldn't start Python in this browser.", event.message));
    waiting.clear();
  };
  const call = (message, transfer = []) => new Promise((resolve, reject) => {
    const id = next++;
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, ...message }, transfer);
  });
  const answer = (reply) => {
    if (reply.status >= 400) throw apiError(reply.status, reply.body);
    return reply.body;
  };
  return {
    onBoot(listener) { bootListeners.push(listener); },
    async request(method, path, json) {
      const job = path.match(/^\/api\/jobs\/([0-9a-f]{12})(?:\?log_from=(\d+))?$/);
      if (method === "GET" && job && jobs.has(job[1])) {  // progress is pushed here while a job runs
        const current = jobs.get(job[1]);
        return { ...current, log: current.log.slice(Number(job[2] || 0)) };
      }
      const body = answer(await call({ kind: "request", method, path, body: json === undefined ? null : JSON.stringify(json) }));
      if (method === "POST" && path === "/api/jobs") jobs.set(body.id, body);
      return body;
    },
    async upload(file) {
      const buffer = await file.arrayBuffer();
      return answer(await call({ kind: "upload", name: file.name, buffer }, [buffer]));
    },
    async read(path) {
      const { data, type } = await call({ kind: "read", path });
      return new Blob([data], { type });
    },
  };
}

async function saveFile(url, name) {
  const blob = await bridge.read(url);
  const link = h("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 60000);
}

async function openFile(url) {
  const tab = window.open("", "_blank"); // opened during the click, so pop-up blockers allow it
  const href = URL.createObjectURL(await bridge.read(url));
  if (tab) tab.location.href = href;
  else location.href = href;
}

async function api(path, options = {}) {
  if (bridge) return bridge.request(options.method || (options.json !== undefined ? "POST" : "GET"), path, options.json);
  const init = { headers: {}, credentials: "same-origin", ...options };
  if (options.json !== undefined) {
    init.method = init.method || "POST";
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.json);
  }
  const response = await fetch(path, init);
  let body = null;
  try { body = await response.json(); } catch (e) { /* empty */ }
  if (response.status === 401 && state.info && state.info.needs_token && path !== "/api/login") {
    state.info.signed_in = false;
    render();
  }
  if (!response.ok) throw apiError(response.status, body);
  return body;
}

function uploadFile(file, onProgress) {
  if (bridge) {
    onProgress(1);
    return bridge.upload(file);
  }
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/uploads");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch (e) { /* empty */ }
      xhr.status < 300 ? resolve(body) : reject(apiError(xhr.status, body));
    };
    xhr.onerror = () => reject(new ApiError("The upload didn't reach the app.", "Is plainml web still running?"));
    xhr.send(form);
  });
}

// ---------- state, theme, routing ------------------------------------------------------------

const state = { info: null, upload: null, task: null, options: {}, models: null, poll: null };

const THEMES = [null, "light", "dark"];
function currentTheme() { try { return localStorage.getItem("plainml-theme"); } catch (e) { return null; } }
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  const button = $("#theme");
  button.replaceChildren(icon(theme === "light" ? "sun" : theme === "dark" ? "moon" : "auto"), theme ? theme[0].toUpperCase() + theme.slice(1) : "Auto");
  for (const frame of document.querySelectorAll("iframe.report-frame")) syncFrame(frame);
}
$("#theme").addEventListener("click", () => {
  const next = THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length];
  try { next ? localStorage.setItem("plainml-theme", next) : localStorage.removeItem("plainml-theme"); } catch (e) { /* ignore */ }
  applyTheme(next);
});

function route() {
  const hash = location.hash.replace(/^#/, "") || "/";
  const [path, query] = hash.split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  return { parts, query: new URLSearchParams(query || "") };
}

function setTab(name) {
  for (const link of document.querySelectorAll("nav.tabs a")) link.classList.toggle("on", link.dataset.tab === name);
}

function render() {
  if (state.poll) { clearTimeout(state.poll); state.poll = null; }
  const view = $("#view");
  view.replaceChildren();
  if (state.info && state.info.needs_token && !state.info.signed_in) { setTab(null); return viewSignIn(view); }
  const { parts, query } = route();
  if (parts[0] === "runs" && parts[1]) { setTab("runs"); return viewRun(view, parts[1]); }
  if (parts[0] === "runs") { setTab("runs"); return viewRuns(view); }
  if (parts[0] === "jobs" && parts[1]) { setTab(null); return viewJob(view, parts[1]); }
  if (parts[0] === "predict") { setTab("predict"); return viewPredict(view, query.get("model")); }
  setTab("new");
  return viewNew(view);
}
window.addEventListener("hashchange", () => { render(); window.scrollTo(0, 0); });

function loading(text = "Loading…") { return h("div", { class: "loading" }, h("span", { class: "spin" }), text); }

function fail(view, error) {
  view.replaceChildren(h("div", { class: "empty" }, h("div", { class: "orb" }, icon("alert")), h("h2", {}, error.message), error.hint ? h("p", { class: "sub" }, error.hint) : null, h("p", {}, h("a", { class: "btn", href: "#/" }, "Start over"))));
}

// ---------- sign in ------------------------------------------------------------------------------

function viewSignIn(view) {
  const input = h("input", { class: "input", type: "password", autocomplete: "current-password", placeholder: "Access token", "aria-label": "Access token" });
  const submit = async (event) => {
    event.preventDefault();
    try {
      await api("/api/login", { json: { token: input.value } });
      state.info.signed_in = true;
      render();
    } catch (error) { toast(error.message, error.hint, true); }
  };
  view.append(h("form", { class: "panel pad signin stack", onsubmit: submit },
    h("h2", {}, "Sign in"),
    h("p", { class: "sub" }, "This plainml app was started with an access token. Ask whoever runs it."),
    input,
    h("button", { class: "btn primary", type: "submit" }, "Continue")));
  input.focus();
}

// ---------- new analysis -------------------------------------------------------------------------

const TASKS = [
  { key: "train", title: "Predict a column", about: "Train and compare models that predict one column from the others.", icon: "target" },
  { key: "forecast", title: "Forecast over time", about: "Project a value forward from its history, with a likely range.", icon: "trend" },
  { key: "cluster", title: "Find groups", about: "Split rows into natural segments and describe what sets each apart.", icon: "groups" },
  { key: "anomaly", title: "Find unusual rows", about: "Score every row and flag the ones that don't fit the pattern.", icon: "alert" },
  { key: "importance", title: "Rank the columns", about: "See which columns matter for a target, compared across many methods.", icon: "bars" },
  { key: "drift", title: "Check for drift", about: "Compare this data with a model's training data or another file.", icon: "drift" },
  { key: "profile", title: "Profile the data", about: "Summarise every column and flag problems before modelling.", icon: "scan" },
  { key: "clean", title: "Clean the data", about: "Fix types, blanks, duplicates and odd values, then download it.", icon: "broom" },
];

const KIND_LABEL = { numeric: "number", categorical: "category", datetime: "date", text: "text", id: "ID", constant: "constant", empty: "empty" };

function dropZone({ compact = false, onFile, label = "Drop a data file here" }) {
  const input = h("input", { type: "file", accept: (state.info?.upload_types || []).join(",") });
  const bar = h("div", { class: "upbar", hidden: true }, h("i"));
  const title = h("strong", {}, label);
  const zone = h("div", { class: `drop${compact ? " compact" : ""}`, role: "button", tabindex: "0", "aria-label": `${label}, or press Enter to browse` },
    h("div", { class: "orb" }, icon("upload")),
    title,
    h("div", { class: "sub" }, "or ", h("span", { style: "color:var(--accent);font-weight:600" }, "browse your files")),
    h("div", { class: "formats" }, `CSV, Excel, Parquet, JSON, TSV · up to ${state.info?.max_upload_mb ?? 500} MB`),
    bar, input);
  let busy = false;
  const take = async (file) => {
    if (!file || busy) return;
    busy = true;
    title.textContent = `Uploading ${file.name}…`;
    bar.hidden = false;
    try {
      const result = await uploadFile(file, (share) => { bar.firstChild.style.width = `${Math.round(share * 100)}%`; if (share >= 1) title.textContent = `Reading ${file.name}…`; });
      await onFile(result);
    } catch (error) {
      toast(error.message, error.hint, true);
      title.textContent = label;
      bar.hidden = true;
    } finally { busy = false; }
  };
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => take(input.files[0]));
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("over"); take(e.dataTransfer.files[0]); });
  return zone;
}

async function restoreUpload() {
  const id = store.get("plainml-upload");
  if (!id || state.upload) return;
  try { state.upload = await api(`/api/uploads/${id}`); } catch (e) { store.set("plainml-upload", null); }
}

async function viewNew(view) {
  view.append(loading());
  await restoreUpload();
  view.replaceChildren();
  if (!state.upload) {
    view.append(
      h("div", { class: "hero" },
        h("div", {},
          h("h1", {}, "What's in your data?"),
          h("p", { class: "lead" }, "Upload a file, choose what you want to find out, and get a clear report, with every result ready to download."),
          bridge ? h("div", { class: "browser-note" }, icon("sparkle"), "Runs entirely in your browser: your data is never uploaded anywhere.") : null),
        dropZone({ onFile: async (summary) => { state.upload = summary; state.task = null; store.set("plainml-upload", summary.id); render(); } })),
      samplesSection(),
    );
    recentRuns(view);
    return;
  }
  view.append(datasetPanel(state.upload));
  const grid = h("div", { class: "tasks", role: "list" });
  const optionsHost = h("div", { class: "section" });
  const pick = (task) => {
    state.task = task.key;
    for (const card of grid.children) card.classList.toggle("on", card.dataset.task === task.key);
    optionsHost.replaceChildren(optionsPanel(task));
    optionsHost.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };
  for (const task of TASKS) {
    grid.append(h("button", { class: "task", type: "button", role: "listitem", "data-task": task.key, onclick: () => pick(task) },
      h("div", { class: "ico" }, icon(task.icon)), h("div", {}, h("b", {}, task.title), h("span", {}, task.about))));
  }
  view.append(h("div", { class: "section" }, h("div", { class: "section-head" }, h("h2", {}, "What do you want to do?")), grid), optionsHost);
  if (state.task) pick(TASKS.find((t) => t.key === state.task));
}

function samplesSection() {
  const samples = state.info.samples || [];
  if (!samples.length) return null;
  const grid = h("div", { class: "tasks samples", role: "list" });
  const use = async (sample, card) => {
    if (grid.classList.contains("busy")) return;
    grid.classList.add("busy");
    card.classList.add("on");
    const ico = card.querySelector(".ico");
    const was = ico.firstChild;
    ico.replaceChildren(h("span", { class: "spin", "aria-label": "Loading" }));
    try {
      const summary = await api(`/api/samples/${encodeURIComponent(sample.key)}`, { method: "POST" });
      state.upload = summary;
      state.task = summary.sample ? summary.sample.task : null;
      store.set("plainml-upload", summary.id);
      render();
    } catch (error) {
      toast(error.message, error.hint, true);
      ico.replaceChildren(was);
      card.classList.remove("on");
      grid.classList.remove("busy");
    }
  };
  for (const sample of samples) {
    const task = TASKS.find((t) => t.key === sample.task) || TASKS[0];
    const card = h("button", { class: "task", type: "button", role: "listitem", title: sample.about },
      h("div", { class: "ico" }, icon(task.icon)),
      h("div", {}, h("b", {}, sample.title), h("span", {}, sample.question), h("span", { class: "meta" }, `${task.title} · ${num(sample.rows)} rows`)));
    card.addEventListener("click", () => use(sample, card));
    grid.append(card);
  }
  return h("div", { class: "section" },
    h("div", { class: "section-head" }, h("div", {}, h("h2", {}, "No data to hand? Try an example"),
      h("p", { class: "sub" }, "Made-up datasets, ready to run. One click loads the data and fills in the settings."))),
    grid);
}

function datasetPanel(upload) {
  const kinds = {};
  for (const column of upload.columns) kinds[column.kind] = (kinds[column.kind] || 0) + 1;
  const replace = h("button", { class: "btn small", type: "button", onclick: () => { state.upload = null; state.task = null; store.set("plainml-upload", null); render(); } }, icon("upload"), "Use another file");
  const columnPills = upload.columns.map((c) => h("span", { class: "pill", title: `${c.name}: ${KIND_LABEL[c.kind] || c.kind}${c.missing_pct ? `, ${c.missing_pct}% blank` : ""}${c.note ? ` (${c.note})` : ""}` },
    h("i", { class: `dot k-${c.kind}` }), h("b", {}, c.name)));
  return h("div", { class: "panel pad" },
    h("div", { class: "dataset" },
      h("div", { class: "file-badge" }, icon("table")),
      h("div", { class: "grow" }, h("div", { class: "name" }, upload.name), h("div", { class: "sub" }, `${num(upload.rows)} rows · ${upload.columns.length} columns · ${bytes(upload.size)}`)),
      replace),
    upload.sample ? h("p", { class: "sample-note" }, icon("sparkle"), h("span", {}, h("b", {}, "Example data. "), upload.sample.about)) : null,
    h("div", { class: "pills", style: "margin-top:16px" }, columnPills),
    h("div", { class: "legend" }, ["numeric", "categorical", "datetime", "text"].filter((k) => kinds[k]).map((k) => h("span", {}, h("i", { class: `dot k-${k}` }), `${KIND_LABEL[k]} (${kinds[k]})`)),
      Object.keys(kinds).some((k) => !["numeric", "categorical", "datetime", "text"].includes(k)) ? h("span", {}, h("i", { class: "dot" }), "not used as input") : null),
    h("details", { class: "more" }, h("summary", {}, `Preview the first ${upload.preview.rows.length} rows`), dataTable(upload.preview)));
}

function dataTable(table, highlight = []) {
  const numeric = table.columns.map((_, i) => table.rows.every((r) => r[i] === null || typeof r[i] === "number"));
  return h("div", {},
    h("div", { class: "table-wrap" },
      h("table", { class: "data" },
        h("thead", {}, h("tr", {}, table.columns.map((c) => h("th", { class: highlight.includes(c) ? "hl" : null, scope: "col" }, c)))),
        h("tbody", {}, table.rows.map((row) => h("tr", {}, row.map((v, i) => h("td", {
          class: [v === null ? "null" : "", numeric[i] ? "num" : "", highlight.includes(table.columns[i]) ? "hl" : ""].join(" ").trim() || null,
          title: v === null ? null : String(v),
        }, v === null ? "—" : typeof v === "number" ? num(v) : String(v)))))))),
    table.total > table.rows.length ? h("div", { class: "table-note" }, `Showing ${table.rows.length} of ${num(table.total)} rows. Download the file for all of them.`) : null);
}

async function recentRuns(view) {
  let runs = [];
  try { runs = await api("/api/runs"); } catch (e) { return; }
  if (!runs.length) return;
  view.append(h("div", { class: "section" },
    h("div", { class: "section-head" }, h("h2", {}, "Recent results"), h("a", { href: "#/runs" }, "All runs →")),
    h("div", { class: "runs" }, runs.slice(0, 4).map(runRow))));
}

// ---------- option forms -------------------------------------------------------------------------

function columnsOf(kinds) {
  return state.upload.columns.filter((c) => !kinds || kinds.includes(c.kind));
}

function field(label, control, hint, wide = false) {
  const id = `f-${Math.random().toString(36).slice(2, 8)}`;
  if (control.matches && control.matches("input,select")) control.id = id;
  return h("div", { class: `field${wide ? " wide" : ""}` },
    control.matches && control.matches("input,select") ? h("label", { for: id }, label) : h("div", { class: "label" }, label),
    control, hint ? h("div", { class: "hint" }, hint) : null);
}

function select(name, choices, value, blank) {
  const node = h("select", { class: "input", name },
    blank !== undefined ? h("option", { value: "" }, blank) : null,
    choices.map((c) => h("option", { value: c.value, selected: String(c.value) === String(value) }, c.label)));
  node.addEventListener("change", () => { state.options[name] = node.value; });
  if (value !== undefined && value !== null) state.options[name] = node.value;
  return node;
}

function columnSelect(name, kinds, value, blank) {
  return select(name, columnsOf(kinds).map((c) => ({ value: c.name, label: `${c.name}  ·  ${KIND_LABEL[c.kind] || c.kind}` })), value, blank);
}

function textInput(name, placeholder, type = "text", value = "") {
  const node = h("input", { class: "input", name, type, placeholder, value, inputmode: type === "number" ? "decimal" : null });
  node.addEventListener("input", () => { state.options[name] = node.value; });
  if (value) state.options[name] = value;
  return node;
}

function segmented(name, choices, value) {
  state.options[name] = value;
  const node = h("div", { class: "seg", role: "radiogroup" });
  for (const c of choices) {
    node.append(h("button", { type: "button", role: "radio", "aria-checked": String(c.value === value), class: c.value === value ? "on" : null, onclick: (e) => {
      state.options[name] = c.value;
      for (const b of node.children) { b.classList.toggle("on", b === e.currentTarget); b.setAttribute("aria-checked", String(b === e.currentTarget)); }
      if (c.onpick) c.onpick();
    } }, c.label));
  }
  return node;
}

function chips(name, choices, selected = []) {
  state.options[name] = [...selected];
  const node = h("div", { class: "chips", role: "group" });
  for (const c of choices) {
    const on = selected.includes(c.value);
    node.append(h("button", { type: "button", class: `chip${on ? " on" : ""}`, "aria-pressed": String(on), title: c.title || null, onclick: (e) => {
      const list = state.options[name];
      const at = list.indexOf(c.value);
      at >= 0 ? list.splice(at, 1) : list.push(c.value);
      e.currentTarget.classList.toggle("on", at < 0);
      e.currentTarget.setAttribute("aria-pressed", String(at < 0));
    } }, c.label));
  }
  return node;
}

function toggle(name, label, about, checked = false) {
  state.options[name] = checked;
  const input = h("input", { type: "checkbox", role: "switch", checked });
  input.addEventListener("change", () => { state.options[name] = input.checked; });
  return h("label", { class: "switch" }, input, h("div", {}, h("b", {}, label), about ? h("span", {}, about) : null));
}

function columnChips(name, kinds, exclude = [], selected = []) {
  const choices = columnsOf(kinds).filter((c) => !exclude.includes(c.name)).map((c) => ({ value: c.name, label: c.name }));
  return chips(name, choices, selected.filter((value) => choices.some((c) => c.value === value)));
}

function advanced(...fields) {
  return h("details", { class: "more field wide" }, h("summary", {}, "More options"), h("div", { class: "form", style: "margin-top:16px" }, fields));
}

function presets(task) {
  const sample = state.upload.sample;
  return sample && sample.task === task ? sample.options : {};
}

function optionsForm(task) {
  const upload = state.upload;
  const preset = presets(task);
  const target = preset.target || upload.suggested_target;
  const predictable = columnsOf(["numeric", "categorical"]);
  switch (task) {
    case "train":
      return [
        field("Column to predict", columnSelect("target", ["numeric", "categorical", "text"], target), "Numbers give a regression; labels give a classification."),
        field("How hard to try", segmented("speed", [
          { value: "quick", label: "Quick" }, { value: "standard", label: "Standard" }, { value: "thorough", label: "Thorough" }], preset.speed || "standard"),
          "Quick tries fast models only. Thorough adds more models" + (state.info.installed.torch ? " (including a neural network)" : "") + " and a stacked ensemble."),
        advanced(
          field("Time limit", textInput("time_budget", "e.g. 5m or 1h"), "Stop starting new models after this long."),
          field("Columns to ignore", columnChips("drop"), "Leave out IDs, leaks, or anything you won't know when predicting.", true),
          toggle("calibrate", "Calibrate probabilities", "So '70% sure' means right 70% of the time."),
          toggle("log_target", "Model the log of the target", "Helps with skewed positive amounts like prices."),
          toggle("private", "Private", "Keep raw data values out of the report and saved files.")),
      ];
    case "forecast": {
      const holidays = state.info.installed.holidays;
      const more = advanced(
        field("Inputs known in advance", columnChips("inputs", ["numeric"], [], preset.inputs || []), "Promotions, prices… Rows after the last known value can hold their planned values.", true),
        field("Public holidays", textInput("country", holidays ? "Country code, e.g. US, GB, IN" : "Needs: pip install holidays", "text", holidays ? preset.country || "" : ""), null),
        field("Combine rows on the same date by", select("agg", [{ value: "sum", label: "Sum" }, { value: "mean", label: "Average" }, { value: "last", label: "Last value" }], "sum")));
      more.open = Boolean(preset.inputs || preset.country);
      return [
        field("Value to forecast", columnSelect("target", ["numeric"], columnsOf(["numeric"]).some((c) => c.name === target) ? target : undefined)),
        field("Date column", columnSelect("date", ["datetime"], undefined, "Detect it for me")),
        field("How far ahead", textInput("horizon", "Automatic", "number", preset.horizon ? String(preset.horizon) : ""), "In periods of the data's frequency (days, weeks, months…)."),
        field("One forecast per", columnSelect("group", ["categorical"], preset.group, "No groups (one series)"), "E.g. a store or product column."),
        more,
      ];
    }
    case "cluster":
      return [
        field("Number of groups", textInput("k", "Find the best number", "number"), "Leave blank to let plainml choose."),
        advanced(
          field("Columns to ignore", columnChips("drop"), null, true),
          toggle("private", "Private", "Keep raw data values out of the report and saved files.")),
      ];
    case "anomaly":
      return [
        field("Expected share of unusual rows (%)", textInput("contamination", "Estimate it for me", "number"), "E.g. 1 for about one row in a hundred."),
        field("Known labels (optional)", columnSelect("label", ["categorical", "numeric"], preset.label, "None"), "A column marking known anomalies, to check the results against."),
        advanced(
          field("Columns to ignore", columnChips("drop"), null, true),
          toggle("private", "Private", "Keep raw data values out of the report and saved files.")),
      ];
    case "importance": {
      const methods = state.info.importance_methods || [];
      return [
        field("Column the others should predict", columnSelect("target", ["numeric", "categorical"], target)),
        field("How many columns to keep", textInput("k", "Decide from the data", "number")),
        field("Methods", chips("methods", methods.map((m) => ({ value: m.key, label: m.label, title: m.about })), methods.filter((m) => m.default).map((m) => m.key)),
          "The defaults balance speed and depth. Hover over a method to see what it does.", true),
        advanced(
          field("Columns to ignore", columnChips("drop"), null, true),
          toggle("private", "Private", "Keep raw data values out of the report and saved files.")),
      ];
    }
    case "drift": {
      const modelField = h("div", { class: "field wide" });
      const fileField = h("div", { class: "field wide", hidden: true });
      const showModels = async () => {
        fileField.hidden = true; modelField.hidden = false;
        state.options.reference_upload = null;
        if (!state.models) state.models = await api("/api/models").catch(() => []);
        const trained = state.models.filter((m) => ["train", "tune"].includes(m.kind));
        modelField.replaceChildren(trained.length
          ? field("Model", select("reference_run", trained.map((m) => ({ value: m.run, label: `${m.target} · ${m.best_model} · ${m.created}` })), trained[0].run), "The new data is compared with the data this model learned from.")
          : h("p", { class: "sub" }, "No trained models yet. Compare with another file instead."));
      };
      const showFile = () => {
        modelField.hidden = true; fileField.hidden = false;
        state.options.reference_run = null;
        fileField.replaceChildren(h("div", { class: "label" }, "Reference file (the \"before\" data)"),
          dropZone({ compact: true, label: "Drop the reference file here", onFile: async (summary) => {
            state.options.reference_upload = summary.id;
            fileField.replaceChildren(h("div", { class: "label" }, "Reference file"), h("span", { class: "pill" }, icon("table"), h("b", {}, summary.name), `${num(summary.rows)} rows`));
          } }));
      };
      setTimeout(showModels);
      return [
        field("Compare with", segmented("reference_mode", [{ value: "model", label: "A trained model", onpick: showModels }, { value: "file", label: "Another file", onpick: showFile }], "model"), null, true),
        modelField, fileField,
        field("Target column (optional)", columnSelect("target", null, undefined, "None"), "Leave it out of the comparison, since it's what gets predicted."),
      ];
    }
    case "profile":
      return [field("Target column (optional)", columnSelect("target", predictable.length ? ["numeric", "categorical"] : null, undefined, "None"), "Adds class balance and leakage checks.")];
    case "clean":
      return [
        field("Target column (optional)", columnSelect("target", null, undefined, "None"), "Rows without a target are dropped; it's never imputed or encoded."),
        field("Outliers", select("outliers", [{ value: "none", label: "Leave them" }, { value: "clip", label: "Clip to a sensible range" }, { value: "remove", label: "Remove those rows" }], "none")),
        toggle("drop_duplicates", "Remove duplicate rows", null, true),
        toggle("impute", "Fill in blanks", "Numbers with the median, categories with the most common value."),
        toggle("encode", "Encode categories as numbers", "One column per category, ready for other tools."),
      ];
    default:
      return [];
  }
}

function optionsPanel(task) {
  state.options = {};
  const form = h("div", { class: "form" }, optionsForm(task.key));
  const run = h("button", { class: "btn primary big", type: "button" }, icon("play"), `Run: ${task.title.toLowerCase()}`);
  run.addEventListener("click", async () => {
    run.disabled = true;
    try {
      const job = await api("/api/jobs", { json: { task: task.key, upload: state.upload.id, options: cleanOptions(state.options) } });
      location.hash = `#/jobs/${job.id}`;
    } catch (error) {
      toast(error.message, error.hint, true);
      run.disabled = false;
    }
  });
  return h("div", { class: "panel pad" },
    h("div", { class: "options-title" }, h("div", { class: "ico" }, icon(task.icon)), h("div", {}, h("h2", {}, task.title), h("div", { class: "sub" }, task.about))),
    form, h("div", { class: "actions" }, run, h("span", { class: "muted" }, state.upload.sample && state.upload.sample.task === task.key
      ? "Filled in for this example: press Run, or change anything first."
      : "Results are saved as a run you can come back to.")));
}

function cleanOptions(options) {
  const out = {};
  for (const [key, value] of Object.entries(options)) {
    if (value === "" || value === null || value === undefined) continue;
    if (Array.isArray(value) && !value.length) continue;
    out[key] = value;
  }
  return out;
}

// ---------- jobs -----------------------------------------------------------------------------------

function viewJob(view, id) {
  const status = h("span", { class: "status" }, h("i", { class: "pulse" }), "Queued");
  const title = h("h1", {}, "Working…");
  const stage = h("div", { class: "stage" }, "Starting");
  const bar = h("div", { class: "progress indeterminate", role: "progressbar", "aria-label": "Progress" }, h("i"));
  const elapsed = h("span", {}, "");
  const percent = h("span", {}, "");
  const log = h("div", { class: "log", role: "log", "aria-live": "off" });
  const logBox = h("details", { class: "more", open: true }, h("summary", {}, "Live log"), log);
  const errorHost = h("div");
  const resultHost = h("div");
  view.append(
    h("div", { class: "crumbs" }, h("a", { href: "#/runs" }, "Runs"), "/", "Job"),
    h("div", { class: "job-head" }, title, status),
    h("div", { class: "panel pad section" }, stage, bar, h("div", { class: "meta-row" }, elapsed, percent), logBox, errorHost),
    resultHost);
  let from = 0;
  const tick = async () => {
    let job;
    try { job = await api(`/api/jobs/${id}?log_from=${from}`); } catch (error) { return fail(view, error); }
    if (route().parts[1] !== id) return;
    title.textContent = job.title;
    stage.textContent = job.status === "failed" ? "Something went wrong" : job.stage;
    elapsed.textContent = job.status === "queued" ? "" : `${duration(job.elapsed)} elapsed`;
    if (job.progress !== null && job.status === "running") {
      bar.classList.remove("indeterminate");
      bar.firstChild.style.width = `${Math.round(job.progress * 100)}%`;
      percent.textContent = `${Math.round(job.progress * 100)}%`;
    }
    for (const line of job.log) {
      const cls = line.startsWith("▸") ? "st" : /^\s/.test(line) ? "dim" : null;
      log.append(h("div", { class: cls }, line || " "));
    }
    if (job.log.length) log.scrollTop = log.scrollHeight;
    from = job.log_size;
    if (job.status === "queued" || job.status === "running") {
      status.replaceChildren(h("i", { class: "pulse" }), job.status === "queued" ? "Queued" : "Running");
      state.poll = setTimeout(tick, 700);
      return;
    }
    bar.classList.remove("indeterminate");
    bar.firstChild.style.width = "100%";
    percent.textContent = "";
    if (job.status === "failed") {
      status.className = "status failed";
      status.textContent = "Failed";
      bar.firstChild.style.background = "var(--bad)";
      errorHost.append(h("div", { class: "error-box" }, h("b", {}, job.error.message), job.error.hint ? h("div", {}, job.error.hint) : null));
      return;
    }
    status.className = "status done";
    status.textContent = `Done in ${duration(job.elapsed)}`;
    logBox.open = false;
    if (job.result.kind === "run") {
      toast(`${job.title}: done in ${duration(job.elapsed)}`);
      location.replace(`#/runs/${encodeURIComponent(job.result.run)}`);
      return;
    }
    resultHost.append(outputResult(job.result));
  };
  tick();
}

function outputResult(result) {
  return h("div", {},
    h("div", { class: "section tiles" }, result.tiles.map(tileNode)),
    result.preview ? h("div", { class: "section" }, h("h2", {}, "Preview"), dataTable(result.preview, result.preview.columns.filter((c) => /^(predicted_|probability_)|^(confidence|cluster|anomaly_score|is_anomaly|forecast|lower_80|upper_80)$/.test(c)))) : null,
    result.report
      ? h("div", { class: "result-grid" }, reportPanel(result.report), h("aside", {}, filesPanel(result.files)))
      : h("div", { class: "files-wide" }, filesPanel(result.files)));
}

// ---------- results -------------------------------------------------------------------------------

function tileNode(tile) {
  return h("div", { class: "tile" }, h("div", { class: "l" }, tile.label), h("div", { class: "v" }, tile.value ?? "—"), tile.sub ? h("div", { class: "s" }, tile.sub) : null);
}

function filesPanel(files) {
  return h("div", { class: "panel" },
    h("div", { class: "panel-head" }, h("h3", {}, "Files"), h("span", { class: "muted", style: "font-size:13px" }, `${files.length} file${files.length === 1 ? "" : "s"}`)),
    h("div", { class: "files" }, files.map(fileRow)));
}

function fileRow(file) {
  const previewHost = h("div", { class: "preview" });
  const actions = h("div", { class: "fa" });
  if (file.preview) {
    const button = h("button", { class: "btn ghost small", type: "button", title: `Preview ${file.name}`, "aria-expanded": "false" }, icon("eye"), "Preview");
    button.addEventListener("click", async () => {
      if (previewHost.childNodes.length) { previewHost.replaceChildren(); button.setAttribute("aria-expanded", "false"); return; }
      previewHost.replaceChildren(loading());
      try {
        previewHost.replaceChildren(dataTable(await api(file.preview)));
        button.setAttribute("aria-expanded", "true");
      } catch (error) { previewHost.replaceChildren(); toast(error.message, error.hint, true); }
    });
    actions.append(button);
  } else if (["html", "md", "txt", "json", "yaml"].includes(file.type)) {
    actions.append(bridge
      ? h("button", { class: "btn ghost small", type: "button", title: `Open ${file.name} in a new tab`, onclick: () => openFile(file.url).catch((e) => toast(e.message, e.hint, true)) }, icon("external"), "Open")
      : h("a", { class: "btn ghost small", href: file.url, target: "_blank", rel: "noopener", title: `Open ${file.name} in a new tab` }, icon("external"), "Open"));
  }
  actions.append(bridge
    ? h("button", { class: "btn small", type: "button", title: `Download ${file.name}`, onclick: () => saveFile(file.download, file.name).catch((e) => toast(e.message, e.hint, true)) }, icon("download"), "Download")
    : h("a", { class: "btn small", href: file.download, download: file.name, title: `Download ${file.name}` }, icon("download"), "Download"));
  return h("div", { class: "file" },
    h("div", { class: `ft ${file.type}` }, file.type.slice(0, 4) || "file"),
    h("div", { class: "t" }, h("b", {}, file.label), file.description ? h("span", {}, file.description) : null, h("span", { class: "fname" }, `${file.name} · ${bytes(file.size)}`)),
    actions, previewHost);
}

function syncFrame(frame) {
  const doc = frame.contentDocument;
  if (!doc || !doc.documentElement) return;
  const theme = currentTheme();
  if (theme) doc.documentElement.dataset.theme = theme;
  else delete doc.documentElement.dataset.theme;
}

function reportPanel(url) {
  const frame = h("iframe", { class: "report-frame", src: bridge ? null : url, title: "Report" });
  if (bridge) bridge.read(url).then((blob) => blob.text()).then((html) => { frame.srcdoc = html; }).catch((e) => toast(e.message, e.hint, true));
  frame.addEventListener("load", () => {
    const doc = frame.contentDocument;
    if (!doc) return;
    doc.head.append(Object.assign(doc.createElement("style"), { textContent: ".theme-toggle,header.top{display:none!important}main{padding-top:4px!important}body{background:transparent!important}" }));
    syncFrame(frame);
    const fit = () => { frame.style.height = `${doc.documentElement.scrollHeight + 4}px`; };
    fit();
    new ResizeObserver(fit).observe(doc.body);
  });
  return h("div", { class: "panel" },
    h("div", { class: "panel-head" }, h("h3", {}, "Report"), bridge
      ? h("button", { class: "btn ghost small", type: "button", onclick: () => openFile(url).catch((e) => toast(e.message, e.hint, true)) }, icon("external"), "Open in a new tab")
      : h("a", { class: "btn ghost small", href: url, target: "_blank", rel: "noopener" }, icon("external"), "Open in a new tab")),
    frame);
}

async function viewRun(view, name) {
  view.append(loading());
  let run;
  try { run = await api(`/api/runs/${encodeURIComponent(name)}`); } catch (error) { return fail(view, error); }
  view.replaceChildren(
    h("div", { class: "crumbs" }, h("a", { href: "#/runs" }, "Runs"), "/", name),
    h("div", { class: "job-head" },
      h("div", {}, h("h1", {}, run.title), h("p", { class: "sub" }, run.subtitle)),
      run.predictable ? h("a", { class: "btn primary", href: `#/predict?model=${encodeURIComponent(name)}` }, icon("sparkle"), run.kind === "forecast" ? "Forecast again" : "Predict with this model") : null),
    h("div", { class: "section tiles" }, run.tiles.map(tileNode)),
    run.sentences.length ? h("div", { class: "section panel pad" }, h("h3", { style: "margin-bottom:6px" }, "Key findings"), h("ul", { class: "findings" }, run.sentences.map((s) => h("li", {}, s)))) : null,
    run.report
      ? h("div", { class: "result-grid" }, reportPanel(run.report), h("aside", {}, filesPanel(run.files)))
      : h("div", { class: "files-wide" }, filesPanel(run.files)));
}

// ---------- runs ----------------------------------------------------------------------------------

const KINDS = {
  train: ["Model", "target"], tune: ["Tuned model", "target"], forecast: ["Forecast", "trend"], cluster: ["Groups", "groups"],
  anomaly: ["Anomalies", "alert"], importance: ["Importance", "bars"], drift: ["Drift", "drift"],
};

function runTitle(run) {
  const [label] = KINDS[run.kind] || [run.kind];
  if (run.kind === "train" || run.kind === "tune") return `Predicting ${run.target}`;
  if (run.kind === "forecast") return `Forecasting ${run.target}`;
  if (run.kind === "importance") return `Columns that matter for ${run.target}`;
  return `${label} in ${run.data || "data"}`;
}

const METRIC_LABELS = {
  f1: "F1", f1_macro: "F1 (macro)", accuracy: "Accuracy", balanced_accuracy: "Balanced accuracy", roc_auc: "ROC-AUC",
  log_loss: "Log loss", precision: "Precision", recall: "Recall", r2: "R²", rmse: "RMSE", mae: "MAE", mape: "MAPE",
  silhouette: "Silhouette", hamming: "Hamming loss",
};

function runRow(run) {
  const [label, iconName] = KINDS[run.kind] || [run.kind, "file"];
  const score = run.score !== null && run.score !== undefined && run.metric ? h("div", { class: "score" }, h("b", {}, num(run.score)), h("span", { class: "m" }, METRIC_LABELS[run.metric] || run.metric)) : h("div", { class: "score" });
  return h("a", { class: "run", href: `#/runs/${encodeURIComponent(run.run)}` },
    h("span", { class: `kind ${run.kind}`, title: label }, icon(iconName)),
    h("div", { style: "min-width:0" }, h("b", {}, runTitle(run)), h("span", { class: "m" }, [label, run.data, run.best_model, run.created].filter(Boolean).join(" · "))),
    score);
}

async function viewRuns(view) {
  view.append(h("h1", {}, "Runs"), h("p", { class: "lead" }, "Everything you've run, newest first. Open one to see its report and download its files."));
  const host = h("div", {}, loading());
  view.append(host);
  let runs;
  try { runs = await api("/api/runs"); } catch (error) { return fail(view, error); }
  if (!runs.length) {
    host.replaceChildren(h("div", { class: "empty panel section" }, h("div", { class: "orb" }, icon("sparkle")), h("h2", {}, "Nothing here yet"), h("p", { class: "sub" }, "Upload a file to run your first analysis."), h("p", {}, h("a", { class: "btn primary", href: "#/" }, icon("upload"), "Upload data"))));
    return;
  }
  let kind = "all";
  let query = "";
  const list = h("div", { class: "runs" });
  const draw = () => {
    const shown = runs.filter((r) => (kind === "all" || r.kind === kind || (kind === "train" && r.kind === "tune")) &&
      (!query || JSON.stringify(r).toLowerCase().includes(query)));
    list.replaceChildren(...(shown.length ? shown.map(runRow) : [h("p", { class: "muted" }, "No runs match.")]));
  };
  const present = new Set(runs.map((r) => (r.kind === "tune" ? "train" : r.kind)));
  const filters = [{ value: "all", label: "All" }, ...Object.entries(KINDS).filter(([k]) => k !== "tune" && present.has(k)).map(([k, [label]]) => ({ value: k, label, onpick: () => { kind = k; draw(); } }))];
  filters[0].onpick = () => { kind = "all"; draw(); };
  const search = h("input", { class: "input", type: "search", placeholder: "Search runs", "aria-label": "Search runs" });
  search.addEventListener("input", () => { query = search.value.toLowerCase(); draw(); });
  host.replaceChildren(h("div", { class: "toolbar" }, segmented("run_kind", filters, "all"), search), list);
  draw();
}

// ---------- predict ---------------------------------------------------------------------------------

async function viewPredict(view, preselected) {
  view.append(h("h1", {}, "Predict"), h("p", { class: "lead" }, "Use a model you've trained on new rows, or continue a forecast."));
  const host = h("div", {}, loading());
  view.append(host);
  let models;
  try { models = state.models = await api("/api/models"); } catch (error) { return fail(view, error); }
  if (!models.length) {
    host.replaceChildren(h("div", { class: "empty panel section" }, h("div", { class: "orb" }, icon("model")), h("h2", {}, "No models yet"), h("p", { class: "sub" }, "Train a model, find groups, find anomalies or forecast first."), h("p", {}, h("a", { class: "btn primary", href: "#/" }, icon("upload"), "Upload data"))));
    return;
  }
  let chosen = models.find((m) => m.run === preselected) || models[0];
  let upload = null;
  const stepTwo = h("div", { class: "section" });
  const cards = h("div", { class: "pick" });
  const drawCards = () => cards.replaceChildren(...models.map((m) => {
    const [label, iconName] = KINDS[m.kind] || [m.kind, "model"];
    return h("button", { type: "button", class: `run${m === chosen ? " on" : ""}`, title: `${runTitle(m)} (${m.run})`, "aria-pressed": String(m === chosen), onclick: () => { chosen = m; drawCards(); drawStepTwo(); } },
      h("span", { class: `kind ${m.kind}` }, icon(iconName)),
      h("div", { style: "min-width:0" }, h("b", {}, runTitle(m)), h("span", { class: "m" }, [label, m.best_model, m.created].filter(Boolean).join(" · "))),
      h("span"));
  }));
  const drawStepTwo = () => {
    state.options = {};
    const forecast = chosen.kind === "forecast";
    const fileHost = h("div");
    const showFile = () => fileHost.replaceChildren(upload
      ? h("div", { class: "dataset panel pad" }, h("div", { class: "file-badge" }, icon("table")), h("div", { class: "grow" }, h("div", { class: "name" }, upload.name), h("div", { class: "sub" }, `${num(upload.rows)} rows · ${upload.columns.length} columns`)),
        h("button", { class: "btn small", type: "button", onclick: () => { upload = null; showFile(); } }, "Change"))
      : dropZone({ compact: true, label: forecast ? "Optional: drop newer history to continue from" : "Drop the rows to predict", onFile: async (summary) => { upload = summary; showFile(); } }));
    showFile();
    const run = h("button", { class: "btn primary big", type: "button" }, icon("play"), forecast ? "Forecast" : "Predict");
    run.addEventListener("click", async () => {
      if (!forecast && !upload) return toast("Upload the rows to predict first.", null, true);
      run.disabled = true;
      try {
        const job = await api("/api/jobs", { json: { task: "predict", upload: upload ? upload.id : null, options: cleanOptions({ ...state.options, model: chosen.run }) } });
        location.hash = `#/jobs/${job.id}`;
      } catch (error) { toast(error.message, error.hint, true); run.disabled = false; }
    });
    stepTwo.replaceChildren(
      h("div", { class: "steps" }, h("b", {}, "2"), forecast ? "Continue the forecast" : "Rows to predict"),
      h("div", { class: "panel pad stack" },
        fileHost,
        forecast ? h("div", { class: "form" }, field("How far ahead", textInput("horizon", "Same as when trained", "number"))) : null,
        chosen.kind === "train" || chosen.kind === "tune" ? toggle("proba", "Add a probability for each class", "For classification models.") : null,
        h("div", { class: "actions" }, run)));
  };
  host.replaceChildren(h("div", { class: "section" }, h("div", { class: "steps" }, h("b", {}, "1"), "Choose a model"), cards), stepTwo);
  drawCards();
  drawStepTwo();
}

// ---------- start -----------------------------------------------------------------------------------

function bootScreen() {
  const stage = h("div", { class: "stage" }, "Starting");
  const bar = h("div", { class: "progress", role: "progressbar", "aria-label": "Setting up" }, h("i"));
  bridge.onBoot(({ stage: text, progress }) => {
    stage.textContent = text;
    bar.firstChild.style.width = `${Math.round(progress * 100)}%`;
  });
  $("#view").replaceChildren(h("div", { class: "boot panel pad" },
    h("h1", {}, "Setting up plainml in your browser"),
    h("p", { class: "lead" }, "Everything runs on this computer, so your data never leaves it. The first visit downloads Python and its data-science libraries (about 50 MB); after that they're cached and start in seconds."),
    stage, bar));
}

(async function start() {
  applyTheme(currentTheme());
  if (bridge) bootScreen();
  try {
    state.info = await api("/api/info");
    $("#version").textContent = `v${state.info.version}${bridge ? " · in your browser" : ""}`;
  } catch (error) {
    return fail($("#view"), bridge
      ? new ApiError("Couldn't start plainml in this browser.", error.hint || error.message)
      : new ApiError("Can't reach the plainml app.", "Start it with: plainml web"));
  }
  render();
})();
