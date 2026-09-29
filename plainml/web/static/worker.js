/* plainml in the browser: this Web Worker runs Python (Pyodide) with plainml installed.
   The page (app.js) sends it the same API requests the server would get; plainml.web.browser
   answers them. Runs are kept in the browser's IndexedDB, so they survive a reload.
   Used only by the static site that `plainml web --export` writes. It's a module worker:
   Pyodide is loaded with import(), which works everywhere importScripts may not. */

const PYODIDE_VERSION = "314.0.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const ROOT = "/plainml"; // plainml's runs folder inside the browser
const PACKAGES = [
  "micropip", "numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl",
  "pyyaml", "rich", "click", "lightgbm", "xgboost",
];

let pyodide = null;
let backend = null;
let starting = null;

function tell(stage, progress) {
  postMessage({ kind: "boot", stage, progress });
}

function syncRuns(load) {
  // load=true reads saved runs from IndexedDB; load=false saves the current ones
  return new Promise((resolve) => {
    pyodide.FS.syncfs(load, (error) => {
      if (error) console.warn("plainml: couldn't sync saved runs", error);
      resolve();
    });
  });
}

async function start() {
  tell("Downloading Python", 0.03);
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  pyodide.FS.mkdirTree(ROOT);
  pyodide.FS.mount(pyodide.FS.filesystems.IDBFS, {}, ROOT);
  await syncRuns(true);

  tell("Downloading scikit-learn, pandas and friends", 0.2);
  let loaded = 0;
  await pyodide.loadPackage(PACKAGES, {
    messageCallback: (message) => {
      if (message.startsWith("Loaded")) {
        loaded += 1;
        tell(message, 0.2 + 0.6 * Math.min(1, loaded / PACKAGES.length));
      }
    },
    errorCallback: (message) => console.warn(message),
  });

  tell("Installing plainml", 0.85);
  const manifest = await (await fetch(new URL("manifest.json", self.location.href), { cache: "no-cache" })).json();
  const plainml = manifest.wheel
    ? new URL(manifest.wheel, self.location.href).href
    : `plainml==${manifest.version}`;
  const micropip = pyodide.pyimport("micropip");
  await micropip.install([plainml, "openpyxl"]);
  try {
    await micropip.install("holidays"); // optional: public holidays for forecasting
  } catch (error) {
    console.warn("plainml: holidays not available", error);
  }

  tell("Starting plainml", 0.97);
  pyodide.globals.set("notify", (text) => postMessage({ kind: "job", job: JSON.parse(text) }));
  backend = pyodide.runPython(
    `from plainml.web.browser import Backend\nBackend(${JSON.stringify(ROOT)}, notify)`,
  );
  tell("Ready", 1);
}

function lastLine(error) {
  const text = String((error && error.message) || error);
  const lines = text.trim().split("\n").filter(Boolean);
  return lines[lines.length - 1] || text;
}

self.onmessage = async (event) => {
  const { id, kind } = event.data;
  try {
    starting = starting || start();
    await starting;
    if (kind === "request") {
      const { method, path, body } = event.data;
      const answer = JSON.parse(backend.request(method, path, body));
      postMessage({ id, ok: true, result: answer });
      if (method === "POST" && path === "/api/jobs" && answer.status === 200) {
        backend.run_pending(); // runs now; progress arrives as "job" messages
        await syncRuns(false);
      }
    } else if (kind === "upload") {
      const answer = JSON.parse(backend.upload(event.data.name, new Uint8Array(event.data.buffer)));
      if (answer.status === 200) await syncRuns(false);
      postMessage({ id, ok: true, result: answer });
    } else if (kind === "read") {
      const proxy = backend.read(event.data.path);
      const [data, type] = proxy.toJs();
      proxy.destroy();
      const copy = new Uint8Array(data); // our own copy, safe to hand to the page
      postMessage({ id, ok: true, result: { data: copy, type } }, [copy.buffer]);
    }
  } catch (error) {
    postMessage({ id, ok: false, error: lastLine(error) });
  }
};
