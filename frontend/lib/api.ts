// Talking to plainml: either the `plainml web` server over HTTP, or (on the static, in-browser
// site) Python running in a Web Worker (public/worker.js), which answers the same requests.

import type { Job, Upload } from "./types";

export class ApiError extends Error {
  hint?: string | null;
  status?: number;
  constructor(message: string, hint?: string | null, status?: number) {
    super(message);
    this.hint = hint;
    this.status = status;
  }
}

type Detail = { message?: string; hint?: string } | { msg: string }[] | string | undefined;

function apiError(status: number, body: { detail?: Detail } | null): ApiError {
  const detail = body?.detail;
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    return new ApiError(detail.message || "Something went wrong.", detail.hint, status);
  }
  if (Array.isArray(detail)) return new ApiError("The request wasn't valid.", detail.map((d) => d.msg).join("; "), status);
  return new ApiError(typeof detail === "string" ? detail : `Request failed (${status}).`, null, status);
}

// --- which version of the site this is ------------------------------------------------------------

export function isBrowserMode(): boolean {
  if (typeof document === "undefined") return false;
  return document.querySelector('meta[name="plainml-mode"]')?.getAttribute("content") === "browser";
}

// --- the in-browser bridge --------------------------------------------------------------------------

export interface BootState {
  stage: string;
  progress: number;
  ready: boolean;
  error?: string;
}

type Pending = { resolve: (value: never) => void; reject: (error: unknown) => void };

class Bridge {
  private worker: Worker;
  private waiting = new Map<number, Pending>();
  private jobs = new Map<string, Job>();
  private listeners = new Set<(state: BootState) => void>();
  private next = 1;
  boot: BootState = { stage: "Starting", progress: 0, ready: false };

  constructor() {
    this.worker = new Worker("/worker.js", { type: "module" });
    this.worker.onmessage = (event: MessageEvent) => {
      const message = event.data;
      if (message.kind === "boot") {
        this.setBoot({ stage: message.stage, progress: message.progress, ready: message.progress >= 1 });
        return;
      }
      if (message.kind === "job") {
        this.jobs.set(message.job.id, message.job);
        return;
      }
      const pending = this.waiting.get(message.id);
      if (!pending) return;
      this.waiting.delete(message.id);
      if (message.ok) pending.resolve(message.result as never);
      else pending.reject(new ApiError("Python in this page hit an error.", message.error));
    };
    this.worker.onerror = (event) => {
      this.setBoot({ ...this.boot, error: event.message || "Couldn't start Python in this browser." });
      for (const pending of this.waiting.values()) pending.reject(new ApiError("Couldn't start Python in this browser.", event.message));
      this.waiting.clear();
    };
  }

  private setBoot(state: BootState) {
    this.boot = state;
    for (const listener of this.listeners) listener(state);
  }

  onBoot(listener: (state: BootState) => void): () => void {
    this.listeners.add(listener);
    listener(this.boot);
    return () => this.listeners.delete(listener);
  }

  private call<T>(message: Record<string, unknown>, transfer: Transferable[] = []): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const id = this.next++;
      this.waiting.set(id, { resolve: resolve as (value: never) => void, reject });
      this.worker.postMessage({ id, ...message }, transfer);
    });
  }

  private answer<T>(reply: { status: number; body: unknown }): T {
    if (reply.status >= 400) throw apiError(reply.status, reply.body as { detail?: Detail });
    return reply.body as T;
  }

  async request<T>(method: string, path: string, json?: unknown): Promise<T> {
    // While a job runs, Python is busy; its progress is pushed here, so answer from that.
    const job = path.match(/^\/api\/jobs\/([0-9a-f]{12})(?:\?log_from=(\d+))?$/);
    if (method === "GET" && job && this.jobs.has(job[1])) {
      const current = this.jobs.get(job[1])!;
      return { ...current, log: current.log.slice(Number(job[2] || 0)) } as T;
    }
    const body = json === undefined ? null : JSON.stringify(json);
    const reply = await this.call<{ status: number; body: unknown }>({ kind: "request", method, path, body });
    const answer = this.answer<T>(reply);
    if (method === "POST" && path === "/api/jobs") this.jobs.set((answer as Job).id, answer as Job);
    return answer;
  }

  async upload(file: File): Promise<Upload> {
    const buffer = await file.arrayBuffer();
    return this.answer<Upload>(await this.call({ kind: "upload", name: file.name, buffer }, [buffer]));
  }

  async read(path: string): Promise<Blob> {
    const { data, type } = await this.call<{ data: Uint8Array; type: string }>({ kind: "read", path });
    return new Blob([data as BlobPart], { type });
  }
}

let bridge: Bridge | null | undefined;

export function getBridge(): Bridge | null {
  if (bridge === undefined) bridge = isBrowserMode() ? new Bridge() : null;
  return bridge;
}

// --- requests ---------------------------------------------------------------------------------------

let onUnauthorized: (() => void) | null = null;
export function setUnauthorizedHandler(handler: () => void) {
  onUnauthorized = handler;
}

export async function api<T>(path: string, options: { method?: string; json?: unknown } = {}): Promise<T> {
  const method = options.method || (options.json !== undefined ? "POST" : "GET");
  const b = getBridge();
  if (b) return b.request<T>(method, path, options.json);
  const init: RequestInit = { method, credentials: "same-origin", headers: {} };
  if (options.json !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(options.json);
  }
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError("Can't reach the plainml app.", "Is plainml web still running?");
  }
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    /* empty */
  }
  if (response.status === 401 && path !== "/api/login") onUnauthorized?.();
  if (!response.ok) throw apiError(response.status, body as { detail?: Detail });
  return body as T;
}

export function uploadFile(file: File, onProgress: (share: number) => void): Promise<Upload> {
  const b = getBridge();
  if (b) {
    onProgress(1);
    return b.upload(file);
  }
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/uploads");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* empty */
      }
      if (xhr.status < 300) resolve(body);
      else reject(apiError(xhr.status, body));
    };
    xhr.onerror = () => reject(new ApiError("The upload didn't reach the app.", "Is plainml web still running?"));
    xhr.send(form);
  });
}

// --- files ------------------------------------------------------------------------------------------

export async function readFile(url: string): Promise<Blob> {
  const b = getBridge();
  if (b) return b.read(url);
  const response = await fetch(url, { credentials: "same-origin" });
  if (!response.ok) throw new ApiError(`Couldn't open the file (${response.status}).`);
  return response.blob();
}

export async function saveFile(url: string, name: string): Promise<void> {
  const b = getBridge();
  if (!b) {
    const link = Object.assign(document.createElement("a"), { href: url, download: name });
    document.body.append(link);
    link.click();
    link.remove();
    return;
  }
  const href = URL.createObjectURL(await b.read(url));
  const link = Object.assign(document.createElement("a"), { href, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(href), 60_000);
}

export async function openFile(url: string): Promise<void> {
  const b = getBridge();
  if (!b) {
    window.open(url, "_blank", "noopener");
    return;
  }
  const tab = window.open("", "_blank"); // opened during the click, so pop-up blockers allow it
  const href = URL.createObjectURL(await b.read(url));
  if (tab) tab.location.href = href;
  else location.href = href;
}
