// Shared helpers for every page: theme toggle, formatting, escaping, fetching, arm names.

export const REPO = "https://github.com/zeyusuntt/inherit-brain";
export const blob = (path) => `${REPO}/blob/main/${path}`;

export const ARM_ORDER = ["A5", "A0", "A1f", "A0v", "A4", "A4c", "A0f", "A2"];

// Short descriptions (paraphrased from overview.json arms[*].description).
export const ARM_SHORT = {
  A5: "inherit-brain: nine read-only graph tools on FalkorDB in the shared agent loop",
  A0: "HERB's own retrieval tools over the raw records, in the same loop and limits",
  A1f: "The whole product's records in one prompt",
  A0v: "The HERB paper's hybrid RAG: one retrieval, one model call",
  A4: "Typed depth-1 access to entity cards, exact aggregation",
  A4c: "The same cards and tools; the model assembles multi-hop step by step",
  A0f: "The A0 retrieval agent over the entity cards",
  A2: "GraphRAG-SDK 1.4.0 with its automatic ontology, one model call",
};
export const ARM_NAME = {
  A5: "inherit-brain", A0: "HERB agent", A1f: "Whole product", A0v: "HERB RAG",
  A4: "Card tools", A4c: "Card tools, stepwise", A0f: "Card retrieval", A2: "GraphRAG-SDK",
};

export const OUTCOME = {
  path: "answered",
  not_in_graph: "refused (no answer in the records)",
  partial: "partial answer",
  run_failed: "run failed",
  unknown_name: "unknown name",
};

export const TYPE_LABEL = {
  person: "person", company: "company", pr: "PR", url: "URL", content: "content", unanswerable: "unanswerable",
};

const MINUS = "−";

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Record text is shown as recorded, except that em and en dashes are shown as hyphens (site text rule).
export function clean(s) {
  return String(s ?? "").replace(/\s*—\s*/g, " - ").replace(/–/g, "-");
}
export const escClean = (s) => esc(clean(s));

export function f1(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "n/a";
  const s = Number(n).toFixed(1);
  return s.startsWith("-") ? MINUS + s.slice(1) : s;
}
export function s1(n) {
  if (n === null || n === undefined) return "n/a";
  const v = Number(n);
  return (v > 0 ? "+" : v < 0 ? MINUS : "") + Math.abs(v).toFixed(1);
}
export function int(n) {
  if (n === null || n === undefined) return "n/a";
  return Math.round(Number(n)).toLocaleString("en-US");
}
export function ms(n) {
  return `${Number(n).toFixed(1)} ms`;
}

// <span data-v="path" data-f="fmt">value</span>, checked by check.mjs against overview.json.
export function v(path, value, fmt = "f1", cls = "") {
  const text = fmt === "f1" ? f1(value) : fmt === "s1" ? s1(value) : fmt === "int" ? int(value) : String(value);
  return `<span class="num${cls ? " " + cls : ""}" data-v="${esc(path)}" data-f="${fmt}">${esc(text)}</span>`;
}

export function get(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

export async function getJSON(url) {
  const res = await fetch(url, { cache: "force-cache" });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------- theme
function storedTheme() {
  try { return localStorage.getItem("theme"); } catch { return null; }
}
function storeTheme(t) {
  try { localStorage.setItem("theme", t); } catch { /* storage unavailable */ }
}
function currentTheme() {
  const set = document.documentElement.getAttribute("data-theme");
  if (set === "light" || set === "dark") return set;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
export function initTheme() {
  const btn = document.querySelector(".theme-toggle");
  if (!btn) return;
  const paint = () => {
    const t = currentTheme();
    btn.textContent = t === "dark" ? "Light theme" : "Dark theme";
    btn.setAttribute("aria-label", t === "dark" ? "Switch to the light theme" : "Switch to the dark theme");
  };
  const s = storedTheme();
  if (s === "light" || s === "dark") document.documentElement.setAttribute("data-theme", s);
  paint();
  btn.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    storeTheme(next);
    paint();
    document.dispatchEvent(new CustomEvent("themechange"));
  });
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", paint);
  }
}

// Re-render a chart whenever its container changes width.
export function onWidth(el, draw) {
  let last = -1;
  const run = () => {
    const w = Math.floor(el.clientWidth);
    if (w > 0 && w !== last) { last = w; draw(w); }
  };
  if ("ResizeObserver" in window) new ResizeObserver(run).observe(el);
  else window.addEventListener("resize", run);
  run();
}

export const svgNS = "http://www.w3.org/2000/svg";
