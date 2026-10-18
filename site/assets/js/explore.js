// Explorer: questions.json for the list, then one data/q/<product>.json per product on demand.
// Deep links: explore.html#q=<key> (keys contain slashes) and explore.html#random. Back and forward work.
import {
  initTheme, getJSON, esc, escClean, clean, f1, int, ARM_ORDER, ARM_NAME, OUTCOME, TYPE_LABEL,
} from "./common.js";
import { highlightCypher } from "./cypher.js";
import { buildAnswerGraph, renderGraph } from "./graphview.js";

initTheme();

const PER_PAGE = 40;
const ID_TYPES = new Set(["person", "company", "pr", "url"]);
const $ = (id) => document.getElementById(id);

const state = { all: [], byKey: new Map(), filtered: [], page: 0, key: null, f: { product: "", type: "", seen: "", outcome: "", text: "" } };
const productCache = new Map();

// ---------------------------------------------------------------- hash routing
const keyHash = (k) => "#q=" + k.split("/").map(encodeURIComponent).join("/");
function parseHash() {
  const h = location.hash.slice(1);
  if (h === "random") return { random: true };
  if (!h) return {};
  try { return { key: new URLSearchParams(h).get("q") }; } catch { return {}; }
}

let routed = null;
function route() {
  const h = parseHash();
  const sig = location.hash;
  if (sig && sig === routed && !h.random) return;
  routed = sig;
  if (h.random) {
    const pool = state.filtered.length ? state.filtered : state.all;
    const pick = pool[Math.floor(Math.random() * pool.length)];
    history.replaceState(null, "", keyHash(pick.k));
    return show(pick.k);
  }
  if (h.key && state.byKey.has(h.key)) return show(h.key);
  if (h.key) return showMissing(h.key);
  routed = null;
  state.key = null;
  document.body.classList.remove("show-detail");
  renderList();
  renderEmpty();
}

function go(key) {
  if (key === state.key) return;
  location.hash = keyHash(key).slice(1);
}

// ---------------------------------------------------------------- filters
function matches(x) {
  const f = state.f;
  if (f.product && x.p !== f.product) return false;
  if (f.type && x.t !== f.type) return false;
  if (f.seen === "seen" && !x.seen) return false;
  if (f.seen === "unseen" && x.seen) return false;
  const a5 = x.s.A5, a0 = x.s.A0;
  if (f.outcome === "a5c" && a5 !== 100) return false;
  if (f.outcome === "a5w" && !(a5 !== null && a5 < 100)) return false;
  if (f.outcome === "a5b" && !(a5 !== null && a0 !== null && a5 > a0)) return false;
  if (f.outcome === "a0b" && !(a5 !== null && a0 !== null && a0 > a5)) return false;
  if (f.text) {
    const t = f.text.toLowerCase();
    if (!x.q.toLowerCase().includes(t) && !x.k.toLowerCase().includes(t)) return false;
  }
  return true;
}
function applyFilters(keepPage = false) {
  state.filtered = state.all.filter(matches);
  if (!keepPage) state.page = 0;
  if (state.key) {
    const i = state.filtered.findIndex((x) => x.k === state.key);
    if (i >= 0 && !keepPage) state.page = Math.floor(i / PER_PAGE);
  }
  renderList();
}

function bindFilters() {
  $("filters").addEventListener("submit", (e) => e.preventDefault());
  const products = [...new Set(state.all.map((x) => x.p))];
  $("f-product").insertAdjacentHTML("beforeend", products.map((p) => `<option value="${esc(p)}">${esc(p)}</option>`).join(""));
  $("f-product").addEventListener("change", (e) => { state.f.product = e.target.value; applyFilters(); });
  $("f-type").addEventListener("change", (e) => { state.f.type = e.target.value; applyFilters(); });
  $("f-seen").addEventListener("change", (e) => { state.f.seen = e.target.value; applyFilters(); });
  $("f-outcome").addEventListener("change", (e) => { state.f.outcome = e.target.value; applyFilters(); });
  let t;
  $("f-text").addEventListener("input", (e) => {
    clearTimeout(t);
    t = setTimeout(() => { state.f.text = e.target.value.trim(); applyFilters(); }, 120);
  });
  $("random").addEventListener("click", () => {
    const pool = state.filtered.length ? state.filtered : state.all;
    go(pool[Math.floor(Math.random() * pool.length)].k);
  });
  $("prev-page").addEventListener("click", () => { if (state.page > 0) { state.page--; renderList(true); } });
  $("next-page").addEventListener("click", () => {
    if ((state.page + 1) * PER_PAGE < state.filtered.length) { state.page++; renderList(true); }
  });
}
function clearFilters() {
  state.f = { product: "", type: "", seen: "", outcome: "", text: "" };
  for (const id of ["f-product", "f-type", "f-seen", "f-outcome", "f-text"]) $(id).value = "";
  applyFilters();
}

// ---------------------------------------------------------------- list
function dots(s) {
  const label = ARM_ORDER.map((a) => `${a} ${s[a] === null || s[a] === undefined ? "not run" : f1(s[a])}`).join(", ");
  const d = ARM_ORDER.map((a) => {
    const val = s[a];
    if (val === null || val === undefined) return `<i class="dot na"></i>`;
    return `<i class="dot ${a === "A5" ? "a5" : "o"}" style="--v:${val}"></i>${a === "A5" ? `<span class="a5n">${f1(val)}</span>` : ""}`;
  }).join("");
  return `<span class="dots" role="img" aria-label="Scores: ${label}">${d}</span>`;
}

function renderList(scrollTop = false) {
  const n = state.filtered.length;
  $("count").innerHTML = `<b>${int(n)}</b> of ${int(state.all.length)} questions`;
  const pages = Math.max(1, Math.ceil(n / PER_PAGE));
  state.page = Math.min(state.page, pages - 1);
  const start = state.page * PER_PAGE;
  const rows = state.filtered.slice(start, start + PER_PAGE);
  const ul = $("qlist");
  if (!n) {
    ul.innerHTML = `<li><div class="state-msg" style="padding:20px 12px"><p>No question matches these filters.</p><p style="margin-top:10px"><button class="btn" type="button" id="clear-f">Clear filters</button></p></div></li>`;
    $("clear-f").addEventListener("click", clearFilters);
  } else {
    ul.innerHTML = rows.map((x) => `<li><a href="${keyHash(x.k)}"${x.k === state.key ? ' aria-current="true"' : ""}>
      <span class="qmeta"><code>${esc(x.k)}</code><span class="tag">${esc(TYPE_LABEL[x.t] || x.t)}</span>${x.seen ? "" : '<span class="tag">never seen</span>'}</span>
      <span class="qtext">${escClean(x.q)}</span>${dots(x.s)}</a></li>`).join("");
  }
  $("page-info").textContent = n ? `${int(start + 1)} to ${int(Math.min(start + PER_PAGE, n))} of ${int(n)}` : "";
  $("prev-page").disabled = state.page === 0;
  $("next-page").disabled = start + PER_PAGE >= n;
  if (scrollTop) ul.scrollTop = 0;
  const cur = ul.querySelector('[aria-current="true"]');
  if (cur && !scrollTop && ul.scrollHeight > ul.clientHeight) {
    const li = cur.parentElement;
    ul.scrollTop = Math.max(0, li.offsetTop - ul.offsetTop - ul.clientHeight / 2 + li.offsetHeight / 2);
  }
}

// ---------------------------------------------------------------- detail
function renderEmpty() {
  $("detail").innerHTML = `<div class="detail-empty">
    <h2>Pick a question from the list.</h2>
    <p>Each question shows the gold answer, every arm's answer and score, A5's tool calls with the Cypher they ran, and the records A5 cited.</p>
    <p>Two with longer paths: <a href="${keyHash("SupportForce/ans/3")}">SupportForce/ans/3</a> (authors and reviewers of a document) and <a href="${keyHash("SecurityForce/ans/21")}">SecurityForce/ans/21</a> (three calls, one tool error, then the answer).</p>
    <p><a href="#random">Open a random question</a></p></div>`;
}
function showMissing(key) {
  state.key = null;
  document.body.classList.add("show-detail");
  $("detail").innerHTML = `<div class="detail-empty"><h2>No held-out question has the key <code>${esc(key)}</code>.</h2><p>Keys look like <code>SupportForce/ans/3</code>. <a href="${keyHash("SupportForce/ans/3")}">Open that one</a>, or pick from the list.</p><p><button class="btn back-list" type="button">Back to the list</button></p></div>`;
  bindBack();
}
function skeleton() {
  return `<div aria-busy="true"><div class="skel skel-line" style="width:30%"></div><div class="skel" style="height:56px;margin:18px 0;width:85%"></div>
    <div class="skel" style="height:70px;margin-bottom:28px"></div><div class="skel" style="height:140px;margin-bottom:28px"></div>
    <div class="skel" style="height:220px"></div><span class="sr-only">Loading the question</span></div>`;
}

async function loadProduct(p) {
  if (!productCache.has(p)) {
    const pr = getJSON(`data/q/${encodeURIComponent(p)}.json`).then((d) => {
      d.byKey = new Map(d.questions.map((q) => [q.k, q]));
      return d;
    });
    productCache.set(p, pr);
    pr.catch(() => productCache.delete(p));
  }
  return productCache.get(p);
}

async function show(key) {
  state.key = key;
  document.body.classList.add("show-detail");
  const idx = state.filtered.findIndex((x) => x.k === key);
  if (idx >= 0) state.page = Math.floor(idx / PER_PAGE);
  renderList();
  const el = $("detail");
  el.innerHTML = skeleton();
  const row = state.byKey.get(key);
  let prod;
  try {
    prod = await loadProduct(row.p);
  } catch (e) {
    if (state.key !== key) return;
    el.innerHTML = `<div class="detail-empty"><h2>Could not load this product's answers.</h2><p class="state-msg err" style="padding:0 0 12px">${esc(e.message)}</p><p><button class="btn" type="button" id="retry">Try again</button></p></div>`;
    $("retry").addEventListener("click", () => show(key));
    return;
  }
  if (state.key !== key) return;
  const q = prod.byKey.get(key);
  if (!q) { el.innerHTML = `<p class="state-msg err">${esc(key)} is missing from data/q/${esc(row.p)}.json.</p>`; return; }
  el.innerHTML = detailHTML(q, row, prod.evidence);
  bindDetail(el);
  if (matchMedia("(max-width: 960px)").matches) window.scrollTo({ top: 0 });
}

function bindBack() {
  document.querySelectorAll(".back-list").forEach((b) => b.addEventListener("click", () => {
    history.pushState(null, "", location.pathname + location.search);
    route();
  }));
}

function bindDetail(el) {
  bindBack();
  el.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => go(b.dataset.goto)));
  el.querySelectorAll("button.evlink").forEach((b) => b.addEventListener("click", () => {
    const t = document.getElementById(b.dataset.ev);
    if (!t) return;
    el.querySelectorAll(".evidence li.is-target").forEach((x) => { x.classList.remove("is-target"); x.style.boxShadow = ""; x.style.borderColor = ""; });
    t.classList.add("is-target");
    t.style.borderColor = "var(--ink)"; t.style.boxShadow = "0 0 0 1px var(--ink)";
    t.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    t.focus({ preventScroll: true });
  }));
}

const evId = (id) => "ev-" + String(id).replace(/[^A-Za-z0-9_-]/g, (c) => "_" + c.charCodeAt(0).toString(16));

function chip(id, name, mark) {
  const g = mark === "ok" ? "✓" : mark === "bad" ? "×" : "";
  const sr = mark === "ok" ? "in the gold answer" : mark === "bad" ? "not in the gold answer" : "";
  return `<span class="chip${mark ? " " + mark : ""}">${g ? `<span class="g" aria-hidden="true">${g}</span>` : ""}${name ? `<span>${escClean(name)}</span>` : ""}<code>${esc(id)}</code>${sr ? `<span class="sr-only">, ${sr}</span>` : ""}</span>`;
}
function answerChips(arm, q) {
  if (!ID_TYPES.has(q.type) || !arm.answer_ids?.length) return "";
  const gold = new Set(q.gold?.ids || []);
  return `<div class="chips">${arm.answer_ids.map((id, i) => chip(id, arm.answer_names?.[i], gold.has(id) ? "ok" : "bad")).join("")}</div>`;
}
const outcomeText = (o) => OUTCOME[o] || o;

function kvHTML(params) {
  if (params && typeof params === "object") {
    const ent = Object.entries(params);
    if (!ent.length) return `<p class="micro">No parameters.</p>`;
    return `<dl class="kv">${ent.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(typeof v === "string" ? clean(v) : JSON.stringify(v))}</dd>`).join("")}</dl>`;
  }
  return `<dl class="kv"><dt>raw</dt><dd>${esc(String(params))}</dd></dl>`;
}

function rowsHTML(step, evidence) {
  const rows = step.top_rows || [];
  if (!rows.length) return step.error ? "" : `<p class="micro" style="margin-top:6px">No rows returned.</p>`;
  const cols = ["answer", "name", "title", "role", "kind", "group", "via"].filter((c) => rows.some((r) => r[c] !== undefined && r[c] !== null && r[c] !== ""));
  const body = rows.map((r) => `<tr>${cols.map((c) => `<td>${c === "answer" ? `<code>${esc(clean(r[c]))}</code>` : escClean(typeof r[c] === "object" ? JSON.stringify(r[c]) : r[c])}</td>`).join("")}<td>${
    (r.evidence || []).map((id) => evidence[id] ? `<button type="button" class="evlink linklike" data-ev="${evId(id)}">${esc(id)}</button>` : `<code>${esc(id)}</code>`).join("<br>") || '<span class="dim">none</span>'}</td></tr>`).join("");
  return `<div class="rows-wrap"><table class="rows"><thead><tr>${cols.map((c) => `<th scope="col">${esc(c)}</th>`).join("")}<th scope="col">evidence</th></tr></thead><tbody>${body}</tbody></table></div>
    ${step.rows > rows.length ? `<p class="micro" style="margin-top:4px">First ${rows.length} of ${int(step.rows)} rows.</p>` : ""}`;
}

function stepHTML(s, evidence) {
  const err = !!s.error;
  const qs = s.queries || [];
  const queries = qs.length ? `<details class="q-details"${s.n === 1 && qs.length <= 3 ? "" : ""}><summary>${qs.length} Cypher ${qs.length === 1 ? "query" : "queries"}</summary>${qs.map((x, i) => `
      <div class="qblock"><p class="qhead"><span>Query ${i + 1}</span><span><b>${esc(x.command)}</b></span><span><b>${int(x.rows)}</b> rows</span><span><b>${Number(x.ms).toFixed(1)}</b> ms</span></p>
      <pre class="code cypher" tabindex="0" aria-label="Cypher query ${i + 1}">${highlightCypher(x.cypher)}</pre>
      <p class="qhead" style="margin-top:6px"><span>params</span><code style="overflow-wrap:anywhere">${esc(JSON.stringify(x.params))}</code></p></div>`).join("")}</details>`
    : `<p class="micro" style="margin-top:6px">${err ? "Rejected before any query ran." : "No database query was needed."}</p>`;
  return `<li class="step${err ? " err" : ""}"><span class="n" aria-hidden="true">${s.n}</span>
    <div class="step-head"><code class="tool">${esc(s.tool)}</code><span class="sr-only">call ${s.n}</span>
      <span class="m">${err ? "tool error" : `${int(s.rows)} rows`}, ${Number(s.ms).toFixed(1)} ms${s.truncated ? ", truncated" : ""}</span></div>
    ${kvHTML(s.params)}
    ${err ? `<p class="err-msg">The tool rejected the call: ${escClean(s.error)}</p>` : ""}
    ${s.note ? `<p class="note">Tool note: ${escClean(s.note)}</p>` : ""}
    ${rowsHTML(s, evidence)}
    ${queries}</li>`;
}

function graphSection(q) {
  let fig = "";
  try { fig = renderGraph(buildAnswerGraph(q), { title: "The path in the graph" }); } catch (e) { console.error(e); fig = ""; }
  return fig ? `<section class="d-sec" aria-labelledby="h-graph"><h2 id="h-graph">The path in the graph</h2>${fig}</section>` : "";
}

function detailHTML(q, row, evidence) {
  const pos = state.filtered.findIndex((x) => x.k === q.k);
  const prev = pos > 0 ? state.filtered[pos - 1].k : null;
  const next = pos >= 0 && pos < state.filtered.length - 1 ? state.filtered[pos + 1].k : null;
  const product = q.k.split("/")[0];
  const seenLine = q.seen ? "Its question string also appears in the development set." : "Its question string never appeared in development.";

  // gold
  let gold;
  if (q.type === "unanswerable" || !q.gold) gold = `<p>No answer in the records. The correct response is to say so.</p>`;
  else if (q.type === "content") gold = `<p>${escClean(q.gold.text)}</p>`;
  else gold = `<div class="chips">${q.gold.ids.map((id, i) => chip(id, q.gold.names?.[i], "")).join("")}</div>`;
  const scoring = ID_TYPES.has(q.type) ? "Scores are F1 x 100 of the answer ids against these gold ids." : "Scores come from HERB's judge prompt, run on the same model as the arms.";

  // A5
  const a5 = q.arms.A5;
  const d = q.a5?.diagnostics || {};
  const a5HTML = a5 ? `<div class="ans-a5">
      <p class="ans-head"><span class="arm">A5 inherit-brain</span><span class="score">${f1(a5.score)}</span><span class="outc">${esc(outcomeText(a5.outcome))}${d.budget_stop ? ", stopped on the token budget" : ""}${a5.failed ? ", run failed" : ""}</span></p>
      ${a5.answer_text ? `<p class="ans-text">${escClean(a5.answer_text)}</p>` : ""}
      ${answerChips(a5, q)}
    </div>` : `<p class="micro">A5 has no answer for this question.</p>`;

  const others = ARM_ORDER.filter((a) => a !== "A5" && q.arms[a]).map((a) => {
    const r = q.arms[a];
    return `<li><details><summary><span class="arm">${a}</span><span class="score">${f1(r.score)}</span><span class="outc">${esc(outcomeText(r.outcome))}</span><span class="peek">${escClean(r.answer_text || "")}</span></summary>
      <div class="arm-body"><p class="micro" style="margin-bottom:6px">${esc(ARM_NAME[a])}</p>${r.answer_text ? `<p class="ans-text">${escClean(r.answer_text)}</p>` : `<p class="ans-text dim">No answer text.</p>`}${answerChips(r, q)}</div></details></li>`;
  }).join("");
  const notRun = ARM_ORDER.filter((a) => !q.arms[a]);

  // path
  const path = q.a5?.path || [];
  const amb = q.a5?.ops_assignment_ambiguous ? `<p class="flag"><b>Query assignment is approximate for this question.</b> Two consecutive calls of the same tool could not be told apart in the run's flat query log, so a query listed under one of them may belong to its neighbour. The calls, parameters and rows are exact.</p>` : "";
  const pathHTML = path.length ? `${amb}<ol class="steps">${path.map((s) => stepHTML(s, evidence)).join("")}</ol>` : `<p class="small">A5 answered without calling a tool.</p>`;
  const diag = `<p class="diag"><span><b>${int(d.model_calls)}</b> model calls</span><span><b>${int(d.tool_calls)}</b> tool calls</span><span><b>${int(d.input_tokens)}</b> input tokens</span><span><b>${Number(d.latency_s).toFixed(1)}</b> s latency</span></p>`;

  // evidence
  const cited = q.a5?.evidence_ids || [];
  const extra = q.a5?.tool_row_evidence_ids || [];
  const evItems = [...cited.map((id) => [id, true]), ...extra.map((id) => [id, false])].map(([id, isCited]) => {
    const e = evidence[id];
    if (!e || !e.kind) return `<li id="${evId(id)}" tabindex="-1"><p class="ev-head"><code>${esc(id)}</code>${isCited ? '<span class="ev-cited">cited in the answer</span>' : "<span>in a tool row</span>"}</p><p class="ev-text dim">This record id is not in the exported bundle excerpts.</p></li>`;
    return `<li id="${evId(id)}" tabindex="-1"><p class="ev-head"><span class="kind">${esc(e.kind)}</span>${e.title ? `<span>${escClean(e.title)}</span>` : ""}${e.author ? `<span>${escClean(e.author)}</span>` : ""}${e.timestamp ? `<span>${esc(e.timestamp)}</span>` : ""}<code>${esc(id)}</code>${isCited ? '<span class="ev-cited">cited in the answer</span>' : "<span>in a tool row</span>"}</p>
      <p class="ev-text">${escClean(e.text || "")}</p></li>`;
  }).join("");

  return `
    <div class="d-nav"><p class="crumb"><button class="btn back-list" type="button">Back to the list</button> <code>${esc(q.k)}</code></p>
      <div class="btns"><button class="btn" type="button" ${prev ? `data-goto="${esc(prev)}"` : "disabled"}>Previous</button><button class="btn" type="button" ${next ? `data-goto="${esc(next)}"` : "disabled"}>Next</button></div></div>
    <h2 class="d-q">${escClean(q.text)}</h2>
    <p class="d-tags"><span class="tag">${esc(product)}</span><span class="tag">${esc(TYPE_LABEL[q.type] || q.type)} question</span><span>${seenLine}</span></p>

    <section class="d-sec" aria-labelledby="h-gold"><h2 id="h-gold">Gold answer</h2><div class="gold">${gold}</div><p class="micro" style="margin-top:8px">${scoring}</p></section>

    <section class="d-sec" aria-labelledby="h-ans"><h2 id="h-ans">Answers</h2>${a5HTML}
      <ul class="arm-list" aria-label="Other arms">${others}</ul>
      ${notRun.length ? `<p class="micro" style="margin-top:8px">Not run on this question: ${notRun.join(", ")} (product subsets).</p>` : ""}</section>

    ${graphSection(q)}

    <section class="d-sec" aria-labelledby="h-path"><h2 id="h-path">A5's path <span class="dim">${path.length} tool ${path.length === 1 ? "call" : "calls"}, in order</span></h2>${pathHTML}${diag}</section>

    <section class="d-sec" aria-labelledby="h-ev"><h2 id="h-ev">Evidence <span class="dim">${cited.length} cited, ${extra.length} more from tool rows</span></h2>
      ${evItems ? `<ul class="evidence">${evItems}</ul><p class="micro" style="margin-top:8px">Excerpts are cut to 600 characters; dashes in record text are shown as hyphens.</p>` : `<p class="small">A5 cited no records for this answer.</p>`}</section>`;
}

// ---------------------------------------------------------------- boot
async function main() {
  try {
    state.all = await getJSON("data/questions.json");
  } catch (e) {
    $("qlist").innerHTML = `<li><p class="state-msg err" style="padding:16px 12px">Could not load data/questions.json (${esc(e.message)}). Reload the page to try again.</p></li>`;
    $("count").textContent = "No questions loaded";
    $("detail").innerHTML = "";
    return;
  }
  state.byKey = new Map(state.all.map((x) => [x.k, x]));
  state.filtered = state.all.slice();
  $("ex-sub").textContent = `${int(state.all.length)} questions in dataset order. Every arm's answer and score, A5's path through the graph, and the records it cited.`;
  bindFilters();
  window.addEventListener("hashchange", route);
  window.addEventListener("popstate", route);
  route();
}
main();
