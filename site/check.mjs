// Site checks: serves site/ on a local port and drives Chromium with Playwright.
//
//   npm i playwright@1.63.0 (in any folder), npx playwright install chromium
//   NODE_PATH=<that folder>/node_modules node site/check.mjs
//
// For index, explore and whatif at 1440x900 and 390x844, light and dark: no console error or failed request, no
// U+2014 / U+2013 in visible text or attributes, no horizontal page scroll. Every [data-v] number is compared with
// data/overview.json, every [data-calc] literal with the source it came from (questions.json, q/*.json, and the
// repository's results/freeze/load_report.json and results/receipts). Screenshots go to site/.shots/.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require("playwright"));
} catch {
  console.error("playwright not found: install it (npm i playwright@1.63.0) and run with NODE_PATH=<dir>/node_modules");
  process.exit(2);
}

const SITE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(SITE, "..");
const SHOTS = path.join(SITE, ".shots");
fs.mkdirSync(SHOTS, { recursive: true });

const readJSON = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
const ov = readJSON(path.join(SITE, "data/overview.json"));
const questions = readJSON(path.join(SITE, "data/questions.json"));

// ---------------------------------------------------------------- expected values, computed here independently
const MINUS = "−";
const fmt = {
  f1: (n) => { const s = Number(n).toFixed(1); return s.startsWith("-") ? MINUS + s.slice(1) : s; },
  s1: (n) => (n > 0 ? "+" : n < 0 ? MINUS : "") + Math.abs(Number(n)).toFixed(1),
  int: (n) => Math.round(Number(n)).toLocaleString("en-US"),
  raw: (x) => (x === null || x === undefined ? "n/a" : String(x)),
  usd: (n) => String(n),
  utc: (s) => {
    const t = new Date(s); const p = (n) => String(n).padStart(2, "0");
    return `${t.getUTCFullYear()}-${p(t.getUTCMonth() + 1)}-${p(t.getUTCDate())}, ${p(t.getUTCHours())}:${p(t.getUTCMinutes())}:${p(t.getUTCSeconds())} UTC`;
  },
};
const lookup = {
  ...ov,
  hyp: Object.fromEntries(ov.hypotheses.map((h) => [h.id, h])),
  sec: Object.fromEntries(ov.secondary_contrasts.map((c) => [c.contrast, c])),
};
const get = (p) => p.split(".").reduce((o, k) => (o == null ? undefined : o[k]), lookup);

function computeCalc() {
  const c = {};
  const lr = readJSON(path.join(REPO, "results/freeze/load_report.json"))["6.0"];
  const nodes = {}, edges = {};
  for (const prod of Object.values(lr)) {
    for (const [k, n] of Object.entries(prod.nodes)) nodes[k] = (nodes[k] || 0) + n;
    for (const [k, n] of Object.entries(prod.edges)) edges[k] = (edges[k] || 0) + n;
  }
  c["graph.nodes"] = fmt.int(Object.values(nodes).reduce((a, b) => a + b, 0));
  c["graph.edges"] = fmt.int(Object.values(edges).reduce((a, b) => a + b, 0));
  c["graph.products"] = String(Object.keys(lr).length);
  c["graph.labels"] = String(Object.keys(nodes).length);
  c["graph.types"] = String(Object.keys(edges).length);
  for (const [k, n] of Object.entries(edges)) c[`edge.${k}`] = fmt.int(n);
  for (const [k, n] of Object.entries(nodes)) c[`node.${k}`] = fmt.int(n);

  const tools = {}; let total = 0, ro = 0, ft = 0, st = 0, bw = 0, vec = 0;
  for (const f of fs.readdirSync(path.join(SITE, "data/q"))) {
    const d = readJSON(path.join(SITE, "data/q", f));
    for (const q of d.questions) for (const s of q.a5.path) {
      tools[s.tool] = (tools[s.tool] || 0) + 1;
      for (const x of s.queries) {
        total++;
        if (x.command === "GRAPH.RO_QUERY") ro++;
        if (x.cypher.includes("db.idx.fulltext")) ft++;
        if (x.cypher.includes("herb_v1.statusAt")) st++;
        if (x.cypher.includes("algo.betweenness")) bw++;
        if (x.cypher.includes("db.idx.vector")) vec++;
      }
    }
  }
  for (const t of ["resolve_person", "resolve_entity", "doc_people", "bug_board", "fix_trail", "feature_fate", "who_said", "pr_board", "bridges"]) {
    c[`tool.${t}`] = fmt.int(tools[t] || 0);
  }
  Object.assign(c, { "cypher.total": fmt.int(total), "cypher.ro": fmt.int(ro), "cypher.fulltext": fmt.int(ft),
    "cypher.statusAt": fmt.int(st), "cypher.betweenness": fmt.int(bw), "cypher.vector": fmt.int(vec) });

  const cnt = (seen, pred) => questions.filter((x) => x.seen === seen && pred(x)).length;
  c["unseen.ans"] = fmt.int(cnt(false, (x) => x.t !== "unanswerable"));
  c["unseen.pr"] = fmt.int(cnt(false, (x) => x.t === "pr"));
  c["seen.ans"] = fmt.int(cnt(true, (x) => x.t !== "unanswerable"));
  c["seen.pr"] = fmt.int(cnt(true, (x) => x.t === "pr"));

  const rec = fs.readdirSync(path.join(REPO, "results/receipts")).map((f) => readJSON(path.join(REPO, "results/receipts", f)));
  const start = new Date(Math.min(...rec.map((r) => Date.parse(r.started_utc))));
  const end = new Date(Math.max(...rec.map((r) => Date.parse(r.ended_utc))));
  const p = (n) => String(n).padStart(2, "0");
  c["run.window"] = `${start.getUTCFullYear()}-${p(start.getUTCMonth() + 1)}-${p(start.getUTCDate())}, ${p(start.getUTCHours())}:${p(start.getUTCMinutes())} to ${p(end.getUTCHours())}:${p(end.getUTCMinutes())} UTC`;
  if (rec.length !== 8) c["run.window"] += " (expected 8 receipts)";
  c["tokens.ratio"] = `${Math.round(ov.arms.A1f.median_input_tokens / ov.arms.A5.median_input_tokens)}x`;
  return c;
}
const CALC = computeCalc();

// ---------------------------------------------------------------- static server
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8", ".woff2": "font/woff2", ".png": "image/png",
  ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8", ".ico": "image/x-icon" };
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  let file = path.normalize(path.join(SITE, url));
  if (!file.startsWith(SITE)) { res.writeHead(403); return res.end(); }
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
  if (!fs.existsSync(file)) { res.writeHead(404); return res.end("not found"); }
  res.writeHead(200, { "content-type": MIME[path.extname(file)] || "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const BASE = `http://127.0.0.1:${server.address().port}/`;

// ---------------------------------------------------------------- checks
const failures = [];
const notes = [];
const fail = (ctx, msg) => failures.push(`[${ctx}] ${msg}`);

async function textChecks(page, ctx) {
  const bad = await page.evaluate(() => {
    const out = [];
    const re = /[–—]/;
    if (re.test(document.body.innerText)) {
      const t = document.body.innerText; const i = t.search(re);
      out.push("visible text: ..." + t.slice(Math.max(0, i - 40), i + 20) + "...");
    }
    if (re.test(document.title)) out.push("title");
    for (const el of document.querySelectorAll("*")) {
      for (const a of el.attributes) if (re.test(a.value) && a.name !== "d") out.push(`attribute ${a.name} on <${el.tagName.toLowerCase()}>`);
    }
    return out;
  });
  bad.forEach((b) => fail(ctx, `dash character in ${b}`));
  const scroll = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, w: window.innerWidth }));
  if (scroll.sw > scroll.w + 1) fail(ctx, `horizontal page scroll: scrollWidth ${scroll.sw} > ${scroll.w}`);
}

async function dataChecks(page, ctx) {
  const vals = await page.$$eval("[data-v]", (els) => els.map((e) => ({ p: e.dataset.v, f: e.dataset.f || "f1", t: e.textContent.trim() })));
  for (const v of vals) {
    const want = fmt[v.f] ? fmt[v.f](get(v.p)) : String(get(v.p));
    if (get(v.p) === undefined) fail(ctx, `data-v path ${v.p} is not in overview.json`);
    else if (v.t !== want) fail(ctx, `data-v ${v.p}: shows "${v.t}", overview.json gives "${want}"`);
  }
  const lits = await page.$$eval("[data-lit]", (els) => els.map((e) => ({ p: e.dataset.lit, t: e.textContent.trim() })));
  for (const l of lits) if (!String(get(l.p) ?? "").includes(l.t)) fail(ctx, `data-lit "${l.t}" not found in ${l.p}`);
  const calcs = await page.$$eval("[data-calc],[data-calc-edge],[data-calc-node]", (els) => els.map((e) => ({
    k: e.dataset.calc || (e.dataset.calcEdge ? "edge." + e.dataset.calcEdge : "node." + e.dataset.calcNode), t: e.textContent.trim() })));
  for (const c of calcs) {
    const want = CALC[c.k];
    if (want === undefined) { fail(ctx, `data-calc ${c.k} has no computed value`); continue; }
    const ok = c.k.startsWith("edge.") || c.k.startsWith("node.") ? c.t === want || c.t.endsWith(" " + want) : c.t === want;
    if (!ok) fail(ctx, `data-calc ${c.k}: shows "${c.t}", source gives "${want}"`);
  }
  return { vals: vals.length, lits: lits.length, calcs: calcs.length };
}

async function overviewChecks(page, ctx) {
  const A = ov.arms, H = lookup.hyp;
  const heroText = await page.$eval("#scores", (e) => e.innerText);
  for (const [lit, val] of [["80.0", A.A5.primary], ["39.2", A.A0.primary], ["26.0", A.A1f.primary]]) {
    if (fmt.f1(val) !== lit) fail(ctx, `hero literal ${lit} does not match overview.json (${val})`);
    if (!heroText.includes(lit)) fail(ctx, `hero does not show ${lit}`);
  }
  const h1Text = await page.$eval("#h1-line", (e) => e.innerText);
  for (const [lit, val] of [["+40.8", fmt.s1(H.H1.diff)], ["35.7", fmt.f1(H.H1.lo)], ["46.2", fmt.f1(H.H1.hi)]]) {
    if (val !== lit) fail(ctx, `H1 literal ${lit} does not match overview.json (${val})`);
    if (!h1Text.includes(lit)) fail(ctx, `H1 line does not show ${lit}`);
  }
  for (const h of ov.hypotheses) {
    const row = await page.$(`#forest tr[data-row="${h.id}"]`);
    if (!row) { fail(ctx, `forest row ${h.id} missing`); continue; }
    const t = await row.innerText();
    const want = h.diff === null ? [h.value, h.threshold] : [fmt.s1(h.diff), fmt.f1(h.lo), fmt.f1(h.hi), h.threshold];
    if (h.result) want.push(h.result);
    for (const w of want) if (!t.includes(w)) fail(ctx, `forest row ${h.id} lacks "${w}"`);
  }
  for (const a of ov.arm_order) {
    const row = await page.$(`#arms-table tr[data-arm="${a}"]`);
    if (!row) { fail(ctx, `arm row ${a} missing`); continue; }
    const cells = await row.$$eval("td", (tds) => tds.map((td) => td.innerText.trim()));
    const r = A[a];
    const want = [a, null, fmt.int(r.n), fmt.f1(r.primary), ...["person", "company", "pr", "url", "content"].map((k) => fmt.f1(r.per_type[k].mean)),
      fmt.f1(r.unanswerable_correct_refusal_judge), fmt.int(r.median_input_tokens), r.budget_stop_rate ?? "n/a"];
    want.forEach((w, i) => { if (w !== null && !(cells[i] || "").startsWith(w)) fail(ctx, `arm ${a} column ${i}: "${cells[i]}" != "${w}"`); });
  }
  const bs = ov.also_reported.by_seen;
  const led = await page.$eval("#ledger", (e) => e.innerText);
  for (const w of [fmt.s1(bs.seen.H1.diff), fmt.s1(bs.unseen.H1.diff), fmt.f1(bs.unseen.H1.lo), fmt.f1(bs.unseen.H1.hi),
    fmt.f1(bs.seen.primary.A5), fmt.f1(bs.unseen.primary.A5), fmt.f1(bs.seen.primary.A0), fmt.f1(bs.unseen.primary.A0)]) {
    if (!led.includes(w)) fail(ctx, `seen/unseen value ${w} missing from "Where it is weaker"`);
  }
  // hero specimen and recorded query rendered from data
  const spec = await page.$eval("#specimen-body", (e) => e.innerText);
  const sf = readJSON(path.join(SITE, "data/q/SupportForce.json")).questions.find((q) => q.k === "SupportForce/ans/3");
  for (const id of sf.arms.A5.answer_ids) if (!spec.includes(id)) fail(ctx, `hero specimen lacks A5 id ${id}`);
  if (!(await page.$("#query-body pre.code"))) fail(ctx, "recorded Cypher query did not render");
  // hero graph picture
  const hg = await page.$eval("#specimen", (e) => ({ svg: e.querySelector(".gv-svg")?.textContent || "", answers: e.querySelectorAll(".gv-answer").length }));
  if (!hg.svg) fail(ctx, "hero graph picture did not render");
  else {
    if (hg.answers !== sf.arms.A5.answer_ids.length) fail(ctx, `hero graph has ${hg.answers} answer nodes`);
    for (const nm of sf.arms.A5.answer_names) if (!hg.svg.includes(nm)) fail(ctx, `hero graph lacks "${nm}"`);
  }
  // Start here: five numbered links that resolve
  const links = await page.$$eval("#start .start-list li > a", (as) => as.map((a) => a.getAttribute("href")));
  if (links.length !== 5) fail(ctx, `Start here has ${links.length} links, want 5`);
  const whatif = readJSON(path.join(SITE, "data/whatif.json"));
  const qkeys = new Set(questions.map((x) => x.k));
  for (const h of links) {
    const m = h.match(/^explore\.html#q=(.+)$/);
    const w = h.match(/^whatif\.html#p=(.+)$/);
    if (m) { if (!qkeys.has(m[1])) fail(ctx, `Start here link ${h} does not resolve`); }
    else if (w) { if (!whatif.products[w[1]]) fail(ctx, `Start here link ${h} does not resolve`); }
    else fail(ctx, `Start here link ${h} is not an explorer or what-if link`);
  }
  // the two picked by script: a refusal A5 got right (A1f 0) and a miss where A0 scored above 0
  const byk = new Map(questions.map((x) => [x.k, x]));
  const r = byk.get(links[2].replace("explore.html#q=", "")), wq = byk.get(links[3].replace("explore.html#q=", ""));
  if (!(r && r.t === "unanswerable" && r.s.A5 === 100 && r.s.A1f === 0)) fail(ctx, "Start here item 3 is not an unanswerable question with A5 100 and A1f 0");
  if (!(wq && wq.t !== "unanswerable" && wq.s.A5 === 0 && wq.s.A0 > 0)) fail(ctx, "Start here item 4 is not a question with A5 0 and A0 above 0");
}

const ctx_graph = [];
// labels the graph must show (matched against its text and its text alternative)
const GRAPH_EXPECT = {
  "SecurityForce/ans/21": ["Julia Miller", "inaccurate api endpoint mapping", "token expiry mismanagement issue", "Bug", "BugEvent", "Person"],
  "SupportForce/ans/3": ["Alice Davis", "Hannah Taylor", "ortaix technical specifications document", "Document", "Person"],
};

async function exploreChecks(page, ctx, key) {
  await page.goto(BASE + "explore.html#q=" + key, { waitUntil: "networkidle" });
  await page.waitForSelector(".steps", { timeout: 15000 }).catch(() => fail(ctx, `${key}: path did not render`));
  const n = await page.$$eval(".qblock pre.code", (e) => e.length);
  if (n < 1) fail(ctx, `${key}: no Cypher block`);
  const product = key.split("/")[0];
  const q = readJSON(path.join(SITE, `data/q/${product}.json`)).questions.find((x) => x.k === key);
  const text = await page.$eval("#detail", (e) => e.textContent);
  for (const id of q.gold.ids) if (!text.includes(id)) fail(ctx, `${key}: gold id ${id} not rendered`);
  for (const id of q.arms.A5.answer_ids) if (!text.includes(id)) fail(ctx, `${key}: A5 id ${id} not rendered`);
  // the graph picture: renders, names every answer entity, and the expected node labels are present
  if (!(await page.$("#detail .gv-svg"))) fail(ctx, `${key}: graph picture did not render`);
  else {
    const g = await page.$eval("#detail .gv", (e) => ({
      svg: e.querySelector("svg").textContent.replace(/\s+/g, " "),
      alt: e.textContent.replace(/\s+/g, " "),
      nodes: e.querySelectorAll(".gv-node").length,
      answers: e.querySelectorAll(".gv-answer").length,
      aria: e.querySelector("svg").getAttribute("aria-label"),
      edgesText: e.querySelectorAll("ul.sr-only li").length,
    }));
    if (!g.aria || !/nodes in \d+ column/.test(g.aria)) fail(ctx, `${key}: graph has no aria-label`);
    const names = q.arms.A5.answer_names || [];
    names.forEach((nm, i) => { if (nm && !g.svg.includes(nm)) fail(ctx, `${key}: graph lacks answer node "${nm}"`); });
    if (g.answers !== q.arms.A5.answer_ids.length) fail(ctx, `${key}: graph has ${g.answers} answer nodes, A5 gave ${q.arms.A5.answer_ids.length}`);
    if (key === "SecurityForce/ans/21") {
      const li = await page.$$eval("#detail .gv ul.sr-only li", (e) => e.map((x) => x.textContent));
      const cnt = (t) => li.filter((x) => x.includes(` ${t} `)).length;
      const solid = await page.$$eval("#detail .gv .gv-edge:not(.gv-unknown)", (e) => e.length);
      if (cnt("FIXES") !== 2 || cnt("OPENED") !== 2 || cnt("HAS_EVENT") !== 4 || solid !== 8 || li.length !== 8) fail(ctx, `${key}: edges FIXES ${cnt("FIXES")}, OPENED ${cnt("OPENED")}, HAS_EVENT ${cnt("HAS_EVENT")}, solid ${solid}, total ${li.length}; want 2, 2, 4, 8, 8`);
    }
    for (const label of (GRAPH_EXPECT[key] || [])) if (!g.alt.includes(label)) fail(ctx, `${key}: graph lacks "${label}"`);
    if (g.nodes > 30) fail(ctx, `${key}: graph draws ${g.nodes} nodes`);
    ctx_graph.push(`${key}: ${g.nodes} nodes, ${g.answers} answer nodes, ${g.edgesText} edges`);
  }
  // default list order is the dataset order
  if (!ctx.includes("390")) {
    const keys = await page.$$eval("#qlist a code", (els) => els.map((e) => e.textContent));
    const want = questions.slice(0, keys.length).map((x) => x.k);
    const i = questions.findIndex((x) => x.k === key);
    const pageStart = Math.floor(i / 40) * 40;
    const want2 = questions.slice(pageStart, pageStart + keys.length).map((x) => x.k);
    if (JSON.stringify(keys) !== JSON.stringify(want2) && JSON.stringify(keys) !== JSON.stringify(want)) fail(ctx, `${key}: list is not in dataset order`);
  }
  return n;
}

const browser = await chromium.launch();
const VARIANTS = [
  { w: 1440, h: 900 }, { w: 390, h: 844 },
];
const THEMES = ["light", "dark"];
const summary = [];

for (const vp of VARIANTS) for (const theme of THEMES) {
  const context = await browser.newContext({ viewport: { width: vp.w, height: vp.h }, colorScheme: theme, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const tag = `${vp.w}-${theme}`;
  let ctx = "";
  const responses = [];
  page.on("console", (m) => { if (m.type() === "error") fail(ctx, `console error: ${m.text()}`); });
  page.on("pageerror", (e) => fail(ctx, `page error: ${e.message}`));
  page.on("requestfailed", (r) => fail(ctx, `request failed: ${r.url()} ${r.failure()?.errorText}`));
  page.on("response", async (r) => {
    if (r.status() >= 400) fail(ctx, `HTTP ${r.status()} for ${r.url()}`);
    responses.push({ url: r.url(), len: Number(r.headers()["content-length"] || 0), ctx });
  });

  // overview
  ctx = `index ${tag}`;
  await page.goto(BASE + "index.html", { waitUntil: "networkidle" });
  await page.waitForSelector("#specimen-body .path", { timeout: 15000 }).catch(() => fail(ctx, "hero specimen did not render"));
  const firstPaint = responses.filter((r) => r.ctx === ctx).map((r) => r.url.replace(BASE, ""));
  if (firstPaint.some((u) => u.includes("questions.json"))) fail(ctx, "overview loaded questions.json");
  for (const f of firstPaint.filter((u) => u.endsWith(".json"))) {
    const size = fs.statSync(path.join(SITE, f)).size;
    if (size > 600 * 1024) fail(ctx, `overview first paint loaded ${f} (${size} bytes > 600 KB)`);
  }
  await page.waitForTimeout(900);
  if (vp.w === 1440 && theme === "light") notes.push(`overview first paint JSON: ${firstPaint.filter((u) => u.endsWith(".json")).join(", ")}`);
  await page.screenshot({ path: path.join(SHOTS, `index-${tag}-fold.png`) });
  await page.$eval("#query", (e) => e.scrollIntoView());
  await page.waitForSelector("#query-body pre.code", { timeout: 15000 }).catch(() => fail(ctx, "recorded query did not load"));
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(150);
  await textChecks(page, ctx);
  const dc = await dataChecks(page, ctx);
  await overviewChecks(page, ctx);
  await page.screenshot({ path: path.join(SHOTS, `index-${tag}.png`), fullPage: true });
  summary.push(`${ctx}: ${dc.vals} data-v, ${dc.lits} data-lit, ${dc.calcs} data-calc checked`);

  // explorer
  for (const key of ["SupportForce/ans/3", "SecurityForce/ans/21"]) {
    ctx = `explore ${tag} ${key}`;
    const n = await exploreChecks(page, ctx, key);
    await page.$eval(".q-details", (d) => { d.open = true; }).catch(() => {});
    await page.waitForTimeout(100);
    await textChecks(page, ctx);
    await (await page.$("#detail .gv"))?.screenshot({ path: path.join(SHOTS, `graph-${tag}-${key.split("/")[0]}.png`) });
    await page.screenshot({ path: path.join(SHOTS, `explore-${tag}-${key.split("/")[0]}.png`), fullPage: vp.w < 500 });
    if (vp.w === 1440) await page.screenshot({ path: path.join(SHOTS, `explore-${tag}-${key.split("/")[0]}-full.png`), fullPage: true });
    summary.push(`${ctx}: ${n} Cypher blocks`);
  }
  ctx = `explore ${tag} random`;
  await page.goto(BASE + "explore.html#random", { waitUntil: "networkidle" });
  await page.waitForSelector(".steps, .d-sec", { timeout: 15000 }).catch(() => fail(ctx, "random question did not render"));
  if (!/^#q=/.test(await page.evaluate(() => location.hash))) fail(ctx, "#random did not resolve to a #q= link");
  ctx = `explore ${tag} list`;
  await page.goto(BASE + "explore.html", { waitUntil: "networkidle" });
  await page.waitForSelector("#qlist a", { timeout: 15000 }).catch(() => fail(ctx, "list did not render"));
  await textChecks(page, ctx);
  await page.screenshot({ path: path.join(SHOTS, `explore-${tag}-list.png`) });
  // a filtered state: A0 beat A5 on never-seen PR questions
  await page.selectOption("#f-type", "pr");
  await page.selectOption("#f-seen", "unseen");
  await page.selectOption("#f-outcome", "a0b");
  await page.waitForTimeout(150);
  const cnt = await page.$eval("#count", (e) => e.innerText);
  const want = questions.filter((x) => x.t === "pr" && !x.seen && x.s.A0 > x.s.A5).length;
  if (!cnt.startsWith(String(want))) fail(ctx, `filter count "${cnt}" != ${want}`);
  if (vp.w === 1440 && theme === "light") await page.screenshot({ path: path.join(SHOTS, `explore-${tag}-filtered.png`) });

  // what-if
  ctx = `whatif ${tag}`;
  await page.goto(BASE + "whatif.html", { waitUntil: "networkidle" });
  await page.waitForSelector(".people > li", { timeout: 15000 }).catch(() => fail(ctx, "people list did not render"));
  await textChecks(page, ctx);
  await page.goto(BASE + "whatif.html#p=SupportForce", { waitUntil: "networkidle" });
  await page.reload({ waitUntil: "networkidle" });
  await page.waitForSelector(".people details[open] .gv-svg", { timeout: 15000 }).catch(() => fail(ctx, "what-if graph did not render for the open person"));
  const wg = await page.$eval(".people details[open] .gv", (e) => ({ t: e.querySelector("svg").textContent, leaving: e.querySelector(".gv-leaving")?.textContent || "", n: e.querySelectorAll(".gv-node").length }));
  if (!/Julia Smith/.test(wg.leaving)) fail(ctx, `what-if graph leaving node is "${wg.leaving}", want Julia Smith`);
  if (wg.n !== 4) fail(ctx, `what-if graph for Julia Smith has ${wg.n} nodes, want 4 (the person and 3 newly unowned items)`);
  if (vp.w === 1440 && theme === "light") await page.screenshot({ path: path.join(SHOTS, `whatif-${tag}-graph.png`), fullPage: false });
  await page.goto(BASE + "whatif.html", { waitUntil: "networkidle" });
  await page.waitForSelector(".people > li", { timeout: 15000 });
  await page.screenshot({ path: path.join(SHOTS, `whatif-${tag}.png`), fullPage: true });
  await page.click("#tab-leads");
  await page.waitForTimeout(100);
  await textChecks(page, ctx + " leads");
  if (theme === "light") await page.screenshot({ path: path.join(SHOTS, `whatif-${tag}-leads.png`), fullPage: true });

  await context.close();
}

// theme toggle and storage
{
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "light" });
  const page = await context.newPage();
  await page.goto(BASE + "index.html", { waitUntil: "networkidle" });
  await page.click(".theme-toggle");
  const t1 = await page.evaluate(() => document.documentElement.dataset.theme);
  await page.reload({ waitUntil: "networkidle" });
  const t2 = await page.evaluate(() => document.documentElement.dataset.theme);
  if (t1 !== "dark" || t2 !== "dark") fail("theme", `toggle did not persist (after click ${t1}, after reload ${t2})`);
  await context.close();
}

await browser.close();
server.close();

console.log(summary.join("\n"));
console.log([...new Set(ctx_graph)].join("\n"));
if (notes.length) console.log(notes.join("\n"));
if (failures.length) {
  console.log(`\nFAIL: ${failures.length} problem(s)`);
  for (const f of [...new Set(failures)]) console.log(" - " + f);
  process.exit(1);
}
console.log(`\nPASS: 3 pages x 2 widths x 2 themes; screenshots in ${path.relative(REPO, SHOTS)}/`);
