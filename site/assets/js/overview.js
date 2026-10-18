// Overview page. First paint loads data/overview.json; the hero question then loads q/SupportForce.json
// (318 KB) and the recorded query loads q/SecurityForce.json only when its section comes near the viewport.
import { VIDEO_URL, BLOG_URL } from "../../config.js";
import {
  initTheme, getJSON, esc, escClean, f1, s1, int, v, blob, ARM_SHORT, ARM_ORDER, OUTCOME,
} from "./common.js";
import { forestRow, forestAxis, scatter, dumbbell, intervalStrip } from "./charts.js";
import { highlightCypher, firstPattern } from "./cypher.js";
import { renderSchema, EDGES_NOT_DRAWN } from "./schema.js";

initTheme();

const HERO_KEY = "SupportForce/ans/3";
const QUERY_KEY = "SecurityForce/ans/21";

const SEC_LABEL = {
  "A0f-A0 (subset)": "A0f minus A0, five products",
  "A4-A0f (subset)": "A4 minus A0f, five products",
  "A5-A4c (subset)": "A5 minus A4c, five products",
  "A0v-A0": "A0v minus A0, all products",
  "A5-A2 (A2 products)": "A5 minus A2, A2's six products",
  "A0-A2 (A2 products)": "A0 minus A2, A2's six products",
};
const SUBSET_NOTE = {
  A4: "Five products plus the 7 multi-hop questions",
  A4c: "Five products",
  A0f: "Five products",
  A2: "Six sampled products",
};

function thresholdValue(t) {
  const m = /lower bound > (-?\d+(?:\.\d+)?)/.exec(t || "");
  return m ? Number(m[1]) : null;
}
function priorValue(t) {
  const m = /prior: at least (\d+(?:\.\d+)?)/.exec(t || "");
  return m ? Number(m[1]) : null;
}
const pct = (s) => (s === null || s === undefined ? "n/a" : String(s));

async function main() {
  let ov;
  try {
    ov = await getJSON("data/overview.json");
  } catch (e) {
    document.getElementById("scores").innerHTML = `<p class="state-msg err">Could not load data/overview.json (${esc(e.message)}). Reload the page, or read results/RESULTS.md in the repository.</p>`;
    return;
  }
  const A = ov.arms;
  const hyp = Object.fromEntries(ov.hypotheses.map((h) => [h.id, h]));

  renderHero(ov, A);
  renderRemoval(ov, A, hyp);
  renderForest(ov, hyp);
  renderScatter(A);
  renderOverlap(ov);
  renderArms(ov);
  renderLedger(ov, A, hyp);
  renderSchema(document.getElementById("schema-figure"));
  renderEdgeTable();
  renderTimeline(ov);
  renderLinks();

  loadSpecimen();
  lazyQuery();
}

// ---------------------------------------------------------------- hero
function renderHero(ov, A) {
  const ratio = Math.round(A.A1f.median_input_tokens / A.A5.median_input_tokens);
  const rows = [
    { arm: "A5", label: "inherit-brain" },
    { arm: "A0", label: "HERB's retrieval tools, same model and loop" },
    { arm: "A1f", label: `The whole product in the prompt, with <span data-calc="tokens.ratio">${ratio}x</span> the tokens` },
  ];
  const el = document.getElementById("scores");
  el.innerHTML = rows.map((r) => {
    const p = A[r.arm].primary;
    return `<div class="score-row${r.arm === "A5" ? " is-a5" : ""}" data-arm="${r.arm}">
      <div class="score-num">${v(`arms.${r.arm}.primary`, p)}</div>
      <div><p class="score-label">${r.label} <span class="sr-only">(${r.arm})</span></p>
        <div class="score-track" aria-hidden="true"><span class="score-line" style="width:${p}%"></span><span class="score-dot" style="left:${p}%"></span></div>
      </div></div>`;
  }).join("");
  requestAnimationFrame(() => el.classList.add("animate"));
  document.getElementById("hero-line").innerHTML =
    `${v("n_questions", ov.n_questions, "int")} held-out questions on Salesforce's HERB benchmark, ${v("n_products", ov.n_products, "int")} products, plan pre-registered before the run.`;
}

// ---------------------------------------------------------------- removal test
function renderRemoval(ov, A, hyp) {
  const h1 = hyp.H1;
  document.getElementById("removal-claim").innerHTML =
    `Swap the graph tools for retrieval and the same model loses <span class="a5">${v("hyp.H1.diff", h1.diff)}</span> points.`;
  const row = (head, a5, a0, cls = "") => `<tr class="${cls}"><th scope="row">${head}</th><td data-arm="inherit-brain (A5)">${a5}</td><td data-arm="HERB agent (A0)"${a0 === "Same" ? ' class="same"' : ""}>${a0}</td></tr>`;
  document.getElementById("ablation").innerHTML = `<table class="ablation">
    <thead><tr><th scope="col"><span class="sr-only">Setting</span></th><th scope="col" class="a5h">inherit-brain (A5)</th><th scope="col">HERB's retrieval tools (A0)</th></tr></thead>
    <tbody>
      ${row("Model", "Qwen3-30B-A3B-Instruct-2507 on Nebius", "Same")}
      ${row("Limits", `<span data-lit="arms.A0.description">12 tool calls</span>, <span data-lit="arms.A0.description">60,000 input tokens</span> per question`, "Same")}
      ${row(`Tools<span class="only">The only change</span>`, "Nine read-only typed tools; each runs fixed Cypher on the product's FalkorDB graph", "The HERB paper's retrieval tools over the raw records", "diff")}
      ${row("Primary endpoint", `<span class="big a5">${v("arms.A5.primary", A.A5.primary)}</span>`, `<span class="big">${v("arms.A0.primary", A.A0.primary)}</span>`)}
      ${row("Median input tokens", v("arms.A5.median_input_tokens", A.A5.median_input_tokens, "int"), v("arms.A0.median_input_tokens", A.A0.median_input_tokens, "int"))}
      ${row("Stopped on the token budget", v("arms.A5.budget_stop_rate", A.A5.budget_stop_rate, "raw"), v("arms.A0.budget_stop_rate", A.A0.budget_stop_rate, "raw"))}
    </tbody></table>`;
  document.getElementById("h1-line").innerHTML =
    `<strong>H1</strong>, A5 minus A0 on the primary endpoint: <strong>${v("hyp.H1.diff", h1.diff, "s1")}</strong>, 95% interval ${v("hyp.H1.lo", h1.lo)} to ${v("hyp.H1.hi", h1.hi)} (${esc(ov.bootstrap)}). Registered threshold: ${esc(h1.threshold)}. Result: ${esc(h1.result)}.`;
  intervalStrip(document.getElementById("h1-strip"), {
    diff: h1.diff, lo: h1.lo, hi: h1.hi, threshold: thresholdValue(h1.threshold), prior: priorValue(h1.threshold),
  });
}

// ---------------------------------------------------------------- forest plot
function renderForest(ov, hyp) {
  const wrap = document.getElementById("forest");
  const rows = [];
  for (const h of ov.hypotheses) {
    const a5 = /^A5/.test(h.comparison) || h.id.startsWith("H");
    const base = `hyp.${h.id}`;
    if (h.diff === null || h.diff === undefined) {
      rows.push(`<tr class="${a5 ? "is-a5" : ""}" data-row="${esc(h.id)}">
        <th scope="row" class="f-id">${esc(h.id)}</th><td class="f-cmp">${esc(h.comparison)}</td>
        <td class="f-plot"><span class="f-text">No interval</span></td>
        <td class="f-val"><b>${v(base + ".value", h.value, "raw")}</b></td>
        <td class="f-thr">${v(base + ".threshold", h.threshold, "raw")}</td><td class="f-res">${v(base + ".result", h.result, "raw")}</td></tr>`);
    } else {
      rows.push(`<tr class="${a5 ? "is-a5" : ""}" data-row="${esc(h.id)}">
        <th scope="row" class="f-id">${esc(h.id)}</th><td class="f-cmp">${esc(h.comparison)}</td>
        <td class="f-plot"><div class="plotcell" data-lo="${h.lo}" data-hi="${h.hi}" data-diff="${h.diff}" data-thr="${thresholdValue(h.threshold) ?? ""}" data-a5="${a5 ? 1 : 0}"></div></td>
        <td class="f-val"><b>${v(base + ".diff", h.diff, "s1")}</b> <span class="ci">[${v(base + ".lo", h.lo)}, ${v(base + ".hi", h.hi)}]</span></td>
        <td class="f-thr">${v(base + ".threshold", h.threshold, "raw")}</td><td class="f-res">${h.result ? v(base + ".result", h.result, "raw") : '<span class="dim">reported</span>'}</td></tr>`);
    }
  }
  rows.push(`<tr class="group"><th colspan="6" scope="rowgroup">Secondary contrasts: same questions, no threshold, no pre-set direction</th></tr>`);
  for (const c of ov.secondary_contrasts) {
    const a5 = /^A5-/.test(c.contrast);
    const base = `sec.${c.contrast}`;
    rows.push(`<tr class="${a5 ? "is-a5" : ""}" data-row="${esc(c.contrast)}">
      <th scope="row" class="f-id" colspan="2" style="font-weight:500;color:var(--ink-2)">${esc(SEC_LABEL[c.contrast] || c.contrast.replace("-", " minus "))}</th>
      <td class="f-plot"><div class="plotcell" data-lo="${c.lo}" data-hi="${c.hi}" data-diff="${c.diff}" data-thr="" data-a5="${a5 ? 1 : 0}"></div></td>
      <td class="f-val"><b>${v(base + ".diff", c.diff, "s1")}</b> <span class="ci">[${v(base + ".lo", c.lo)}, ${v(base + ".hi", c.hi)}]</span></td>
      <td class="f-thr"></td><td class="f-res"></td></tr>`);
  }
  wrap.innerHTML = `<table class="forest">
    <colgroup><col style="width:84px"><col style="width:21%"><col><col style="width:172px"><col style="width:150px"><col style="width:120px"></colgroup>
    <caption class="sr-only">Hypotheses and secondary contrasts: difference on the primary endpoint with 95% interval</caption>
    <thead><tr><th scope="col">Row</th><th scope="col">Comparison</th><th scope="col" class="f-plot"><span class="sr-only">Interval plot, scale </span><div class="axiscell"></div></th><th scope="col">Difference [95% interval]</th><th scope="col">Threshold</th><th scope="col">Result</th></tr></thead>
    <tbody>${rows.join("")}</tbody></table>
    <p class="figure-note">Points are differences on the primary endpoint, lines their 95% intervals (${esc(ov.bootstrap)}, over the ${int(ov.n_products)} held-out products). Rows that involve A5 are drawn in blue; the rest in grey. Paper-style endpoint, four types with content: H1 ${v("paper_style.H1.diff", ov.paper_style.H1.diff, "s1")} [${v("paper_style.H1.lo", ov.paper_style.H1.lo)}, ${v("paper_style.H1.hi", ov.paper_style.H1.hi)}], H2 ${v("paper_style.H2.diff", ov.paper_style.H2.diff, "s1")} [${v("paper_style.H2.lo", ov.paper_style.H2.lo)}, ${v("paper_style.H2.hi", ov.paper_style.H2.hi)}].</p>`;
  wrap.querySelectorAll(".plotcell").forEach((el) => forestRow(el, {
    lo: Number(el.dataset.lo), hi: Number(el.dataset.hi), diff: Number(el.dataset.diff),
    threshold: el.dataset.thr === "" ? null : Number(el.dataset.thr), a5: el.dataset.a5 === "1",
  }));
  wrap.querySelectorAll(".axiscell").forEach((el) => forestAxis(el));
}

// ---------------------------------------------------------------- scatter and citation overlap
function renderScatter(A) {
  const names = { A5: "A5 inherit-brain", A0: "A0 HERB agent", A1f: "A1f whole product", A0v: "A0v HERB RAG" };
  const pts = ["A5", "A0", "A1f", "A0v"].map((a) => ({ label: names[a], score: A[a].primary, tokens: A[a].median_input_tokens, a5: a === "A5" }));
  scatter(document.getElementById("scatter"), pts);
  document.getElementById("scatter-table").insertAdjacentHTML("beforeend", `<table><thead><tr><th>Arm</th><th>Primary</th><th>Median input tokens</th></tr></thead><tbody>${
    ["A5", "A0", "A1f", "A0v"].map((a) => `<tr><td>${a}</td><td>${v(`arms.${a}.primary`, A[a].primary)}</td><td>${v(`arms.${a}.median_input_tokens`, A[a].median_input_tokens, "int")}</td></tr>`).join("")}</tbody></table>`);
}

function renderOverlap(ov) {
  const c = ov.also_reported.citation_overlap;
  document.getElementById("overlap").innerHTML = `
    <p class="figure-title">The path behind the answer</p>
    <p class="stat">${v("also_reported.citation_overlap.A5.precision", c.A5.precision)}<small>%</small></p>
    <p class="vs">of the records A5 cites are among HERB's own citations for that question. For A0 it is ${v("also_reported.citation_overlap.A0.precision", c.A0.precision)}%.</p>
    <ul class="strip" aria-label="Share of cited records that HERB also cites, per arm">
      ${["A5", "A0", "A0v", "A2", "A1f"].map((a) => `<li class="${a === "A5" ? "is-a5" : ""}"><span class="arm">${a}</span><span class="track" aria-hidden="true"><span class="dot" style="left:${c[a].precision}%"></span></span><span class="val">${v(`also_reported.citation_overlap.${a}.precision`, c[a].precision)}%</span></li>`).join("")}
    </ul>
    <p class="micro">Descriptive and ungraded: measured on answerable questions (${v("also_reported.citation_overlap.A5.n", c.A5.n, "int")} for A5; A2 on its six products) and never part of any score. Recall is low for every arm (A5 ${v("also_reported.citation_overlap.A5.recall", c.A5.recall)}%) because HERB lists many records per question. The card arms cite cards, not records, and are left out.</p>`;
}

// ---------------------------------------------------------------- per-arm table
function renderArms(ov) {
  const A = ov.arms;
  const head = `<thead><tr>
    <th scope="col">Arm</th><th scope="col">What it is</th><th scope="col">Questions</th><th scope="col">Primary</th>
    <th scope="col">Person</th><th scope="col">Company</th><th scope="col">PR</th><th scope="col">URL</th><th scope="col">Content</th>
    <th scope="col">Correct refusal</th><th scope="col">Median input tokens</th><th scope="col">Budget stops</th></tr></thead>`;
  const row = (a) => {
    const r = A[a];
    const t = (k) => `<td>${v(`arms.${a}.per_type.${k}.mean`, r.per_type[k].mean)}</td>`;
    const sub = SUBSET_NOTE[a] ? `<span class="sub">Subset: ${SUBSET_NOTE[a]}</span>` : "";
    return `<tr class="${a === "A5" ? "is-a5" : ""}" data-arm="${a}">
      <td>${a}</td><td class="what">${esc(ARM_SHORT[a])}${sub}</td>
      <td>${v(`arms.${a}.n`, r.n, "int")}</td><td class="primary">${v(`arms.${a}.primary`, r.primary)}</td>
      ${t("person")}${t("company")}${t("pr")}${t("url")}${t("content")}
      <td>${v(`arms.${a}.unanswerable_correct_refusal_judge`, r.unanswerable_correct_refusal_judge)}</td>
      <td>${v(`arms.${a}.median_input_tokens`, r.median_input_tokens, "int")}</td>
      <td>${r.budget_stop_rate === null ? '<span data-v="arms.' + a + '.budget_stop_rate" data-f="raw">n/a</span><span class="sr-only"> (one model call, no loop)</span>' : v(`arms.${a}.budget_stop_rate`, r.budget_stop_rate, "raw")}</td></tr>`;
  };
  const order = ov.arm_order || ARM_ORDER;
  const full = order.filter((a) => !SUBSET_NOTE[a]);
  const sub = order.filter((a) => SUBSET_NOTE[a]);
  document.getElementById("arms-table").innerHTML = `${head}<tbody>
    <tr class="group"><td colspan="12">All ${int(ov.n_questions)} held-out questions</td></tr>
    ${full.map(row).join("")}
    <tr class="group"><td colspan="12">Product subsets: compare these only through the paired contrasts above</td></tr>
    ${sub.map(row).join("")}</tbody>`;
  document.getElementById("arms-note").textContent = ov.arm_note;
  document.getElementById("arms-foot").innerHTML =
    `${esc(ov.caveats[3])} Correct refusal: the HERB judge's score on the ${int(A.A5.n_unanswerable)} unanswerable questions. Budget stops: share of questions where a loop arm hit its 60,000-token input budget; n/a for single-call arms.`;
}

// ---------------------------------------------------------------- where it is weaker
function renderLedger(ov, A, hyp) {
  const bs = ov.also_reported.by_seen;
  const post = ov.post_hoc.find((p) => p.primary_endpoint) || ov.post_hoc[1];
  const sec = Object.fromEntries(ov.secondary_contrasts.map((c) => [c.contrast, c]));
  const a4 = sec["A4-A0f (subset)"];
  const items = [
    {
      wide: true,
      fig: `${v("also_reported.by_seen.unseen.H1.diff", bs.unseen.H1.diff, "s1")}<small>lead over A0 on never-seen question strings</small>`,
      h: `The lead shrinks on questions never seen in development.`,
      p: `A5 minus A0 is ${v("also_reported.by_seen.seen.H1.diff", bs.seen.H1.diff, "s1")} on the ${v("also_reported.by_seen.seen.n_questions", bs.seen.n_questions, "int")} question strings that also appear in development, and ${v("also_reported.by_seen.unseen.H1.diff", bs.unseen.H1.diff, "s1")} [${v("also_reported.by_seen.unseen.H1.lo", bs.unseen.H1.lo)}, ${v("also_reported.by_seen.unseen.H1.hi", bs.unseen.H1.hi)}] on the ${v("also_reported.by_seen.unseen.n_questions", bs.unseen.n_questions, "int")} never seen: about a third as large, still above zero. A5 scores ${v("also_reported.by_seen.seen.primary.A5", bs.seen.primary.A5)} and ${v("also_reported.by_seen.unseen.primary.A5", bs.unseen.primary.A5)} on the two halves; A0 scores ${v("also_reported.by_seen.seen.primary.A0", bs.seen.primary.A0)} and ${v("also_reported.by_seen.unseen.primary.A0", bs.unseen.primary.A0)}. The halves are not the same mix: <span data-calc="unseen.pr">135</span> of the <span data-calc="unseen.ans">243</span> answerable unseen questions are PR questions, against <span data-calc="seen.pr">10</span> of <span data-calc="seen.ans">416</span> seen ones.`,
      chart: true,
    },
    {
      fig: `${v("arms.A5.per_type.pr.mean", A.A5.per_type.pr.mean)}<small>A5 on PR questions</small>`,
      h: "PR questions are A5's weakest type.",
      p: `Against ${v("arms.A5.per_type.person.mean", A.A5.per_type.person.mean)} on person and ${v("arms.A5.per_type.company.mean", A.A5.per_type.company.mean)} on company questions, over ${v("arms.A5.per_type.pr.n", A.A5.per_type.pr.n, "int")} PR questions.`,
    },
    {
      fig: `${v("sec.A4-A0f (subset).diff", a4.diff, "s1")}<small>A4 minus A0f, five products</small>`,
      h: "Typed depth-1 access did worse than retrieval over the same cards.",
      p: `A4 minus A0f is ${v("sec.A4-A0f (subset).diff", a4.diff, "s1")}, 95% interval ${v("sec.A4-A0f (subset).lo", a4.lo)} to ${v("sec.A4-A0f (subset).hi", a4.hi)}. A4 stopped on its input budget on ${v("arms.A4.budget_stop_rate", A.A4.budget_stop_rate, "raw")} of questions.`,
    },
    {
      fig: `<span data-lit="hyp.H3.result">7</span><small>questions behind H3</small>`,
      h: "H3 rests on one multi-hop family.",
      p: `A5 minus A4 on that family is ${v("hyp.H3.diff", hyp.H3.diff, "s1")} [${v("hyp.H3.lo", hyp.H3.lo)}, ${v("hyp.H3.hi", hyp.H3.hi)}], on 7 held-out questions, all person questions, one per product. Read it as a small pre-registered check, not a general finding about traversal.`,
    },
    {
      fig: `${v("arms.A5.per_type.content.mean", A.A5.per_type.content.mean)}<small>A5 on content questions, judged</small>`,
      h: "The judge is the same model.",
      p: "Content and unanswerable questions are scored with HERB's judge prompts run on the model under test, so those scores are not comparable with the HERB paper's GPT-4o-judged numbers.",
    },
    {
      fig: `${v("arms.A0.budget_stop_rate", A.A0.budget_stop_rate, "raw")}<small>of A0's questions hit the token budget</small>`,
      h: "The comparison runs under a fixed budget.",
      p: `A0 stopped on its 60,000-token input budget on ${v("arms.A0.budget_stop_rate", A.A0.budget_stop_rate, "raw")} of questions, A5 on ${v("arms.A5.budget_stop_rate", A.A5.budget_stop_rate, "raw")}. The differences are measured under that registered limit and no arm was given more, so this run does not say how A0 would score with a larger budget.`,
    },
    {
      fig: `${v("post_hoc.1.primary_endpoint.A1f", post.primary_endpoint.A1f)}<small>A1f with customer ids mapped to names</small>`,
      h: "A1f's score is partly a format failure.",
      p: `A1f answered ${v("post_hoc.1.changed_company_answers.A1f", post.changed_company_answers.A1f, "int")} of its ${v("arms.A1f.per_type.company.n", A.A1f.per_type.company.n, "int")} company questions with customer ids instead of names. Mapped to names, its primary endpoint is ${v("post_hoc.1.primary_endpoint.A1f", post.primary_endpoint.A1f)}, not ${v("arms.A1f.primary", A.A1f.primary)}, and H2 is ${v("post_hoc.1.H2.diff", post.H2.diff, "s1")} instead of ${v("hyp.H2.diff", hyp.H2.diff, "s1")}. A post-hoc check, not pre-registered.`,
    },
  ];
  const ul = document.getElementById("ledger");
  ul.innerHTML = items.map((it, i) => `<li class="${it.wide ? "wide" : ""}">
      <p class="fig">${it.fig}</p>
      <div><h3>${it.h}</h3><p>${it.p}</p></div>
      ${it.chart ? `<div class="dumbbell"><div id="dumbbell"></div><details class="table-alt"><summary>Show as a table</summary><table><thead><tr><th></th><th>A5</th><th>A0</th><th>A5 minus A0</th></tr></thead><tbody>
        <tr><th scope="row">Seen in development</th><td>${f1(bs.seen.primary.A5)}</td><td>${f1(bs.seen.primary.A0)}</td><td>${s1(bs.seen.H1.diff)} [${f1(bs.seen.H1.lo)}, ${f1(bs.seen.H1.hi)}]</td></tr>
        <tr><th scope="row">Never seen</th><td>${f1(bs.unseen.primary.A5)}</td><td>${f1(bs.unseen.primary.A0)}</td><td>${s1(bs.unseen.H1.diff)} [${f1(bs.unseen.H1.lo)}, ${f1(bs.unseen.H1.hi)}]</td></tr></tbody></table></details></div>` : ""}
    </li>`).join("");
  dumbbell(document.getElementById("dumbbell"), [
    { label: `Seen in development (${int(bs.seen.n_questions)})`, a5: bs.seen.primary.A5, a0: bs.seen.primary.A0 },
    { label: `Never seen (${int(bs.unseen.n_questions)})`, a5: bs.unseen.primary.A5, a0: bs.unseen.primary.A0 },
  ]);
}

// ---------------------------------------------------------------- schema table
function renderEdgeTable() {
  document.querySelector("#edge-table tbody").innerHTML = EDGES_NOT_DRAWN.map((e) =>
    `<tr><td><code>${e.type}</code></td><td>${esc(e.ends)}</td><td class="n" data-calc="edge.${e.type}">${int(e.count)}</td></tr>`).join("");
}

// ---------------------------------------------------------------- timeline
function renderTimeline(ov) {
  const t = new Date(ov.approved_utc);
  const pad = (n) => String(n).padStart(2, "0");
  const when = `${t.getUTCFullYear()}-${pad(t.getUTCMonth() + 1)}-${pad(t.getUTCDate())}, ${pad(t.getUTCHours())}:${pad(t.getUTCMinutes())}:${pad(t.getUTCSeconds())} UTC`;
  const steps = [
    { when: "", h: "Plan written", p: `<a href="${blob("PREREG.md")}">PREREG.md</a> fixes the eight arms, the endpoints, H1 to H4 with thresholds, and the statistics.` },
    { when: `<span data-v="approved_utc" data-f="utc">${when}</span>`, h: "Hash posted", p: `sha256 <code data-v="prereg_sha256" data-f="raw">${esc(ov.prereg_sha256)}</code>, as a <a href="${esc(ov.gist_url)}">public gist</a> and a <a href="${esc(ov.wayback_url)}">Wayback snapshot</a>.` },
    { when: `<span data-calc="run.window">2026-10-17, 03:12 to 16:14 UTC</span>`, h: "Held-out run", p: "Eight arms, each run once over the held-out questions, A5 first. Run receipts record the code commit, model and output hashes." },
    { when: "", h: "Grading", p: "HERB's set F1 on answer ids; HERB's judge prompts for content and unanswerable questions. A failed answer scores 0 and stays in the denominator." },
    { when: "", h: "Results", p: `<a href="${blob("results/RESULTS.md")}">RESULTS.md</a>, with per-question tables, receipts and the two post-hoc checks marked as such.` },
  ];
  document.getElementById("timeline").innerHTML = steps.map((s) =>
    `<li><span class="when">${s.when || "&nbsp;"}</span><h3>${s.h}</h3><p>${s.p}</p></li>`).join("");
  document.getElementById("spend").innerHTML =
    `Total model spend for the project, through grading: ${v("total_spend_usd", ov.total_spend_usd, "usd")} USD <span class="dim">of a ${v("spend_cap_usd", ov.spend_cap_usd, "int")} USD cap.</span>`;
}

function renderLinks() {
  const extra = [];
  if (VIDEO_URL) extra.push(`<a href="${esc(VIDEO_URL)}">Video</a>`);
  if (BLOG_URL) extra.push(`<a href="${esc(BLOG_URL)}">Write-up</a>`);
  if (extra.length) {
    document.getElementById("hero-links").insertAdjacentHTML("beforeend", extra.join(""));
    document.getElementById("footer-extra").innerHTML = extra.join(" ");
  }
}

// ---------------------------------------------------------------- hero specimen
async function loadSpecimen() {
  const body = document.getElementById("specimen-body");
  try {
    const prod = await getJSON(`data/q/${HERO_KEY.split("/")[0]}.json`);
    const q = prod.questions.find((x) => x.k === HERO_KEY);
    if (!q) throw new Error(`${HERO_KEY} not in the product file`);
    body.innerHTML = specimenHTML(q);
  } catch (e) {
    body.innerHTML = `<p class="state-msg err">Could not load the example question (${esc(e.message)}). <a href="explore.html#q=${HERO_KEY}">Open it in the explorer</a>.</p>`;
  }
}

function specimenHTML(q) {
  const a5 = q.arms.A5;
  const gold = new Set(q.gold?.ids || []);
  const inGold = a5.answer_ids.filter((x) => gold.has(x)).length;
  const steps = q.a5.path.map((s) => {
    const params = typeof s.params === "object" && s.params
      ? Object.entries(s.params).map(([k, val]) => `${esc(k)}: ${esc(typeof val === "string" ? val : JSON.stringify(val))}`).join("; ")
      : esc(String(s.params));
    const queries = s.queries.map((qq) => `<li class="p-query"><code class="pat">${esc(firstPattern(qq.cypher))}</code> <span class="p-meta">${int(qq.rows)} rows</span></li>`).join("");
    return `<li class="p-tool"><p class="p-title"><code>${esc(s.tool)}</code></p><p class="p-meta">${params}</p><p class="p-meta">${int(s.rows)} rows, ${s.ms.toFixed(1)} ms, ${s.queries.length} read-only Cypher ${s.queries.length === 1 ? "query" : "queries"}</p></li>${queries}`;
  }).join("");
  const chips = a5.answer_ids.map((id, i) => {
    const ok = gold.has(id);
    const name = escClean(a5.answer_names?.[i] || "");
    return `<span class="chip ${ok ? "ok" : "bad"}" title="${name}"><span class="g" aria-hidden="true">${ok ? "✓" : "×"}</span><code>${esc(id)}</code><span class="sr-only">${name}, ${ok ? "in the gold answer" : "not in the gold answer"}</span></span>`;
  }).join("");
  const others = ARM_ORDER.filter((a) => a !== "A5" && q.arms[a]).map((a) => `<span>${a} <b>${f1(q.arms[a].score)}</b></span>`).join("");
  return `
    <p class="micro" style="margin-bottom:6px">${esc(HERO_KEY.split("/")[0])}, ${esc(q.type)} question, ${q.seen ? "seen in development" : "never seen in development"}</p>
    <p class="specimen-q">${escClean(q.text)}</p>
    <ol class="path" aria-label="A5's path">
      ${steps}
      <li class="p-answer"><p class="p-title">Answer: ${a5.answer_ids.length} employee ids, ${inGold} of ${gold.size} gold ids, score <span class="a5">${f1(a5.score)}</span></p>
        <div class="chips">${chips}</div></li>
    </ol>
    <div class="others"><span>Same question, other arms (F1 x 100):</span>${others}</div>
    <div class="specimen-links"><a href="explore.html#q=${HERO_KEY}">Open this question in the explorer</a><a href="explore.html#random">Random question</a></div>`;
}

// ---------------------------------------------------------------- recorded query (lazy)
function lazyQuery() {
  const target = document.getElementById("query");
  let done = false;
  const go = () => { if (!done) { done = true; loadQuery(); } };
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); go(); } }, { rootMargin: "800px 0px" });
    io.observe(target);
  } else go();
}

async function loadQuery() {
  const body = document.getElementById("query-body");
  try {
    const prod = await getJSON(`data/q/${QUERY_KEY.split("/")[0]}.json`);
    const q = prod.questions.find((x) => x.k === QUERY_KEY);
    const calls = q.a5.path;
    const call = calls.find((c) => c.tool === "bug_board" && c.queries.some((x) => /:FIXES\]/.test(x.cypher)));
    const qq = call.queries.find((x) => /:FIXES\]/.test(x.cypher));
    const params = Object.entries(call.params).map(([k, val]) => `${esc(k)}=${esc(JSON.stringify(val))}`).join(", ");
    const a5 = q.arms.A5;
    const gold = new Set(q.gold.ids);
    const right = a5.answer_ids.length && a5.answer_ids.every((x) => gold.has(x));
    body.innerHTML = `
      <p class="query-caption">Call ${call.n} of ${calls.length} for <a href="explore.html#q=${QUERY_KEY}"><code>${QUERY_KEY}</code></a>: <q>${escClean(q.text)}</q> The model chose <code>${esc(call.tool)}(${params})</code>; the tool ran this query:</p>
      <pre class="code" tabindex="0" aria-label="Cypher query">${highlightCypher(qq.cypher)}</pre>
      <p class="code-meta"><span><b>${esc(qq.command)}</b></span><span>params <b>${esc(JSON.stringify(qq.params))}</b></span><span><b>${int(qq.rows)}</b> rows</span><span><b>${qq.ms.toFixed(1)}</b> ms</span></p>
      <p class="small" style="margin-top:12px">Line breaks added for reading; the text is as recorded. <span class="a5">$parameters</span> are the only part the model fills. <a href="${blob("docs/CYPHER.md")}">Every tool's Cypher</a> is in docs/CYPHER.md. Final answer: ${a5.answer_names.map((n, i) => `${escClean(n)} (<code>${esc(a5.answer_ids[i])}</code>)`).join(", ")}, ${right ? "which matches the gold answer" : "scored " + f1(a5.score)}.</p>`;
  } catch (e) {
    body.innerHTML = `<p class="state-msg err">Could not load the recorded query (${esc(e.message)}).</p>`;
  }
}

main();
