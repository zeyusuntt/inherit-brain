// Layered inline-SVG pictures of a path in the graph. Deterministic: one column per node label,
// barycentre ordering inside a column, cubic edges. No force layout, no library.
// Solid edges are real edges read from the graph (call.edges) of a type the call's Cypher traverses;
// a node with no such edge keeps one dashed line to its row's answer node.
import { esc, escClean } from "./common.js";

// Column order, left to right, following the tools' joins.
const RANK = {
  Observation: 0, Url: 0, Document: 0, Feature: 0, Meeting: 0,
  Company: 0.5, Customer: 1, Bug: 2, BugEvent: 3, PR: 4, Person: 5, Competitor: 5, Product: 5,
};
const LOOKUP_TOOLS = new Set(["resolve_entity", "resolve_person"]);

// Relationship types named in the call's Cypher: [x:TYPE] and [:TYPE].
export function cypherTypes(queries) {
  const out = new Set();
  for (const qq of queries || []) for (const m of String(qq.cypher || "").matchAll(/\[\w*:(\w+)/g)) out.add(m[1]);
  return out;
}

// ---------------------------------------------------------------- names
const tailAfter = (id, n) => id.split(":").slice(n).join(":");
function shortName(label, id, names) {
  const raw = id.slice(label.length + 1);
  if (label === "Person") return names.get(raw) || raw;
  if (label === "Document") return raw.replace(/[_-]+/g, " ");
  if (label === "Bug") return (raw.split(":").pop() || raw).replace(/-/g, " ");
  if (label === "PR") { const n = raw.match(/pull\/(\d+)/); return n ? "#" + n[1] : raw; }
  if (label === "BugEvent") return raw.split(":")[1] || raw;
  if (label === "Feature") return tailAfter(id, 2) || raw;
  if (label === "Url") return raw.replace(/^https?:\/\/(www\.)?/, "");
  return raw.split(":").pop() || raw;
}
function wrapName(name, max) {
  if (name.length <= max) return [name];
  let cut = name.lastIndexOf(" ", max);
  if (cut < max * 0.5) cut = max;
  return [name.slice(0, cut).trimEnd(), trunc(name.slice(cut).trim(), max)];
}
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1).trimEnd() + "…" : s);

// ---------------------------------------------------------------- graph from an answer
export function buildAnswerGraph(q) {
  const a5 = q.arms?.A5;
  const path = q.a5?.path || [];
  const ansIds = new Set(a5?.answer_ids || []);
  const goldIds = new Set(q.gold?.ids || []);
  const hasGold = goldIds.size > 0;
  const names = new Map();
  (a5?.answer_ids || []).forEach((id, i) => a5.answer_names?.[i] && names.set(id, a5.answer_names[i]));
  (q.gold?.ids || []).forEach((id, i) => q.gold.names?.[i] && names.set(id, q.gold.names[i]));
  for (const s of path) for (const r of s.top_rows || []) if (r.name && String(r.answer).startsWith("eid_")) names.set(r.answer, r.name);

  const useLookups = !path.some((s) => !LOOKUP_TOOLS.has(s.tool) && (s.top_rows || []).length);
  const nodes = new Map();   // key -> node
  const edges = new Map();   // "a>b" -> edge
  let order = 0;
  const node = (key) => {
    if (!nodes.has(key)) {
      const label = key.split(":")[0];
      const id = key.slice(label.length + 1);
      nodes.set(key, { key, label, id, order: order++, answer: false, gold: false, deg: 0 });
    }
    return nodes.get(key);
  };

  const rowAnswer = new Map();   // node key -> the answer node of the first row that returned it
  const real = new Map();        // "a>b" -> {from, to, types:Set}: edges read from the graph, of a type the call's Cypher traverses
  for (const s of path) {
    if (LOOKUP_TOOLS.has(s.tool) && !useLookups) continue;
    for (const r of s.top_rows || []) {
      const refs = (r.refs || []).filter(Boolean);
      const ans = String(r.answer ?? "");
      let ansKey = refs.find((x) => x.slice(x.indexOf(":") + 1) === ans)
        || refs.find((x) => x.endsWith(":" + ans))
        || refs.find((x) => x === "Person:" + ans);
      if (!ansKey && /^eid_/.test(ans)) ansKey = "Person:" + ans;
      else if (!ansKey && /^https?:\/\/.+\/pull\/\d+/.test(ans)) ansKey = "PR:" + ans;
      else if (!ansKey && q.type === "company" && ansIds.has(ans)) ansKey = "Company:" + ans;   // a company is a property of Customer, not a node
      for (const k of refs) node(k);
      if (!ansKey) continue;
      node(ansKey);
      for (const k of refs) if (k !== ansKey && !rowAnswer.has(k)) rowAnswer.set(k, ansKey);
    }
    const types = cypherTypes(s.queries);
    for (const [src, t, dst] of s.edges || []) {
      if (!types.has(t) || !nodes.has(src) || !nodes.has(dst) || src === dst) continue;
      const ek = src + ">" + dst;
      if (!real.has(ek)) real.set(ek, { from: src, to: dst, types: new Set() });
      real.get(ek).types.add(t);
    }
  }
  const degree = new Map();
  for (const e of real.values()) {
    edges.set(e.from + ">" + e.to, { from: e.from, to: e.to, rel: [...e.types].sort().join(" + "), arrow: true });
    degree.set(e.from, 1); degree.set(e.to, 1);
  }
  // a node with no real edge keeps one dashed line to its row's answer node
  for (const [k, ak] of rowAnswer) {
    if (degree.has(k) || !nodes.has(ak) || nodes.get(ak).label === nodes.get(k).label) continue;
    edges.set(ak + ">" + k, { from: ak, to: k, rel: null, arrow: false });
  }
  // answer ids whose rows were not exported (the export keeps the first rows only) still belong in the picture
  let noRows = 0;
  (a5?.answer_ids || []).forEach((id) => {
    const key = q.type === "company" ? "Company:" + id : /^eid_/.test(id) ? "Person:" + id : /^https?:\/\/.+\/pull\/\d+/.test(id) ? "PR:" + id : null;
    if (key && !nodes.has(key)) { node(key); noRows++; }
  });
  if (!nodes.size) return null;
  for (const n of nodes.values()) {
    n.answer = ansIds.has(n.id);
    n.gold = hasGold && n.answer && goldIds.has(n.id);
    n.bad = hasGold && n.answer && !goldIds.has(n.id);
    n.name = n.label === "Company" ? n.id : shortName(n.label, n.key, names);
    n.sub = n.label;
  }
  // two nodes with one name (SupportForce/ans/3 has two Alice Davis): show the id so they can be told apart
  const seen = new Map();
  for (const n of nodes.values()) seen.set(n.label + "|" + n.name, (seen.get(n.label + "|" + n.name) || 0) + 1);
  for (const n of nodes.values()) if (n.label === "Person" && seen.get(n.label + "|" + n.name) > 1) n.sub = "Person " + n.id.replace(/^eid_/, "");
  return { nodes: [...nodes.values()], edges: [...edges.values()], goldAvailable: hasGold, noRows };
}

// ---------------------------------------------------------------- graph from a what-if person
export function buildWhatIfGraph(x) {
  const nodes = new Map(); const edges = [];
  const add = (key, label, name, extra = {}) => { if (!nodes.has(key)) nodes.set(key, { key, label, id: key, name, sub: label, order: nodes.size, answer: false, ...extra }); return nodes.get(key); };
  const p = add("P:" + x.person.eid, "Person", x.person.name, { col: 0, leaving: true });
  const owned = new Set();
  for (const it of x.newly_unowned) {
    const n = add(it.id, it.kind, it.title || it.id, { col: 1 }); n.col = 1; owned.add(it.id);
    edges.push({ from: p.key, to: n.key, rel: "only owner", arrow: true });
  }
  for (const a of x.affected) {
    const path = a.path || [];
    path.forEach((st, i) => {
      const col = owned.has(st.id) ? 1 : 1 + (path.length - 1 - i);
      const n = add(st.id, st.kind, st.title || st.id, { col });
      if (!owned.has(st.id)) n.col = Math.min(n.col, col);
      if (i > 0) edges.push({ from: path[i - 1].id, to: st.id, rel: a.rules?.[i - 1] || "depends on", arrow: true });
    });
  }
  const seen = new Set();
  const uniq = edges.filter((e) => { const k = e.from + ">" + e.to; if (seen.has(k)) return false; seen.add(k); return true; });
  return { nodes: [...nodes.values()], edges: uniq, whatif: true };
}

// ---------------------------------------------------------------- layout and drawing
function limit(g, cap) {
  const deg = new Map(g.nodes.map((n) => [n.key, 0]));
  for (const e of g.edges) { deg.set(e.from, deg.get(e.from) + 1); deg.set(e.to, deg.get(e.to) + 1); }
  if (g.nodes.length <= cap) return { nodes: g.nodes, edges: g.edges, more: 0 };
  const ranked = g.nodes.slice().sort((a, b) =>
    (b.answer - a.answer) || (b.leaving ? 1 : 0) - (a.leaving ? 1 : 0) || (deg.get(b.key) - deg.get(a.key)) || (a.order - b.order));
  const keep = new Set(ranked.slice(0, cap).map((n) => n.key));
  return {
    nodes: g.nodes.filter((n) => keep.has(n.key)),
    edges: g.edges.filter((e) => keep.has(e.from) && keep.has(e.to)),
    more: g.nodes.length - cap,
  };
}

export function renderGraph(g, opts = {}) {
  if (!g || !g.nodes.length) return "";
  const compact = !!opts.compact;
  const cap = opts.cap || (compact ? 14 : 30);
  const L = limit(g, cap);
  const GAP = compact ? 56 : 92, PAD = 10;
  const NAMEMAX = compact ? 21 : 26;
  const lines = new Map(L.nodes.map((n) => [n.key, compact ? [trunc(n.name, NAMEMAX)] : wrapName(n.name, NAMEMAX)]));
  const wrapped = [...lines.values()].some((l) => l.length > 1);
  const NH = wrapped ? 46 : 34, RH = NH + (compact ? 6 : 10);

  // columns
  const cols = [];
  const colOf = new Map();
  if (g.whatif) {
    const ids = [...new Set(L.nodes.map((n) => n.col))].sort((a, b) => a - b);
    ids.forEach((c, i) => { colOf.set(c, i); cols.push([]); });
    for (const n of L.nodes) cols[colOf.get(n.col)].push(n);
  } else {
    const labels = [];
    for (const n of L.nodes) if (!labels.includes(n.label)) labels.push(n.label);
    // events hang off a bug; once PRs are in the picture they go to the far side so Bug and PR stay adjacent
    const rank = (l) => (l === "BugEvent" && labels.includes("PR") ? (labels.includes("Customer") ? 6 : 1.5) : RANK[l] ?? 6);
    labels.sort((a, b) => (rank(a) - rank(b)) || labels.indexOf(a) - labels.indexOf(b));
    labels.forEach((l) => cols.push([]));
    for (const n of L.nodes) cols[labels.indexOf(n.label)].push(n);
  }
  cols.forEach((c, ci) => c.forEach((n) => { n.ci = ci; }));
  const tall = Math.max(...cols.map((c) => c.length));
  const colIdx = new Map(L.nodes.map((n) => [n.key, n.ci]));
  const gapCount = new Map(), gapInfo = new Map();   // edges between adjacent columns k and k+1
  let hasSkip = false;
  for (const e of L.edges) {
    const ca = colIdx.get(e.from), cb = colIdx.get(e.to), d = Math.abs(ca - cb);
    if (d === 1) {
      const k = Math.min(ca, cb);
      const g0 = gapInfo.get(k) || { n: 0, rels: new Set(), plain: 0 };
      g0.n++; if (e.rel) g0.rels.add(e.rel); else g0.plain++;
      gapInfo.set(k, g0); gapCount.set(k, g0.n);
    } else if (d > 1) hasSkip = true;
  }
  // many edges with one relation, or too many to label one by one: name the relation once above the gap
  const uniform = (k) => { const g0 = gapInfo.get(k); return g0 && g0.plain === 0 && g0.rels.size === 1; };
  const stripGap = (k) => uniform(k) && (gapCount.get(k) >= 3);
  const useStrip = [...gapCount.keys()].some(stripGap);
  const LANES = 5;
  const laneBase = useStrip ? 18 : 6;
  const headH = hasSkip ? laneBase + 4 + LANES * 4 : useStrip ? 18 : 6;
  const footH = hasSkip ? 8 + LANES * 4 : PAD;
  const H = headH + tall * RH + footH;
  const colW = cols.map((c) => Math.ceil(Math.max(112, ...c.map((n) => Math.max(Math.max(...lines.get(n.key).map((t) => t.length)) * 6.7 + 36, n.sub.length * 6.2 + 20)))));
  const colX = []; let acc = PAD;
  colW.forEach((w, i) => { colX.push(acc); acc += w + GAP; });
  const W = acc - GAP + PAD;
  const yOf = new Map();
  const place = () => cols.forEach((c) => {
    const off = headH + ((tall - c.length) * RH) / 2;
    c.forEach((n, i) => yOf.set(n.key, off + i * RH + RH / 2));
  });
  place();
  const nb = new Map(L.nodes.map((n) => [n.key, []]));
  for (const e of L.edges) { nb.get(e.from).push(e.to); nb.get(e.to).push(e.from); }
  const byKey = new Map(L.nodes.map((n) => [n.key, n]));
  const bary = (n) => {
    const ys = nb.get(n.key).map((k) => yOf.get(k));
    return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : yOf.get(n.key);
  };
  for (let it = 0; it < 6; it++) {
    const seq = it % 2 ? cols.map((_, i) => cols.length - 1 - i) : cols.map((_, i) => i);
    for (const ci of seq) {
      const c = cols[ci];
      const score = new Map(c.map((n) => [n.key, bary(n)]));
      c.sort((a, b) => (score.get(a.key) - score.get(b.key)) || a.order - b.order);
      place();
    }
  }
  const xOf = (ci) => colX[ci];

  // edges
  const defs = `<defs><marker id="gv-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L7 4 L1 7 z" class="gv-arrowhead"/></marker></defs>`;
  let ePaths = "", eText = "";
  const lane = { top: 0, bottom: 0 };
  const bodyTop = headH, bodyBot = headH + tall * RH;
  for (const e of L.edges) {
    const a = byKey.get(e.from), b = byKey.get(e.to);
    if (a.ci === b.ci) continue;
    const lo = a.ci < b.ci ? a : b, hi = lo === a ? b : a;
    const x1 = xOf(lo.ci) + colW[lo.ci], y1 = yOf.get(lo.key), x2 = xOf(hi.ci), y2 = yOf.get(hi.key);
    const arrow = e.arrow ? (e.to === hi.key ? ' marker-end="url(#gv-arrow)"' : ' marker-start="url(#gv-arrow)"') : "";
    const cls = `gv-edge${e.rel ? "" : " gv-unknown"}`;
    if (hi.ci - lo.ci > 1) {
      // skips a column: route through a lane above or below the nodes so it is not hidden behind them
      const top = (y1 + y2) / 2 <= (bodyTop + bodyBot) / 2;
      const i = lane[top ? "top" : "bottom"]++ % LANES;
      const ly = top ? laneBase + i * 4 : H - 6 - i * 4;
      ePaths += `<path class="${cls}" d="M${x1} ${y1} C${x1 + GAP * 0.6} ${y1} ${x1 + GAP * 0.4} ${ly} ${x1 + GAP * 0.9} ${ly} L${x2 - GAP * 0.9} ${ly} C${x2 - GAP * 0.4} ${ly} ${x2 - GAP * 0.6} ${y2} ${x2} ${y2}"${arrow}/>`;
      continue;
    }
    const dx = (x2 - x1) / 2;
    ePaths += `<path class="${cls}" d="M${x1} ${y1} C${x1 + dx} ${y1} ${x2 - dx} ${y2} ${x2} ${y2}"${arrow}/>`;
    if (e.rel) {
      if (!stripGap(lo.ci) && gapCount.get(lo.ci) <= 14) eText += `<text class="gv-rel" x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 + 3}" text-anchor="middle">${esc(e.rel)}</text>`;
    }
  }
  for (const k of gapCount.keys()) {
    if (stripGap(k)) eText += `<text class="gv-rel" x="${xOf(k) + colW[k] + GAP / 2}" y="12" text-anchor="middle">${esc([...gapInfo.get(k).rels][0])}</text>`;
  }

  // nodes
  let nodesSvg = "";
  for (const c of cols) for (const n of c) {
    const x = xOf(n.ci), y = yOf.get(n.key) - NH / 2, NW = colW[n.ci], ln = lines.get(n.key);
    const ny = ln.length > 1 ? 15 : (wrapped ? 19 : 14);
    const nameSvg = ln.map((t, i) => `<text class="gv-name" x="${x + 9}" y="${y + ny + i * 13}">${escClean(t)}</text>`).join("");
    const subY = y + ny + (ln.length - 1) * 13 + 13;
    const cls = `gv-node${n.answer ? " gv-answer" : ""}${n.leaving ? " gv-leaving" : ""}`;
    const mark = n.gold ? `<text class="gv-mark gv-ok" x="${x + NW - 7}" y="${y + 13}" text-anchor="end" aria-hidden="true">✓</text>`
      : n.bad ? `<text class="gv-mark gv-bad" x="${x + NW - 7}" y="${y + 13}" text-anchor="end" aria-hidden="true">×</text>` : "";
    nodesSvg += `<g class="${cls}"><title>${escClean(n.label)}: ${escClean(n.name)}${n.gold ? " (in the gold answer)" : ""}</title>
      <rect x="${x}" y="${y}" width="${NW}" height="${NH}" rx="6"/>
      ${nameSvg}<text class="gv-sub" x="${x + 9}" y="${subY}">${esc(n.sub)}</text>${mark}</g>`;
  }

  // text alternative
  const labelsInOrder = cols.map((c) => c[0].label);
  const alt = `${opts.title || "Graph of the path"}: ${L.nodes.length} nodes in ${cols.length} ${cols.length === 1 ? "column" : "columns"} (${[...new Set(labelsInOrder)].join(", ")}) and ${L.edges.length} edges${L.more ? `, with ${L.more} more nodes not drawn` : ""}.`;
  const nm = (k) => { const n = byKey.get(k); return `${n.name} (${n.label})`; };
  const edgeList = L.edges.slice(0, 80).map((e) => `<li>${escClean(nm(e.from))} ${e.rel ? esc(e.rel) + " " : "is in the same row as "}${escClean(nm(e.to))}</li>`).join("");

  const relNames = [...new Set(L.edges.flatMap((e) => (e.rel ? e.rel.split(" + ") : [])))].sort();
  const legend = g.whatif
    ? `Arrows point from the person to what they solely own, and from an item to the item it depends on; the label is the recorded link.`
    : `Solid: an edge in the graph of a type the tool's Cypher traverses. Dashed: returned in the same row.${relNames.length ? ` Types drawn: ${relNames.join(", ")}.` : ""}${g.goldAvailable ? " Check: in the gold answer." : ""} Blue: A5's answer.`;
  return `<figure class="gv${compact ? " gv-compact" : ""}">
    <div class="gv-scroll" tabindex="0" role="group" aria-label="${esc(alt)}">
      <svg class="gv-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(alt)}" focusable="false">${defs}${ePaths}${nodesSvg}${eText}</svg>
    </div>
    <ul class="sr-only" aria-label="Edges">${edgeList}</ul>
    <figcaption class="micro">${g.noRows ? `${g.noRows} answer nodes have no exported rows, so no edges. ` : ""}${L.more ? `The drawing stops at ${L.nodes.length} nodes, and ${L.more} more are not drawn. ` : ""}${legend}</figcaption>
  </figure>`;
}
