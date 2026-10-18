// Schema diagram. Labels, edge types and their endpoints follow src/inherit/ontology.py (SIGNATURES); counts are the
// totals over the 30 product graphs on FalkorDB 6.0.2 (results/freeze/load_report.json), checked by check.mjs.
import { int, esc } from "./common.js";

export const NODE_COUNTS = {
  Message: 33632, Person: 6402, FeatureEvent: 4007, PR: 3562, BugEvent: 2569, Feature: 1757, ExternalBug: 900,
  Bug: 899, Observation: 674, Url: 606, Customer: 521, Document: 400, Meeting: 321, Channel: 302, Competitor: 126, Product: 30,
};
export const EDGE_COUNTS = {
  IN_CHANNEL: 33632, POSTED: 32350, COLLABORATED: 26450, MENTIONS: 12947, ATTENDED: 8884, HAS: 7065, HAS_EVENT: 6576,
  DISCUSSES: 4737, REVIEWED: 4609, SHARED: 4593, SUGGESTED: 4399, SPOKE_IN: 3693, OPENED: 3562, ACTED_BY: 2569,
  CONTRIBUTED: 2093, ABOUT: 1957, REPORTED_BY: 1012, ASSIGNED_TO: 979, SIMILAR_TO: 899, FIXES: 488, STATED: 419,
  AUTHORED: 400, IMPLEMENTED_BY: 296, REVERTS: 89,
};
export const EDGES_NOT_DRAWN = [
  { type: "HAS", ends: "Product to bugs, documents, meetings, features, PRs, competitors", count: EDGE_COUNTS.HAS },
  { type: "SHARED", ends: "Person to Url, Document, PR", count: EDGE_COUNTS.SHARED },
  { type: "SUGGESTED", ends: "Person to Document (review suggestion or question)", count: EDGE_COUNTS.SUGGESTED },
  { type: "SPOKE_IN", ends: "Person to Meeting", count: EDGE_COUNTS.SPOKE_IN },
  { type: "ABOUT", ends: "PR to Feature; Url, Observation to Competitor, Product; Observation to Feature, Bug, Person; Message to PR", count: EDGE_COUNTS.ABOUT },
  { type: "STATED", ends: "Person to Observation", count: EDGE_COUNTS.STATED },
];

const N = {
  Person: [540, 300, 150, 56, true],
  Message: [210, 300], Channel: [210, 140], Document: [240, 500], Meeting: [480, 530],
  Feature: [540, 100], FeatureEvent: [305, 46],
  PR: [830, 112], Bug: [880, 320], ExternalBug: [1010, 210], Customer: [1010, 450], BugEvent: [780, 500],
  Product: [620, 604, 112], Observation: [745, 604, 112], Competitor: [870, 604, 112], Url: [995, 604, 112],
};
// [from, to, type, bend]; bend is the control-point offset to the left of the direction of travel.
const E = [
  ["Person", "Document", "AUTHORED", 0],
  ["Person", "Meeting", "ATTENDED", 0],
  ["Person", "Message", "POSTED", 40],
  ["Message", "Person", "MENTIONS", 40],
  ["Message", "Channel", "IN_CHANNEL", 0],
  ["Message", "Feature", "DISCUSSES", 0],
  ["Person", "Feature", "CONTRIBUTED", 0],
  ["Feature", "FeatureEvent", "HAS_EVENT", 0, "nocount"],
  ["Person", "PR", "OPENED", 34],
  ["Person", "PR", "REVIEWED", -34],
  ["Feature", "PR", "IMPLEMENTED_BY", 0],
  ["PR", "Bug", "FIXES", 0],
  ["Bug", "BugEvent", "HAS_EVENT", 0, "nocount"],
  ["BugEvent", "Person", "ACTED_BY", 36],
  ["BugEvent", "Person", "ASSIGNED_TO", -36],
  ["Bug", "Customer", "REPORTED_BY", 0],
  ["ExternalBug", "Bug", "SIMILAR_TO", 0],
];

const box = (name) => {
  const [x, y, w = 128, h = 44, hub = false] = N[name];
  return { x, y, w, h, hub };
};
function clip(b, tx, ty) {
  const dx = tx - b.x, dy = ty - b.y;
  const hw = b.w / 2 + 4, hh = b.h / 2 + 4;
  const s = Math.min(hw / Math.abs(dx || 1e-9), hh / Math.abs(dy || 1e-9));
  return [b.x + dx * s, b.y + dy * s];
}
const r = (n) => Math.round(n * 10) / 10;

function label(x, y, type, count) {
  const text = count ? `${type} ${int(count)}` : type;
  const w = text.length * 6.7 + 8;
  return `<rect class="e-bg" x="${r(x - w / 2)}" y="${r(y - 8)}" width="${r(w)}" height="15" rx="3"/>` +
    `<text class="e-label" x="${r(x)}" y="${r(y + 3.5)}" text-anchor="middle"${count ? ` data-calc-edge="${type}"` : ""}>${esc(text)}</text>`;
}

export function renderSchema(el) {
  if (!el) return;
  let edges = "", labels = "";
  for (const [a, b, type, bend, flag] of E) {
    const A = box(a), B = box(b);
    const mx = (A.x + B.x) / 2, my = (A.y + B.y) / 2;
    const dx = B.x - A.x, dy = B.y - A.y, len = Math.hypot(dx, dy);
    const cx = mx + (-dy / len) * bend, cy = my + (dx / len) * bend;
    const [x1, y1] = clip(A, cx, cy);
    const [x2, y2] = clip(B, cx, cy);
    edges += `<path class="e-line" d="M${r(x1)},${r(y1)} Q${r(cx)},${r(cy)} ${r(x2)},${r(y2)}" marker-end="url(#arr)"/>`;
    const lx = 0.25 * x1 + 0.5 * cx + 0.25 * x2, ly = 0.25 * y1 + 0.5 * cy + 0.25 * y2;
    labels += label(lx, ly, type, flag === "nocount" ? null : EDGE_COUNTS[type]);
  }
  // self loops: COLLABORATED on Person (right side), REVERTS on PR (top)
  const P = box("Person");
  edges += `<path class="e-line" d="M${P.x + 79},${P.y - 10} C${P.x + 135},${P.y - 58} ${P.x + 135},${P.y + 58} ${P.x + 79},${P.y + 10}" marker-end="url(#arr)"/>`;
  labels += `<rect class="e-bg" x="${P.x + 126}" y="${P.y - 8}" width="${"COLLABORATED 26,450".length * 6.7 + 8}" height="15" rx="3"/><text class="e-label" x="${P.x + 130}" y="${P.y + 3.5}" data-calc-edge="COLLABORATED">COLLABORATED ${int(EDGE_COUNTS.COLLABORATED)}</text>`;
  const R = box("PR");
  edges += `<path class="e-line" d="M${R.x - 22},${R.y - 26} C${R.x - 44},${R.y - 86} ${R.x + 44},${R.y - 86} ${R.x + 22},${R.y - 26}" marker-end="url(#arr)"/>`;
  labels += label(R.x, R.y - 78, "REVERTS", EDGE_COUNTS.REVERTS);

  let nodes = "";
  for (const name of Object.keys(N)) {
    const b = box(name);
    nodes += `<g><rect class="n-box${b.hub ? " hub" : ""}" x="${b.x - b.w / 2}" y="${b.y - b.h / 2}" width="${b.w}" height="${b.h}" rx="6"/>` +
      `<text class="n-label" x="${b.x}" y="${b.y - 2}" text-anchor="middle"${b.hub ? ' style="font-size:15px"' : ""}>${name}</text>` +
      `<text class="n-count" x="${b.x}" y="${b.y + 14}" text-anchor="middle" data-calc-node="${name}">${int(NODE_COUNTS[name])}</text></g>`;
  }
  const note = `<text class="aside-note" x="564" y="566">Also in the graph, linked by HAS, ABOUT, STATED and SHARED (table below)</text>`;
  el.innerHTML = `<svg viewBox="124 0 956 640" role="img" aria-labelledby="schema-t schema-d">
    <title id="schema-t">Graph schema: 16 node labels and the main edge types</title>
    <desc id="schema-d">Person is the hub: AUTHORED documents, ATTENDED meetings, POSTED messages (messages are IN_CHANNEL a channel and MENTION people), OPENED and REVIEWED PRs, CONTRIBUTED to features, COLLABORATED with people. PRs FIX bugs and REVERT PRs; features are IMPLEMENTED_BY PRs; bugs and features HAVE_EVENT histories; bug events are ACTED_BY and ASSIGNED_TO people; bugs are REPORTED_BY customers; external bugs are SIMILAR_TO bugs; messages DISCUSS features.</desc>
    <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path class="arrow" d="M0,1 L10,5 L0,9 z"/></marker></defs>
    ${edges}${labels}${nodes}${note}</svg>`;
}
