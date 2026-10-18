// Cypher display: line breaks before top-level clauses (whitespace only, the query text is unchanged otherwise)
// and light highlighting. Parameters ($name) use the A5 accent: they are the only part the model chooses;
// the rest of each query is fixed in the tool.
import { esc } from "./common.js";

const CLAUSES = ["OPTIONAL MATCH", "MATCH", "WHERE", "WITH", "RETURN", "ORDER BY", "LIMIT", "UNWIND", "CALL", "YIELD", "SKIP"];
const KEYWORDS = new Set(["MATCH", "OPTIONAL", "WHERE", "RETURN", "WITH", "AND", "OR", "NOT", "IN", "IS", "NULL", "DISTINCT",
  "AS", "CALL", "YIELD", "ORDER", "BY", "LIMIT", "UNWIND", "CASE", "WHEN", "THEN", "ELSE", "END", "TRUE", "FALSE", "SKIP",
  "DESC", "ASC", "EXISTS", "CONTAINS", "STARTS", "ENDS", "XOR", "UNION", "ALL"]);

export function formatCypher(q) {
  const s = String(q || "").trim();
  let out = "";
  let depth = 0;
  let quote = null;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (quote) {
      out += c;
      if (c === "\\") { out += s[++i] ?? ""; continue; }
      if (c === quote) quote = null;
      continue;
    }
    if (c === "'" || c === '"') { quote = c; out += c; continue; }
    if (c === "(" || c === "[" || c === "{") depth++;
    if (c === ")" || c === "]" || c === "}") depth = Math.max(0, depth - 1);
    if (depth === 0 && i > 0 && /\s/.test(c)) {
      const rest = s.slice(i + 1);
      const hit = CLAUSES.find((k) => rest.startsWith(k) && !/[A-Za-z0-9_]/.test(rest[k.length] || ""));
      if (hit && !(hit === "MATCH" && /OPTIONAL$/.test(out))) { out = out.replace(/\s+$/, "") + "\n"; continue; }
      if (/^AND\b/.test(rest) && out.slice(out.lastIndexOf("\n") + 1).length > 60) {
        out = out.replace(/\s+$/, "") + "\n  "; continue;
      }
    }
    out += c;
  }
  return out;
}

const TOKEN = /('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|(\$[A-Za-z_][A-Za-z0-9_]*)|(:[A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*)|(\s+|[^\s])/g;

export function highlightCypher(q) {
  const text = formatCypher(q);
  let html = "";
  let m;
  TOKEN.lastIndex = 0;
  while ((m = TOKEN.exec(text))) {
    const [t, str, par, lbl, word] = m;
    if (str) html += `<span class="str">${esc(t)}</span>`;
    else if (par) html += `<span class="par">${esc(t)}</span>`;
    else if (lbl) html += `<span class="lbl">${esc(t)}</span>`;
    else if (word && KEYWORDS.has(word)) html += `<span class="kw">${esc(t)}</span>`;
    else html += esc(t);
  }
  return html;
}

// The first MATCH pattern of a query, for a one-line summary: "(p:Person)-[au:AUTHORED]->(d:Document)".
export function firstPattern(q) {
  const m = /MATCH\s+(.+?)(?:\s+WHERE\s|\s+RETURN\s|\s+MATCH\s|\s+WITH\s|$)/.exec(String(q || ""));
  if (m) return m[1];
  const c = /CALL\s+([\w.]+)/.exec(String(q || ""));
  return c ? `CALL ${c[1]}` : String(q || "").slice(0, 60);
}
