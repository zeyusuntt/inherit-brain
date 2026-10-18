// SVG charts drawn at the container's real pixel width (no scaled text). A5 rows use the accent; everything else grey.
import { f1, s1, int, onWidth } from "./common.js";

const lin = (d0, d1, r0, r1) => (x) => r0 + ((x - d0) / (d1 - d0)) * (r1 - r0);

// ---------------------------------------------------------------- forest plot (one SVG per table row, shared scale)
export const FOREST_DOMAIN = [-30, 100];
export const FOREST_TICKS = [-25, 0, 25, 50, 75, 100];

export function forestRow(el, { lo, hi, diff, threshold, a5 }) {
  onWidth(el, (w) => {
    const pad = 8;
    const x = lin(FOREST_DOMAIN[0], FOREST_DOMAIN[1], pad, w - pad);
    const h = 22, y = 11;
    const cls = a5 ? "a5" : "o";
    let g = FOREST_TICKS.map((t) => `<line class="${t === 0 ? "zero" : "grid"}" x1="${x(t)}" x2="${x(t)}" y1="0" y2="${h}"/>`).join("");
    if (threshold !== null && threshold !== undefined) {
      g += `<line class="thr" x1="${x(threshold)}" x2="${x(threshold)}" y1="2" y2="${h - 2}"/>`;
    }
    g += `<line class="ci-${cls}" x1="${x(lo)}" x2="${x(hi)}" y1="${y}" y2="${y}"/>`;
    g += a5
      ? `<circle class="pt-a5" cx="${x(diff)}" cy="${y}" r="5"/>`
      : `<rect class="pt-o" x="${x(diff) - 4}" y="${y - 4}" width="8" height="8"/>`;
    el.innerHTML = `<svg class="chart-svg plot-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true">${g}</svg>`;
  });
}

export function forestAxis(el) {
  onWidth(el, (w) => {
    const pad = 8;
    const x = lin(FOREST_DOMAIN[0], FOREST_DOMAIN[1], pad, w - pad);
    const g = FOREST_TICKS.map((t) => {
      const anchor = t === FOREST_TICKS[FOREST_TICKS.length - 1] ? "end" : t === FOREST_TICKS[0] ? "start" : "middle";
      const tx = anchor === "end" ? x(t) + pad : anchor === "start" ? x(t) - pad + 2 : x(t);
      return `<text x="${tx}" y="13" text-anchor="${anchor}">${t > 0 ? "+" + t : t < 0 ? "−" + Math.abs(t) : "0"}</text>`;
    }).join("");
    el.innerHTML = `<svg class="chart-svg axis-svg" width="${w}" height="20" viewBox="0 0 ${w} 20" aria-hidden="true">${g}</svg>`;
  });
}

// ---------------------------------------------------------------- score against tokens (log x)
export function scatter(el, points) {
  onWidth(el, (w) => {
    const narrow = w < 520;
    const h = narrow ? 300 : 340;
    const m = { l: 40, r: 18, t: 30, b: 44 };
    const lx = (v) => Math.log10(v);
    const xd = [lx(3000), lx(200000)];
    const x = (v) => lin(xd[0], xd[1], m.l, w - m.r)(lx(v));
    const y = lin(0, 100, h - m.b, m.t);
    const xt = [5000, 10000, 25000, 50000, 100000, 200000];
    const yt = [0, 25, 50, 75, 100];
    let g = "";
    g += yt.map((t) => `<line class="grid" x1="${m.l}" x2="${w - m.r}" y1="${y(t)}" y2="${y(t)}"/><text x="${m.l - 8}" y="${y(t) + 4}" text-anchor="end">${t}</text>`).join("");
    g += xt.map((t) => `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${m.t}" y2="${h - m.b}"/><text x="${x(t)}" y="${h - m.b + 18}" text-anchor="middle">${t >= 1000 ? t / 1000 + "k" : t}</text>`).join("");
    g += `<text x="${(m.l + w - m.r) / 2}" y="${h - 6}" text-anchor="middle">Median input tokens per question (log scale)</text>`;
    g += `<text x="${m.l - 34}" y="12" text-anchor="start">Primary endpoint</text>`;
    for (const p of points) {
      const px = x(p.tokens), py = y(p.score);
      const right = px < w * 0.68;
      const lx2 = right ? px + 11 : px - 11;
      const anchor = right ? "start" : "end";
      if (p.a5) g += `<circle class="pt-a5" cx="${px}" cy="${py}" r="6.5"/>`;
      else g += `<circle class="ring-o" cx="${px}" cy="${py}" r="5.5"/>`;
      g += `<text class="${p.a5 ? "lab-a5" : "lab"}" x="${lx2}" y="${py - 2}" text-anchor="${anchor}">${p.label}</text>`;
      g += `<text x="${lx2}" y="${py + 13}" text-anchor="${anchor}">${f1(p.score)} at ${int(p.tokens)}</text>`;
    }
    el.innerHTML = `<svg class="chart-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Primary endpoint against median input tokens. ${points.map((p) => `${p.label}: ${f1(p.score)} at ${int(p.tokens)} tokens`).join("; ")}.">${g}</svg>`;
  });
}

// ---------------------------------------------------------------- seen / unseen dumbbell
export function dumbbell(el, rows) {
  onWidth(el, (w) => {
    const rowH = 52, top = 8, h = top + rows.length * rowH + 22;
    const m = { l: 6, r: 64 };
    const x = lin(0, 100, m.l, w - m.r);
    let g = [0, 25, 50, 75, 100].map((t) => `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${top}" y2="${h - 20}"/><text x="${x(t)}" y="${h - 4}" text-anchor="${t === 0 ? "start" : t === 100 ? "end" : "middle"}">${t}</text>`).join("");
    rows.forEach((r, i) => {
      const y = top + i * rowH + 32;
      g += `<text class="lab" x="${m.l}" y="${y - 16}">${r.label}</text>`;
      g += `<line class="ci-o" x1="${x(Math.min(r.a0, r.a5))}" x2="${x(Math.max(r.a0, r.a5))}" y1="${y}" y2="${y}" style="stroke: var(--mark-soft)"/>`;
      g += `<circle class="ring-o" cx="${x(r.a0)}" cy="${y}" r="5"/>`;
      g += `<circle class="pt-a5" cx="${x(r.a5)}" cy="${y}" r="6"/>`;
      const a0Left = r.a0 < r.a5;
      g += `<text x="${x(r.a0) + (a0Left ? -9 : 9)}" y="${y + 4}" text-anchor="${a0Left ? "end" : "start"}">A0 ${f1(r.a0)}</text>`;
      g += `<text class="lab-a5" x="${x(r.a5) + (a0Left ? 10 : -10)}" y="${y + 4}" text-anchor="${a0Left ? "start" : "end"}">A5 ${f1(r.a5)}</text>`;
    });
    el.innerHTML = `<svg class="chart-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${rows.map((r) => `${r.label}: A5 ${f1(r.a5)}, A0 ${f1(r.a0)}`).join("; ")}">${g}</svg>`;
  });
}

// ---------------------------------------------------------------- one interval with threshold and prior
export function intervalStrip(el, { diff, lo, hi, threshold, prior, max = 60 }) {
  onWidth(el, (w) => {
    const h = 64, y = 26;
    const x = lin(-5, max, 8, w - 10);
    let g = "";
    for (let t = 0; t <= max; t += 10) g += `<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="8" y2="${y + 12}"/><text x="${x(t)}" y="${h - 6}" text-anchor="middle">${t > 0 ? "+" + t : t}</text>`;
    g += `<line class="thr" x1="${x(threshold)}" x2="${x(threshold)}" y1="4" y2="${y + 14}"/>`;
    g += `<text x="${x(threshold) + 5}" y="11" text-anchor="start">threshold 0</text>`;
    if (prior !== null && prior !== undefined) {
      g += `<line class="grid" x1="${x(prior)}" x2="${x(prior)}" y1="4" y2="${y + 14}" style="stroke: var(--ink-3); stroke-dasharray: 1 3"/>`;
      g += `<text x="${x(prior) + 5}" y="11" text-anchor="start">prior ${prior}</text>`;
    }
    g += `<line class="ci-a5" x1="${x(lo)}" x2="${x(hi)}" y1="${y}" y2="${y}" style="stroke-width:3"/>`;
    g += `<circle class="pt-a5" cx="${x(diff)}" cy="${y}" r="6"/>`;
    el.innerHTML = `<svg class="chart-svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="A5 minus A0: ${s1(diff)}, 95% interval ${f1(lo)} to ${f1(hi)}; threshold ${threshold}">${g}</svg>`;
  });
}
