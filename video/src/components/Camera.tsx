import {Easing, Img, interpolate, staticFile, useCurrentFrame} from 'remotion';
import type {ReactNode} from 'react';
import {captures, type CaptureName} from '../content';
import {fonts, palette, radius} from '../theme';

export type Rect = {x: number; y: number; w: number; h: number};
export type Key = {at: number; rect: Rect};
export type Mark = {rect: Rect; at: number; until?: number; color?: string; kind?: 'box' | 'underline' | 'wash'; label?: string; labelSide?: 'top' | 'bottom' | 'right'};

type Box = {x: number; y: number; width: number; height: number};
const toRect = (b: Box): Rect => ({x: b.x, y: b.y, w: b.width, h: b.height});

// The box of a text in a capture (CSS pixels of the captured page), the nth match.
export function needle(name: CaptureName, text: string, nth = 0): Rect {
  const found = (captures[name].needles as (Box & {needle: string})[]).filter((b) => b.needle === text);
  const b = found[nth];
  if (!b) throw new Error(`capture ${name}: no text box for "${text}" (#${nth})`);
  return toRect(b);
}
// The box of an element whose class (or tag) matches and whose text contains `contains`, the nth match.
export function element(name: CaptureName, cls: string, contains = '', nth = 0): Rect {
  const found = (captures[name].boxes as (Box & {tag: string; cls: string; text: string})[])
    .filter((b) => (b.tag === cls || b.cls.split(' ').includes(cls)) && b.text.includes(contains));
  const b = found[nth];
  if (!b) throw new Error(`capture ${name}: no element .${cls} containing "${contains}" (#${nth})`);
  return toRect(b);
}
export const pad = (r: Rect, px: number, py = px): Rect => ({x: r.x - px, y: r.y - py, w: r.w + 2 * px, h: r.h + 2 * py});
export const union = (...rs: Rect[]): Rect => {
  const x = Math.min(...rs.map((r) => r.x)), y = Math.min(...rs.map((r) => r.y));
  return {x, y, w: Math.max(...rs.map((r) => r.x + r.w)) - x, h: Math.max(...rs.map((r) => r.y + r.h)) - y};
};
export const rect = (x: number, y: number, w: number, h: number): Rect => ({x, y, w, h});

const ease = Easing.bezier(0.45, 0, 0.2, 1);
const MOVE = 22;

// A real page, framed by a camera that moves between rectangles of the page (CSS pixels of the
// full-page capture). Marks are highlight boxes on real elements, drawn in page coordinates.
// `fadeEdges` fades the left and right edges, so a crop that cuts lines of text reads as deliberate.
export function Camera({name, keys, marks = [], width, height, children, full = true, fadeEdges = false}: {name: CaptureName; keys: Key[]; marks?: Mark[]; width: number; height: number; children?: ReactNode; full?: boolean; fadeEdges?: boolean}) {
  const frame = useCurrentFrame();
  const size = full ? captures[name].full : captures[name].viewport;
  let i = 0;
  while (i < keys.length - 1 && frame >= keys[i + 1].at) i++;
  const a = keys[i], b = keys[Math.min(i + 1, keys.length - 1)];
  const t = b === a ? 0 : interpolate(frame, [b.at - MOVE, b.at], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease});
  const r = {x: a.rect.x + (b.rect.x - a.rect.x) * t, y: a.rect.y + (b.rect.y - a.rect.y) * t, w: a.rect.w + (b.rect.w - a.rect.w) * t, h: a.rect.h + (b.rect.h - a.rect.h) * t};
  // A slow push-in while a key holds, so a held page never sits perfectly still.
  const held = frame - (keys[i].at);
  const push = 1 + Math.min(0.03, Math.max(0, held) * 0.0004);
  const s = Math.min(width / r.w, height / r.h) * push;
  const left = width / 2 - (r.x + r.w / 2) * s;
  const top = height / 2 - (r.y + r.h / 2) * s;
  return <div data-camera={name} style={{width, height, position: 'relative', overflow: 'hidden', borderRadius: radius, background: palette.page, border: `2px solid ${palette.line}`, boxShadow: '0 18px 60px rgba(22,24,28,0.08)'}}>
    <div style={{position: 'absolute', inset: 0, ...(fadeEdges ? edgeFade : {})}}>
    <div style={{position: 'absolute', left, top, width: size.w * s, height: size.h * s}}>
      <Img src={staticFile(`captures/${name}${full ? '-full' : ''}.png`)} style={{position: 'absolute', inset: 0, width: '100%', height: '100%'}}/>
      {marks.map((m, k) => {
        if (frame < m.at || (m.until !== undefined && frame >= m.until)) return null;
        const p = interpolate(frame, [m.at, m.at + 10], [0, 1], clamp01);
        const color = m.color ?? palette.accent;
        const style = m.kind === 'underline'
          ? {left: m.rect.x * s, top: (m.rect.y + m.rect.h) * s + 2, width: m.rect.w * s * p, height: 7, background: color, borderRadius: 3}
          : m.kind === 'wash'
            ? {left: m.rect.x * s - 6, top: m.rect.y * s - 4, width: m.rect.w * s + 12, height: m.rect.h * s + 8, background: color, opacity: 0.18 * p, borderRadius: radius}
            : {left: m.rect.x * s - 10, top: m.rect.y * s - 8, width: m.rect.w * s + 20, height: m.rect.h * s + 16, border: `6px solid ${color}`, borderRadius: radius + 2, opacity: p};
        const side = m.labelSide ?? 'top';
        const labelPos = side === 'top' ? {left: -6, bottom: '100%', marginBottom: 10} : side === 'bottom' ? {left: -6, top: '100%', marginTop: 10} : {left: '100%', top: -6, marginLeft: 16};
        return <div key={k} data-mark={m.label ?? 'box'} style={{position: 'absolute', ...style}}>
          {m.label && <div style={{position: 'absolute', ...labelPos, whiteSpace: 'nowrap', fontFamily: fonts.sans, fontSize: 40, fontWeight: 600, color: palette.white, background: color, padding: '6px 18px', borderRadius: radius, opacity: p}}>{m.label}</div>}
        </div>;
      })}
    </div>
    </div>
    {children}
  </div>;
}
const clamp01 = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const edgeMask = 'linear-gradient(to right, transparent 0%, #000 6%, #000 94%, transparent 100%)';
const edgeFade = {maskImage: edgeMask, WebkitMaskImage: edgeMask} as const;
