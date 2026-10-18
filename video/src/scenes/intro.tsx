import {AbsoluteFill, interpolate, spring, useCurrentFrame} from 'remotion';
import {fonts, layout, palette, radius} from '../theme';
import {Appear, clamp} from '../components/Motion';
import {Chip, Heading} from '../components/ui';
import {cuer, FPS, type SceneProps} from './types';

// Nine of the graph's node labels and their main edges, endpoints as in src/inherit/ontology.py
// (the site's schema figure). Coordinates are node centres inside the schema panel.
type NodeName = 'Person' | 'Message' | 'Meeting' | 'Document' | 'Feature' | 'PR' | 'Customer' | 'Bug' | 'BugEvent';
const NODES: Record<NodeName, [number, number]> = {
  Message: [390, 140], Meeting: [390, 440], Document: [390, 740], Person: [1040, 440],
  Feature: [1620, 140], PR: [1620, 440], Customer: [2100, 140], Bug: [2100, 440], BugEvent: [2100, 740],
};
const W = 300, H = 108, HUB_W = 360, HUB_H = 130;
const EDGES: [NodeName, NodeName, string, number][] = [
  ['Person', 'Message', 'POSTED', 0], ['Person', 'Meeting', 'ATTENDED', 0],
  ['Person', 'Document', 'AUTHORED', 110], ['Person', 'Document', 'SUGGESTED', -110],
  ['Person', 'PR', 'OPENED', -120], ['Person', 'PR', 'REVIEWED', 120],
  ['Feature', 'PR', 'IMPLEMENTED_BY', 0], ['PR', 'Bug', 'FIXES', 0],
  ['Bug', 'Customer', 'REPORTED_BY', 0], ['Bug', 'BugEvent', 'HAS_EVENT', 0],
];
const size = (n: NodeName) => (n === 'Person' ? [HUB_W, HUB_H] : [W, H]);
function clip(n: NodeName, tx: number, ty: number) {
  const [x, y] = NODES[n];
  const [w, h] = size(n);
  const dx = tx - x, dy = ty - y;
  const s = Math.min((w / 2 + 10) / Math.abs(dx || 1e-9), (h / 2 + 10) / Math.abs(dy || 1e-9));
  return [x + dx * s, y + dy * s];
}
function geometry(a: NodeName, b: NodeName, bend: number) {
  const [ax, ay] = NODES[a], [bx, by] = NODES[b];
  const mx = (ax + bx) / 2, my = (ay + by) / 2, len = Math.hypot(bx - ax, by - ay);
  const cx = mx + (-(by - ay) / len) * bend, cy = my + ((bx - ax) / len) * bend;
  const [x1, y1] = clip(a, cx, cy), [x2, y2] = clip(b, cx, cy);
  return {d: `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`, lx: 0.25 * x1 + 0.5 * cx + 0.25 * x2, ly: 0.25 * y1 + 0.5 * cy + 0.25 * y2};
}

const PANEL = {left: layout.margin, top: 330, width: layout.width, height: 880};

export function Intro(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const turns = cue('It turns');
  const lit: Record<NodeName, number> = {
    Person: cue('people'), Bug: cue('bugs'), BugEvent: cue('bugs') + 4, Customer: cue('bugs') + 8,
    Document: cue('documents'), Meeting: cue('documents') + 4, Message: cue('documents') + 8,
    PR: cue('pull requests'), Feature: cue('pull requests') + 4,
  };
  const edgesAt = cue('who did what');
  // An edge draws once both of its ends are lit; "who did what to which" darkens every edge and label.
  const edgeAt = (a: NodeName, b: NodeName) => Math.max(lit[a], lit[b]) + 4;
  const dark = frame >= edgesAt;
  const big = interpolate(frame, [turns - 8, turns + 6], [1, 0], clamp);
  return <AbsoluteFill style={{background: palette.page}}>
    {big > 0 && <div style={{position: 'absolute', left: 220, top: 330, opacity: big}}>
      <Appear at={0}><div style={{fontFamily: fonts.sans, fontSize: 56, fontWeight: 500, color: palette.ink3}}>Zeyu Sun · Graph Hacks 2026, Track 3: Company Brain</div></Appear>
      <Appear at={cue('inherit-brain')}><div style={{fontFamily: fonts.sans, fontSize: 250, fontWeight: 700, letterSpacing: -9, lineHeight: 1, color: palette.ink, marginTop: 40}}>inherit-brain</div></Appear>
      <Appear at={cue('a company brain')}><div style={{fontFamily: fonts.sans, fontSize: 84, fontWeight: 500, color: palette.ink2, marginTop: 44}}>a company brain on FalkorDB</div></Appear>
    </div>}
    {frame >= turns - 8 && <div style={{opacity: 1 - big}}>
      <Heading kicker="inherit-brain, a company brain on FalkorDB" title="One graph per product: who did what to which"/>
      <div style={{position: 'absolute', right: layout.margin, top: 250}}><Chip mono size={38}>herb:&lt;product&gt; · 30 graphs</Chip></div>
      <div data-schema style={{position: 'absolute', ...PANEL}}>
        <svg width={PANEL.width} height={PANEL.height} style={{position: 'absolute', inset: 0, overflow: 'visible'}}>
          <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="16" markerHeight="16" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill={palette.ink2}/></marker><marker id="arr-soft" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="16" markerHeight="16" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill={palette.mark}/></marker></defs>
          {EDGES.map(([a, b, , bend], i) => {
            const at = edgeAt(a, b);
            const t = interpolate(frame, [at, at + 14], [0, 1], clamp);
            if (t <= 0) return null;
            return <path key={i} d={geometry(a, b, bend).d} fill="none" stroke={dark ? palette.ink2 : palette.mark} strokeWidth={dark ? 5 : 4} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - t} markerEnd={t > 0.95 ? (dark ? 'url(#arr)' : 'url(#arr-soft)') : undefined}/>;
          })}
        </svg>
        {EDGES.map(([a, b, label, bend]) => {
          const g = geometry(a, b, bend);
          const at = edgeAt(a, b) + 8;
          if (frame < at) return null;
          return <div key={label} data-edge={label} style={{position: 'absolute', left: g.lx, top: g.ly, transform: 'translate(-50%, -50%)', fontFamily: fonts.mono, fontSize: 34, fontWeight: dark ? 600 : 500, color: dark ? palette.ink : palette.ink3, background: palette.page, padding: '2px 12px', borderRadius: radius, whiteSpace: 'nowrap', opacity: interpolate(frame, [at, at + 8], [0, 1], clamp)}}>{label}</div>;
        })}
        {(Object.keys(NODES) as NodeName[]).map((n) => {
          const [x, y] = NODES[n];
          const [w, h] = size(n);
          const on = frame >= lit[n];
          const pop = on ? spring({frame: frame - lit[n], fps: FPS, config: {damping: 14, stiffness: 180}, durationInFrames: 14}) : 0;
          return <div key={n} data-node={n} data-lit={on ? 'true' : 'false'} style={{position: 'absolute', left: x - w / 2, top: y - h / 2, width: w, height: h, boxSizing: 'border-box', display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: palette.surface, border: `${n === 'Person' ? 4 : 3}px solid ${on ? palette.ink : palette.lineStrong}`, borderRadius: radius,
            fontFamily: fonts.sans, fontSize: n === 'Person' ? 58 : 48, fontWeight: 650, color: on ? palette.ink : palette.mark,
            opacity: on ? 1 : 0.45, transform: `scale(${1 + 0.06 * Math.sin(Math.PI * Math.min(1, pop))})`}}>{n}</div>;
        })}
      </div>
      <Appear at={edgesAt} style={{position: 'absolute', left: layout.margin, top: 1150}}>
        <div style={{fontFamily: fonts.sans, fontSize: 36, color: palette.ink3}}>9 of the graph's 16 node labels and their main edges; the site draws all of them.</div>
      </Appear>
    </div>}
  </AbsoluteFill>;
}
