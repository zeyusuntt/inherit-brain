import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette, radius} from '../theme';
import {Appear, clamp, useProgress} from '../components/Motion';
import {Heading, thousands} from '../components/ui';
import {cuer, type SceneProps} from './types';

const X0 = 560, X1 = 2380, ROW0 = 300, ROW_H = 140;
const xOf = (v: number) => X0 + (X1 - X0) * v / 100;
const name = (id: string) => facts.arms.find((a) => a.id === id)!.name;

function ScoreRow({index, value, label, self, at}: {index: number; value: number; label: string; self: boolean; at: number}) {
  const frame = useCurrentFrame();
  const t = useProgress(at, 18);
  if (frame < at) return null;
  const y = ROW0 + index * ROW_H;
  const color = self ? palette.accent : palette.ink3;
  return <div data-score={value.toFixed(1)}>
    <div style={{position: 'absolute', left: layout.margin, top: y, width: 380, fontFamily: fonts.sans, fontSize: 120, fontWeight: 700, letterSpacing: -3, lineHeight: 1, color, fontVariantNumeric: 'tabular-nums', opacity: Math.min(1, t * 2)}}>{(value * t).toFixed(1)}</div>
    <div style={{position: 'absolute', left: X0, top: y - 4, fontFamily: fonts.sans, fontSize: 44, fontWeight: self ? 650 : 500, color: self ? palette.ink : palette.ink2}}>{label}</div>
    <div style={{position: 'absolute', left: X0, top: y + 80, width: (xOf(value) - X0) * t, height: self ? 8 : 6, background: color, borderRadius: 4}}/>
    <div style={{position: 'absolute', left: X0 + (xOf(value) - X0) * t - 20, top: y + 64 + (self ? 0 : -1), width: 40, height: 40, borderRadius: 20, background: color}}/>
  </div>;
}

function Panel({children, at, until}: {children: React.ReactNode; at: number; until?: number}) {
  const frame = useCurrentFrame();
  if (frame < at || (until !== undefined && frame >= until)) return null;
  return <Appear at={at} style={{position: 'absolute', left: layout.margin, right: layout.margin, top: 820, height: 390}}>{children}</Appear>;
}

// The three scores on a 0 to 100 axis, each on its spoken number; then the token medians, the lead
// over HERB's tools with its interval, and the seen and never-seen split.
export function Results(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const s = facts.scores, tk = facts.tokens, h1 = facts.h1, seen = facts.by_seen.seen, unseen = facts.by_seen.unseen;
  const twelve = cue('twelve times'), swap = cue('Swap the graph tools'), forty = cue('forty points'), wordings = cue('On wordings'), twenty = cue('twenty points');
  const gap = useProgress(forty, 16);
  const yA0 = ROW0 + ROW_H + 84;
  return <AbsoluteFill style={{background: palette.page}}>
    <Heading kicker={`Held-out, ${thousands(facts.n_questions)} questions, ${facts.n_products_held_out} products`} title="Primary endpoint, F1 x 100" size={56}/>
    <ScoreRow index={0} value={s.A5} label={name('A5')} self at={cue('eighty')}/>
    <ScoreRow index={1} value={s.A0} label={name('A0')} self={false} at={cue('thirty-nine')}/>
    <ScoreRow index={2} value={s.A1f} label={name('A1f')} self={false} at={cue('twenty-six')}/>
    <div style={{position: 'absolute', left: X0, width: X1 - X0, top: 740, height: 2, background: palette.lineStrong}}/>
    {[0, 50, 100].map((v) => <div key={v} style={{position: 'absolute', left: xOf(v), top: 752, transform: 'translateX(-50%)', fontFamily: fonts.mono, fontSize: 34, color: palette.ink3}}>{v}</div>)}
    {frame >= forty && <div data-gap style={{position: 'absolute', left: xOf(s.A0), top: yA0 - 2, width: (xOf(s.A5) - xOf(s.A0)) * gap, borderTop: `6px dashed ${palette.ink}`}}>
      <div style={{position: 'absolute', right: 0, top: -76, fontFamily: fonts.sans, fontSize: 48, fontWeight: 700, color: palette.ink, whiteSpace: 'nowrap', opacity: gap}}>+{h1.diff}</div>
    </div>}

    <Panel at={twelve} until={swap}>
      <div style={{fontFamily: fonts.sans, fontSize: 42, color: palette.ink3}}>Median input tokens per question</div>
      {[{label: name('A5'), v: tk.A5, self: true}, {label: name('A1f'), v: tk.A1f, self: false}].map((r, i) => <div key={r.label} style={{display: 'flex', alignItems: 'center', gap: 30, marginTop: i ? 22 : 30}}>
        <span style={{width: 700, whiteSpace: 'nowrap', fontFamily: fonts.sans, fontSize: 42, fontWeight: r.self ? 650 : 500, color: r.self ? palette.ink : palette.ink2}}>{r.label}</span>
        <span style={{height: 44, width: 860 * r.v / tk.A1f, background: r.self ? palette.accent : palette.markSoft, borderRadius: radius}}/>
        <span style={{fontFamily: fonts.mono, fontSize: 44, color: palette.ink, fontVariantNumeric: 'tabular-nums'}}>{thousands(r.v)}</span>
      </div>)}
      <div style={{position: 'absolute', right: 0, top: 30, textAlign: 'right'}}>
        <div style={{fontFamily: fonts.sans, fontSize: 170, fontWeight: 700, letterSpacing: -5, lineHeight: 1, color: palette.ink}}>{tk.ratio}x</div>
        <div style={{fontFamily: fonts.sans, fontSize: 38, color: palette.ink3, marginTop: 6}}>the input, scoring {s.A1f.toFixed(1)}</div>
      </div>
    </Panel>

    <Panel at={swap} until={wordings}>
      <div style={{fontFamily: fonts.sans, fontSize: 46, fontWeight: 500, color: palette.ink2}}>Swap the graph tools for HERB's retrieval tools: same model, same loop</div>
      {frame >= forty && <Appear at={forty} style={{display: 'flex', alignItems: 'baseline', gap: 44, marginTop: 30}}>
        <span data-h1 style={{fontFamily: fonts.sans, fontSize: 170, fontWeight: 700, letterSpacing: -5, color: palette.ink, lineHeight: 1}}>+{h1.diff}</span>
        <span style={{fontFamily: fonts.sans, fontSize: 56, fontWeight: 500, color: palette.ink2}}>points <span style={{fontFamily: fonts.mono}}>({h1.lo} to {h1.hi})</span></span>
        <span style={{fontFamily: fonts.sans, fontSize: 38, color: palette.ink3}}>95% interval, product-clustered bootstrap</span>
      </Appear>}
    </Panel>

    <Panel at={wordings}>
      <div style={{fontFamily: fonts.sans, fontSize: 42, color: palette.ink3}}>Lead over HERB's retrieval tools, split by whether the question's wording appeared in development</div>
      <div style={{display: 'flex', gap: 90, marginTop: 30}}>
        <div style={{flex: 1, borderLeft: `6px solid ${palette.markSoft}`, paddingLeft: 32}}>
          <div style={{fontFamily: fonts.sans, fontSize: 44, color: palette.ink3}}>Seen in development, {seen.n} questions</div>
          <div style={{display: 'flex', alignItems: 'baseline', gap: 28, marginTop: 10}}>
            <span style={{fontFamily: fonts.sans, fontSize: 120, fontWeight: 700, letterSpacing: -3, color: palette.ink3, lineHeight: 1.05}}>+{seen.diff}</span>
            <span style={{fontFamily: fonts.mono, fontSize: 44, color: palette.ink3}}>({seen.lo} to {seen.hi})</span>
          </div>
        </div>
        <div style={{flex: 1, borderLeft: `6px solid ${palette.ink}`, paddingLeft: 32}}>
          <div style={{fontFamily: fonts.sans, fontSize: 44, fontWeight: 600, color: palette.ink}}>Never seen, {unseen.n} questions</div>
          <div style={{display: 'flex', alignItems: 'baseline', gap: 28, marginTop: 10, opacity: interpolate(frame, [twenty, twenty + 8], [0, 1], clamp)}}>
            <span data-unseen style={{fontFamily: fonts.sans, fontSize: 120, fontWeight: 700, letterSpacing: -3, color: palette.ink, lineHeight: 1.05}}>+{unseen.diff}</span>
            <span style={{fontFamily: fonts.mono, fontSize: 44, color: palette.ink}}>({unseen.lo} to {unseen.hi})</span>
          </div>
        </div>
      </div>
    </Panel>
  </AbsoluteFill>;
}
