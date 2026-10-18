import {AbsoluteFill, interpolate, spring, useCurrentFrame} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette, radius} from '../theme';
import {Appear, clamp} from '../components/Motion';
import {cuer, FPS, type SceneProps} from './types';

type Record = typeof facts.cold_open.columns[number]['records'][number];
const COL_W = 552, GAP_X = 32, CARD_H = 174, GAP_Y = 16, FEED_TOP = 268, FEED_H = 1220 - 268;
const PREFILLED = 5, INTERVAL = 24;
const fade = 'linear-gradient(to bottom, #000 82%, transparent 100%)';

function Card({r, titled}: {r: Record; titled: boolean}) {
  return <div style={{height: CARD_H, boxSizing: 'border-box', padding: '16px 22px', background: palette.surface, border: `2px solid ${palette.line}`, borderRadius: radius, overflow: 'hidden'}}>
    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 16}}>
      <span style={{fontFamily: fonts.sans, fontSize: 36, fontWeight: 600, color: palette.ink, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>{r.author}</span>
      <span style={{fontFamily: fonts.mono, fontSize: 34, color: palette.ink3, whiteSpace: 'nowrap'}}>{r.ts.slice(0, 10)}</span>
    </div>
    {titled
      ? <>
        <div style={{fontFamily: fonts.sans, fontSize: 36, fontWeight: 550, color: palette.ink, marginTop: 8, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>{r.where}</div>
        <div style={{fontFamily: fonts.sans, fontSize: 34, color: palette.ink2, marginTop: 4, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>{r.text}</div>
      </>
      : <div style={{fontFamily: fonts.sans, fontSize: 34, lineHeight: 1.3, color: palette.ink2, marginTop: 8, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden'}}>{r.text}</div>}
  </div>;
}

// One kind of record, stacking up: each new record lands on top and pushes the older ones down.
function Feed({records, start, titled}: {records: Record[]; start: number; titled: boolean}) {
  const frame = useCurrentFrame();
  const arrive = (k: number) => (k < PREFILLED ? -100 : start + (k - PREFILLED) * INTERVAL);
  // Older records slide down first; the new one fades in on top once there is room.
  const settle = (k: number) => spring({frame: frame - arrive(k), fps: FPS, config: {damping: 22, stiffness: 160}, durationInFrames: 12});
  return <div style={{position: 'absolute', top: FEED_TOP, height: FEED_H, width: COL_W, overflow: 'hidden', maskImage: fade, WebkitMaskImage: fade}}>
    {records.map((r, k) => {
      if (frame < arrive(k)) return null;
      let y = 0;
      for (let j = k + 1; j < records.length; j++) if (frame >= arrive(j)) y += (CARD_H + GAP_Y) * settle(j);
      const enter = interpolate(frame - arrive(k), [6, 16], [0, 1], clamp);
      return <div key={r.id} data-record={r.id} style={{position: 'absolute', left: 0, right: 0, top: y - (1 - enter) * 24, opacity: enter}}><Card r={r} titled={titled}/></div>;
    })}
  </div>;
}

export function ColdOpen(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const named = [cue('Slack'), cue('meetings'), cue('documents'), cue('pull requests')];
  const firstQuestion = cue('Your first question');
  const ask = cue('Who wrote');
  const dim = interpolate(frame, [firstQuestion, firstQuestion + 14], [0, 1], clamp);
  const co = facts.cold_open;
  return <AbsoluteFill style={{background: palette.page}}>
    <div style={{position: 'absolute', left: layout.margin, top: layout.top, fontFamily: fonts.sans, fontSize: 40, fontWeight: 500, color: palette.ink3}}>
      Real records from one HERB product, {co.product} <span style={{fontFamily: fonts.mono, fontSize: 36}}>{co.first} to {co.last}</span>
    </div>
    {co.columns.map((c, i) => {
      const lit = frame >= named[i];
      return <div key={c.kind} style={{position: 'absolute', left: layout.margin + i * (COL_W + GAP_X), top: 0, width: COL_W, height: 1440}}>
        <div style={{position: 'absolute', top: 186, fontFamily: fonts.sans, fontSize: 50, fontWeight: 650, color: lit ? palette.ink : palette.ink3, letterSpacing: -0.5}}>
          {c.label}
          <div style={{height: 6, marginTop: 6, borderRadius: 3, background: palette.ink, width: `${interpolate(frame, [named[i], named[i] + 10], [0, 100], clamp)}%`}}/>
        </div>
        <Feed records={c.records} start={6 + i * 7} titled={c.kind === 'Document' || c.kind === 'PR'}/>
      </div>;
    })}
    {dim > 0 && <AbsoluteFill style={{background: `rgba(246,247,248,${0.9 * dim})`}}/>}
    <div style={{position: 'absolute', left: 220, right: 220, top: 470, textAlign: 'center'}}>
      <Appear at={firstQuestion}><div style={{fontFamily: fonts.sans, fontSize: 52, fontWeight: 500, color: palette.ink3}}>Your first question</div></Appear>
      <Appear at={ask}><div data-question style={{fontFamily: fonts.sans, fontSize: 124, fontWeight: 650, letterSpacing: -3, lineHeight: 1.08, color: palette.ink, marginTop: 34}}>Who wrote the technical spec, and who reviewed it?</div></Appear>
    </div>
  </AbsoluteFill>;
}
