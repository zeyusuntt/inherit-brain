import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette} from '../theme';
import {Appear, clamp} from '../components/Motion';
import {Heading, thousands} from '../components/ui';
import {cuer, type SceneProps} from './types';

const TOP = 450, ROW = 84;

// The eight arms of the run, one model for all; inherit-brain in the accent, the rest grey, and the
// four the narration names turn dark as they are named.
export function Arms(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const named: Record<string, number> = {
    A0: cue("HERB's own retrieval tools"), A1f: cue('reading the whole product'), A0v: cue('vector search'), A2: cue('GraphRAG SDK'),
  };
  return <AbsoluteFill style={{background: palette.page}}>
    <Heading kicker="Same questions, same limits, eight arms" title="Eight ways to answer, one model"/>
    <Appear at={cue('Qwen3')} style={{position: 'absolute', left: layout.margin, top: 300}}>
      <span style={{fontFamily: fonts.mono, fontSize: 44, fontWeight: 500, color: palette.ink}}>{facts.model_id}</span>
      <span style={{fontFamily: fonts.sans, fontSize: 44, color: palette.ink2}}> on Nebius, for every arm</span>
    </Appear>
    <div style={{position: 'absolute', left: layout.margin, right: layout.margin, top: TOP - 56, display: 'flex', fontFamily: fonts.sans, fontSize: 34, color: palette.ink3}}>
      <span style={{width: 170, paddingLeft: 30}}>Arm</span><span style={{flex: 1}}/><span>Questions run</span>
    </div>
    {facts.arms.map((a, i) => {
      const self = a.id === 'A5';
      const at = named[a.id];
      const on = at !== undefined && frame >= at;
      const t = at === undefined ? 0 : interpolate(frame, [at, at + 8], [0, 1], clamp);
      const color = self ? palette.accent : on ? palette.ink : palette.mark;
      const bar = self ? palette.accent : on ? palette.ink : 'transparent';
      return <div key={a.id} data-arm={a.id} data-named={on ? 'true' : 'false'} style={{position: 'absolute', left: layout.margin, right: layout.margin, top: TOP + i * ROW, height: ROW - 10,
        display: 'flex', alignItems: 'center', borderTop: `2px solid ${palette.line}`, opacity: interpolate(frame, [i * 3, i * 3 + 10], [0, 1], clamp)}}>
        <div style={{width: 8, alignSelf: 'stretch', marginTop: 8, background: bar, opacity: self ? 1 : t, borderRadius: 3}}/>
        <span style={{width: 162, paddingLeft: 22, fontFamily: fonts.mono, fontSize: 40, color: self ? palette.accent : on ? palette.ink2 : palette.mark}}>{a.id}</span>
        <span style={{flex: 1, fontFamily: fonts.sans, fontSize: 54, fontWeight: self || on ? 650 : 500, color}}>{a.name}</span>
        <span style={{fontFamily: fonts.mono, fontSize: 40, color: self ? palette.accent : palette.ink3, fontVariantNumeric: 'tabular-nums'}}>{thousands(a.n)}</span>
      </div>;
    })}
    <div style={{position: 'absolute', left: layout.margin + 30, top: TOP + facts.arms.length * ROW + 14, fontFamily: fonts.sans, fontSize: 36, color: palette.ink3}}>
      The three card arms ran on five held-out products and GraphRAG-SDK on six; each is compared on the same questions.
    </div>
  </AbsoluteFill>;
}
