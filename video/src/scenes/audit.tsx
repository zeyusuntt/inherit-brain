import {AbsoluteFill, interpolate, spring, useCurrentFrame} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette, radius} from '../theme';
import {Appear, clamp} from '../components/Motion';
import {Heading} from '../components/ui';
import {cuer, FPS, type SceneProps} from './types';

const CELL_W = 364, GAP = 24, TOP = 370, CELL_H = 700;
// The passing round shows the same kinds of gate the earlier rounds failed (PREREG section 12, round-6 table).
const FINAL_SHOWN = ['bug events', 'review relations', 'bug recall', 'reviewer recall'];

function Gate({label, value, note, color}: {label: string; value: string | null; note?: string | null; color: string}) {
  return <div style={{marginTop: 12}}>
    <div style={{fontFamily: fonts.sans, fontSize: 34, color: palette.ink2, lineHeight: 1.15}}>{label}</div>
    {value && <div style={{fontFamily: fonts.mono, fontSize: 40, fontWeight: 600, color, fontVariantNumeric: 'tabular-nums'}}>{value}</div>}
    {note && <div style={{fontFamily: fonts.sans, fontSize: 34, color: palette.ink3, lineHeight: 1.15}}>{note}</div>}
  </div>;
}

// Six audit rounds as six cells: rounds 1 to 5 failed, round 6 passed, each with its gates as registered.
export function Audit(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const failedAt = cue('failed it five times');
  const passAt = cue('drew a fresh sample');
  const keptAt = cue('kept the failures on the record');
  const finals = facts.audit_final.filter((g) => FINAL_SHOWN.includes(g.label));
  return <AbsoluteFill style={{background: palette.page}}>
    <Heading kicker="Extraction audit on the six development products, PREREG section 12" title="Six rounds, a fresh sample each time" size={64}/>
    <Appear at={cue('Two AI auditors')} style={{position: 'absolute', left: layout.margin, top: 285}}>
      <div style={{fontFamily: fonts.sans, fontSize: 42, color: palette.ink2}}>Two model auditors, Codex and a Claude subagent, checked sampled facts against the raw records.</div>
    </Appear>
    {facts.audit.map((r, i) => {
      const at = r.result === 'failed' ? failedAt + i * 7 : passAt;
      const on = frame >= at;
      const pop = on ? spring({frame: frame - at, fps: FPS, config: {damping: 16, stiffness: 170}, durationInFrames: 14}) : 0;
      const tone = r.result === 'failed' ? palette.bad : palette.ok;
      const wash = r.result === 'failed' ? palette.badWash : palette.okWash;
      const gates = r.result === 'failed' ? r.failed : finals.map((g) => ({...g, note: null}));
      return <div key={r.round} data-round={r.round} data-result={on ? r.result : 'pending'} style={{position: 'absolute', left: layout.margin + i * (CELL_W + GAP), top: TOP, width: CELL_W, height: CELL_H, boxSizing: 'border-box',
        padding: '26px 26px', background: on ? wash : palette.surface, border: `3px solid ${on ? tone : palette.line}`, borderRadius: radius, opacity: interpolate(frame, [i * 3, i * 3 + 10], [0, 1], clamp)}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: 64}}>
          <span style={{fontFamily: fonts.sans, fontSize: 46, fontWeight: 650, color: palette.ink, whiteSpace: 'nowrap'}}>Round {r.round}</span>
          {on && <span style={{fontFamily: fonts.sans, fontSize: 34, fontWeight: 650, color: palette.white, background: tone, borderRadius: radius, padding: '2px 12px', transform: `scale(${0.6 + 0.4 * pop})`, transformOrigin: 'right center'}}>{r.result}</span>}
        </div>
        <div style={{fontFamily: fonts.mono, fontSize: 34, color: palette.ink3, marginBottom: 6}}>rubric {r.rubric}</div>
        {gates.map((g, k) => <Gate key={k} label={g.label} value={g.value} note={g.note} color={on ? tone : palette.ink}/>)}
      </div>;
    })}
    <Appear at={keptAt} style={{position: 'absolute', left: layout.margin, top: TOP + CELL_H + 26}}>
      <div data-kept style={{fontFamily: fonts.sans, fontSize: 64, fontWeight: 650, color: palette.ink, letterSpacing: -1}}>Failures kept on the record.</div>
    </Appear>
  </AbsoluteFill>;
}
