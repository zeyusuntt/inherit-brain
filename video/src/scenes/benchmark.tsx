import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette, radius} from '../theme';
import {Appear, clamp} from '../components/Motion';
import {Camera, element, needle, pad, rect} from '../components/Camera';
import {Heading, thousands} from '../components/ui';
import {cuer, type SceneProps} from './types';

const TILE_W = 368, TILE_H = 106, GAP = 16, COLS = 6, GRID_TOP = 334, ROW_GAP = 24;

function Stat({value, label, at}: {value: string; label: string; at: number}) {
  return <Appear at={at} style={{flex: 1}}>
    <div style={{fontFamily: fonts.sans, fontSize: 124, fontWeight: 700, letterSpacing: -4, lineHeight: 1, color: palette.ink, fontVariantNumeric: 'tabular-nums'}}>{value}</div>
    <div style={{fontFamily: fonts.sans, fontSize: 46, fontWeight: 500, color: palette.ink2, marginTop: 14}}>{label}</div>
  </Appear>;
}

// HERB's 30 products split 6 development and 24 held out; then the pre-registration, hashed and posted.
export function Benchmark(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const six = cue('I built on six');
  const held = cue('held back');
  const questions = cue('twelve hundred');
  const rest = cue('twenty-four');
  const before = cue('Before the held-out run');
  const posted = cue('posted the hash');
  const pr = facts.prereg;
  const all = [...facts.products.development.map((n) => ({n, dev: true})), ...facts.products.held_out.map((n) => ({n, dev: false}))];
  const out = interpolate(frame, [before - 10, before], [1, 0], clamp);
  const box = element('gist', 'file');
  // The bullet list: the sha256, the commit and the posting date, with the paragraph above it.
  const list = rect(needle('gist', 'sha256', 1).x - 110, needle('gist', 'sha256', 1).y - 90, 720, 200);
  return <AbsoluteFill style={{background: palette.page}}>
    {out > 0 && <div style={{opacity: out}}>
      <Heading kicker="HERB, Salesforce AI Research's benchmark" title={`${all.length} simulated products`} size={64}/>
      {all.map(({n, dev}, i) => {
        const col = i % COLS, row = Math.floor(i / COLS);
        const devOn = dev && frame >= six;
        const heldOn = !dev && frame >= held;
        return <div key={n} data-product={n} data-split={dev ? 'development' : 'held-out'} style={{position: 'absolute', left: layout.margin + col * (TILE_W + GAP), top: GRID_TOP + row * (TILE_H + ROW_GAP), width: TILE_W, height: TILE_H, boxSizing: 'border-box',
          display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: radius, fontFamily: fonts.sans, fontSize: 35, fontWeight: heldOn ? 600 : 500, whiteSpace: 'nowrap',
          background: devOn ? palette.surface2 : palette.surface, color: devOn ? palette.ink3 : heldOn ? palette.ink : palette.ink2,
          border: `${heldOn ? 3 : 2}px ${devOn ? 'dashed' : 'solid'} ${heldOn ? palette.ink : palette.line}`,
          opacity: interpolate(frame, [i, i + 8], [0, 1], clamp)}}>{n}</div>;
      })}
      {frame >= six && <div style={{position: 'absolute', left: layout.margin + 12, top: GRID_TOP - 50, fontFamily: fonts.sans, fontSize: 36, fontWeight: 600, color: palette.ink3, opacity: interpolate(frame, [six, six + 10], [0, 1], clamp)}}>Development: the tools were built on these six</div>}
      <div style={{position: 'absolute', left: layout.margin + 12, right: layout.margin, top: 1000, display: 'flex', gap: 80}}>
        <Stat at={six} value={String(facts.products.development.length)} label="products to build on"/>
        <Stat at={rest} value={String(facts.n_products_held_out)} label="products held out"/>
        <Stat at={questions} value={thousands(facts.n_questions)} label="held-out questions"/>
      </div>
    </div>}
    {frame >= before - 10 && <div style={{opacity: 1 - out}}>
      <Heading kicker="Pre-registered before the held-out run" title="The plan, hashed and posted in public" size={64}/>
      <div style={{position: 'absolute', left: layout.margin, top: 330}}>
        <Camera name="gist" fadeEdges keys={[{at: 0, rect: pad(rect(box.x, box.y + 60, box.w, 420), 10, 10)}, {at: posted, rect: list}]}
          marks={[{rect: needle('gist', pr.sha256), at: posted + 8, kind: 'underline'}]} width={1460} height={830}/>
      </div>
      <div style={{position: 'absolute', left: 1700, right: layout.margin, top: 360}}>
        <Appear at={before + 4}>
          <div style={{fontFamily: fonts.sans, fontSize: 40, color: palette.ink3}}>PREREG.md sha256</div>
          <div data-sha style={{fontFamily: fonts.mono, fontSize: 84, fontWeight: 600, color: palette.ink, marginTop: 8}}>{pr.short}...</div>
        </Appear>
        <Appear at={cue('I froze the code')} style={{marginTop: 54}}>
          <div style={{fontFamily: fonts.sans, fontSize: 40, color: palette.ink3}}>Code frozen at commit</div>
          <div style={{fontFamily: fonts.mono, fontSize: 56, color: palette.ink, marginTop: 8}}>{pr.commit}</div>
        </Appear>
        <Appear at={posted} style={{marginTop: 54}}>
          <div style={{fontFamily: fonts.sans, fontSize: 40, color: palette.ink3}}>Posted</div>
          <div style={{fontFamily: fonts.mono, fontSize: 52, color: palette.ink, marginTop: 8}}>{pr.posted_utc}</div>
          <div style={{fontFamily: fonts.sans, fontSize: 40, color: palette.ink2, marginTop: 20, lineHeight: 1.3}}>Public gist, with a Wayback Machine snapshot</div>
        </Appear>
      </div>
    </div>}
  </AbsoluteFill>;
}
