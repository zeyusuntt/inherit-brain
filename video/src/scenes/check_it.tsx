import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette, radius} from '../theme';
import {Appear, clamp} from '../components/Motion';
import {Camera, element, needle, pad, rect, union} from '../components/Camera';
import {cuer, type SceneProps} from './types';

const NAME = 'explore_check' as const;
const CAM = {left: layout.margin, top: 190, width: layout.width, height: 1020};

// The explorer in motion on another held-out question: the list and the question, the path's
// Cypher, then the gold answer beside every arm's answer and score; then the link.
export function CheckIt(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const query = cue('every query');
  const gold = cue('gold answer');
  const link = cue('The link is below');
  const step1 = union(element(NAME, 'step-head', 'resolve_entity'), element(NAME, 'q-details', 'MATCH (p:Product)'));
  const answers = union(element(NAME, 'gold'), element(NAME, 'arm-list'), needle(NAME, 'Gold answer'));
  const dim = interpolate(frame, [link, link + 12], [0, 1], clamp);
  return <AbsoluteFill style={{background: palette.page}}>
    <div style={{position: 'absolute', left: layout.margin, top: layout.top, display: 'flex', gap: 28, alignItems: 'baseline'}}>
      <span style={{fontFamily: fonts.sans, fontSize: 44, fontWeight: 650, color: palette.ink}}>Every held-out answer, in the explorer</span>
      <span style={{fontFamily: fonts.mono, fontSize: 36, color: palette.ink3}}>{facts.n_questions.toLocaleString('en-US')} questions</span>
    </div>
    <div style={{position: 'absolute', ...CAM}}>
      <Camera name={NAME} width={CAM.width} height={CAM.height}
        keys={[{at: 0, rect: rect(40, 70, 1360, 830)}, {at: query, rect: pad(step1, 16, 12)}, {at: gold, rect: pad(answers, 16, 12)}]}
        marks={[
          {rect: needle(NAME, 'GRAPH.RO_QUERY', 0), at: query + 6, until: gold, kind: 'underline'},
          {rect: needle(NAME, 'GRAPH.RO_QUERY', 1), at: query + 10, until: gold, kind: 'underline'},
          {rect: element(NAME, 'gold'), at: gold + 6, color: palette.ink, label: "HERB's gold answer", labelSide: 'top'},
          {rect: element(NAME, 'ans-head'), at: gold + 14, label: 'every arm, scored', labelSide: 'right'},
        ]}/>
      {dim > 0 && <div style={{position: 'absolute', inset: 0, background: palette.page, opacity: dim, borderRadius: radius}}/>}
      {frame >= link && <Appear at={link} style={{position: 'absolute', left: 0, right: 0, top: 330, textAlign: 'center'}}>
        <div style={{fontFamily: fonts.sans, fontSize: 52, fontWeight: 500, color: palette.ink3}}>The demo site</div>
        <div data-url style={{fontFamily: fonts.mono, fontSize: 104, fontWeight: 600, color: palette.ink, marginTop: 24, letterSpacing: -2}}>{facts.site_label}</div>
        <div style={{fontFamily: fonts.sans, fontSize: 44, color: palette.ink2, marginTop: 30}}>Overview, explorer, and the if-someone-leaves view; the link is in the description</div>
      </Appear>}
    </div>
  </AbsoluteFill>;
}
