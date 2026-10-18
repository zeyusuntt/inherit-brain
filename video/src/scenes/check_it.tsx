import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette, radius} from '../theme';
import {Appear, clamp} from '../components/Motion';
import {Camera, element, needle, pad, rect, union} from '../components/Camera';
import {cuer, type SceneProps} from './types';

const NAME = 'explore_check' as const;
const CAM = {left: layout.margin, top: 190, width: layout.width, height: 1020};

// The explorer in motion on another held-out question (SecurityForce/ans/21): the list and the
// question, its path picture (BugEvent, Bug, PR, Person by HAS_EVENT, FIXES, OPENED), then the gold
// answer beside every arm's answer and score; then the link.
export function CheckIt(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const path = cue('Every answer');
  const gold = cue('gold answer');
  const link = cue('The link is below');
  const figure = union(needle(NAME, 'The path in the graph'), element(NAME, 'gv-svg'));
  const armList = element(NAME, 'arm-list');
  const answers = union(element(NAME, 'gold'), needle(NAME, 'Gold answer'), rect(armList.x, armList.y, armList.w, 130));
  const person = element(NAME, 'gv-node', 'Person:');
  const dim = interpolate(frame, [link, link + 12], [0, 1], clamp);
  return <AbsoluteFill style={{background: palette.page}}>
    <div style={{position: 'absolute', left: layout.margin, top: layout.top, display: 'flex', gap: 28, alignItems: 'baseline'}}>
      <span style={{fontFamily: fonts.sans, fontSize: 44, fontWeight: 650, color: palette.ink}}>Every held-out answer, in the explorer</span>
      <span style={{fontFamily: fonts.mono, fontSize: 36, color: palette.ink3}}>{facts.n_questions.toLocaleString('en-US')} questions</span>
    </div>
    <div style={{position: 'absolute', ...CAM}}>
      <Camera name={NAME} width={CAM.width} height={CAM.height}
        keys={[{at: 0, rect: rect(40, 70, 1360, 830)}, {at: path, rect: pad(figure, 16, 14)}, {at: gold, rect: pad(answers, 16, 12)}]}
        marks={[
          {rect: person, at: path + 10, until: gold, label: 'the answer', labelSide: 'top'},
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
