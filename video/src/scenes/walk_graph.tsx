import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette} from '../theme';
import {Appear, clamp} from '../components/Motion';
import {Camera, element, needle, pad, rect, union} from '../components/Camera';
import {Note} from '../components/ui';
import {cuer, type SceneProps} from './types';

const NAME = 'explore_q3' as const;
const CAM = {left: layout.margin, top: 180, width: layout.width, height: 950};
// The path picture is tall and narrow: a portrait camera on it, the reading of it beside.
const PATH_CAM = {width: 820, height: 950};

// The explorer on the live site, SupportForce/ans/3: the question, the path picture (three Document
// versions, ten Person nodes, AUTHORED and SUGGESTED edges read from the graph), the ten ids checked
// against gold, the two Alice Davis, and the evidence behind each row.
export function WalkGraph(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const wg = facts.walk_graph;
  const walks = cue('It walks the graph');
  const ten = cue('Ten people');
  const alice = cue('two different people');
  const record = cue('each with the record');

  const question = union(rect(element(NAME, 'd-nav').x, element(NAME, 'd-nav').y + 4, element(NAME, 'd-nav').w, 10), element(NAME, 'gold'));
  const docs = [0, 1, 2].map((i) => element(NAME, 'gv-node', 'Document:', i));
  const people = Array.from({length: wg.answer.length}, (_, i) => element(NAME, 'gv-node', 'Person:', i));
  const picture = pad(union(...docs, ...people), 16, 18);
  const answer = pad(element(NAME, 'ans-a5'), 12, 10);
  const chipsA5 = element(NAME, 'chips', '✓ Alice Martinez eid_294ba09d');
  // The second match of each is the A5 answer's chip (the first is the gold answer's).
  const [aliceA, aliceB] = wg.alice_davis.map((eid) => element(NAME, 'chip', `Alice Davis ${eid}`, 1));
  // The doc_people call and its first six rows, with the evidence column boxed.
  const table = element(NAME, 'rows');
  const head = element(NAME, 'step-head', 'doc_people');
  const rows = pad(rect(head.x, head.y, table.w, table.y + 300 - head.y), 14, 12);
  const evidence = needle(NAME, 'evidence');
  const evidenceCol = rect(evidence.x - 8, table.y + 2, table.x + table.w - evidence.x + 4, 296);

  const keys = [
    {at: 0, rect: pad(question, 20, 16)},
    {at: ten, rect: answer},
    {at: alice, rect: rect(chipsA5.x - 35, chipsA5.y - 30, 800, 180)},
    {at: record, rect: rows},
  ];
  const marks = [
    {rect: chipsA5, at: ten + 6, until: alice, color: palette.ok, label: `${wg.n_correct} of ${wg.n_gold} gold ids`},
    {rect: aliceA, at: alice + 4, until: record},
    {rect: aliceB, at: alice + 10, until: record},
    {rect: evidenceCol, at: record + 6},
  ];
  const showPath = frame >= walks - 6 && frame < ten + 10;
  const pathOpacity = interpolate(frame, [walks - 6, walks + 4, ten, ten + 10], [0, 1, 1, 0], clamp);
  return <AbsoluteFill style={{background: palette.page}}>
    <div style={{position: 'absolute', left: layout.margin, top: layout.top, display: 'flex', gap: 28, alignItems: 'baseline'}}>
      <span style={{fontFamily: fonts.sans, fontSize: 44, fontWeight: 650, color: palette.ink}}>The explorer on the demo site</span>
      <span style={{fontFamily: fonts.mono, fontSize: 36, color: palette.ink3}}>explore.html#q={wg.key}</span>
    </div>
    <div style={{position: 'absolute', ...CAM}}>
      <Camera name={NAME} keys={keys} marks={marks} width={CAM.width} height={CAM.height}/>
    </div>
    {showPath && <div data-path-picture style={{position: 'absolute', ...CAM, background: palette.page, opacity: pathOpacity, display: 'flex', gap: 70}}>
      <Camera name={NAME} width={PATH_CAM.width} height={PATH_CAM.height} keys={[{at: 0, rect: picture}]}
        marks={docs.map((d, i) => ({rect: d, at: walks + 10 + i * 4, color: palette.ink}))}/>
      <div style={{flex: 1, paddingTop: 20}}>
        <div style={{fontFamily: fonts.sans, fontSize: 44, fontWeight: 650, color: palette.ink}}>The path in the graph</div>
        <div style={{fontFamily: fonts.sans, fontSize: 38, color: palette.ink3, marginTop: 8}}>drawn on the site from the real edges between the answer's nodes</div>
        <Appear at={walks + 10}><div style={{fontFamily: fonts.sans, fontSize: 64, fontWeight: 650, color: palette.ink, marginTop: 70}}>{wg.documents.length} versions of the spec</div></Appear>
        <Appear at={walks + 22}><div style={{fontFamily: fonts.sans, fontSize: 52, color: palette.ink2, marginTop: 26}}>linked by <span style={{fontFamily: fonts.mono, fontWeight: 600, color: palette.ink}}>AUTHORED</span> and <span style={{fontFamily: fonts.mono, fontWeight: 600, color: palette.ink}}>SUGGESTED</span> edges to</div></Appear>
        <Appear at={walks + 34}><div style={{fontFamily: fonts.sans, fontSize: 64, fontWeight: 650, color: palette.accent, marginTop: 26}}>{wg.answer.length} people{wg.n_correct === wg.answer.length ? ', each in the gold answer' : `, ${wg.n_correct} in the gold answer`}</div></Appear>
        <Appear at={walks + 46}><div style={{marginTop: 70}}><Note>{wg.tool}: one tool call, {wg.n_queries} read-only Cypher queries, {wg.rows} rows</Note></div></Appear>
      </div>
    </div>}
    <div style={{position: 'absolute', left: layout.margin, top: CAM.top + CAM.height + 22}}>
      {frame >= ten && frame < alice && <Appear at={ten + 4}><Note>All {wg.n_gold} ids returned, each checked against HERB's gold answer</Note></Appear>}
      {frame >= alice && frame < record && <Appear at={alice + 6}><Note color={palette.accent}>Two different people named Alice Davis: {wg.alice_davis.join(' and ')}</Note></Appear>}
      {frame >= record && <Appear at={record + 4}><Note>Each row carries the records that put the person there</Note></Appear>}
    </div>
  </AbsoluteFill>;
}
