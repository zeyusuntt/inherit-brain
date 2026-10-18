import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette} from '../theme';
import {Appear} from '../components/Motion';
import {Camera, element, needle, pad, rect, union} from '../components/Camera';
import {Note} from '../components/ui';
import {cuer, type SceneProps} from './types';

const NAME = 'explore_q3' as const;
const CAM = {left: layout.margin, top: 180, width: layout.width, height: 950};

// The explorer on the live site, SupportForce/ans/3: the question, the one doc_people call and the
// Cypher it ran, the ten ids checked against gold, the two Alice Davis, and the evidence behind each row.
export function WalkGraph(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const wg = facts.walk_graph;
  const walks = cue('It walks the graph');
  const ten = cue('Ten people');
  const alice = cue('two different people');
  const record = cue('each with the record');

  const question = union(rect(element(NAME, 'd-nav').x, element(NAME, 'd-nav').y + 4, element(NAME, 'd-nav').w, 10), element(NAME, 'gold'));
  const authored = element(NAME, 'qblock', 'AUTHORED');
  const suggested = element(NAME, 'cypher', 'SUGGESTED');
  const cypherTight = pad(rect(authored.x, authored.y, authored.w, suggested.y + suggested.h - authored.y), 16, 12);
  const answer = pad(element(NAME, 'ans-a5'), 12, 10);
  const chipsA5 = element(NAME, 'chips', '✓ Alice Martinez eid_294ba09d');
  // The second match of each is the A5 answer's chip (the first is the gold answer's).
  const [aliceA, aliceB] = wg.alice_davis.map((eid) => element(NAME, 'chip', `Alice Davis ${eid}`, 1));
  const rows = pad(union(element(NAME, 'step-head', 'doc_people'), element(NAME, 'rows'), needle(NAME, 'First 5 of 10 rows.')), 14, 12);
  const evidenceCol = rect(needle(NAME, 'evidence').x - 8, element(NAME, 'rows').y + 2, element(NAME, 'rows').x + element(NAME, 'rows').w - needle(NAME, 'evidence').x + 4, element(NAME, 'rows').h - 4);

  const keys = [
    {at: 0, rect: pad(question, 20, 16)},
    {at: walks, rect: cypherTight},
    {at: ten, rect: answer},
    {at: alice, rect: rect(chipsA5.x - 35, chipsA5.y - 30, 800, 180)},
    {at: record, rect: rows},
  ];
  const marks = [
    {rect: needle(NAME, 'AUTHORED'), at: walks + 8, until: ten, kind: 'underline' as const},
    {rect: needle(NAME, 'SUGGESTED'), at: walks + 14, until: ten, kind: 'underline' as const},
    {rect: chipsA5, at: ten + 6, until: alice, color: palette.ok, label: `${wg.n_correct} of ${wg.n_gold} gold ids`},
    {rect: aliceA, at: alice + 4, until: record},
    {rect: aliceB, at: alice + 10, until: record},
    {rect: evidenceCol, at: record + 6},
  ];
  return <AbsoluteFill style={{background: palette.page}}>
    <div style={{position: 'absolute', left: layout.margin, top: layout.top, display: 'flex', gap: 28, alignItems: 'baseline'}}>
      <span style={{fontFamily: fonts.sans, fontSize: 44, fontWeight: 650, color: palette.ink}}>The explorer on the demo site</span>
      <span style={{fontFamily: fonts.mono, fontSize: 36, color: palette.ink3}}>explore.html#q={wg.key}</span>
    </div>
    <div style={{position: 'absolute', ...CAM}}>
      <Camera name={NAME} keys={keys} marks={marks} width={CAM.width} height={CAM.height}/>
    </div>
    <div style={{position: 'absolute', left: layout.margin, top: CAM.top + CAM.height + 22}}>
      {frame >= walks && frame < ten && <Appear at={walks + 4}><Note>{wg.tool}: one tool call, {wg.n_queries} read-only Cypher queries, {wg.rows} rows</Note></Appear>}
      {frame >= ten && frame < alice && <Appear at={ten + 4}><Note>All {wg.n_gold} ids returned, each checked against HERB's gold answer</Note></Appear>}
      {frame >= alice && frame < record && <Appear at={alice + 6}><Note color={palette.accent}>Two different people named Alice Davis: {wg.alice_davis.join(' and ')}</Note></Appear>}
      {frame >= record && <Appear at={record + 4}><Note>Each row carries the records that put the person there</Note></Appear>}
    </div>
  </AbsoluteFill>;
}
