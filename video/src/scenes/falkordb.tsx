import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette, radius} from '../theme';
import {Appear, clamp} from '../components/Motion';
import {Camera, element, needle, pad, rect, union} from '../components/Camera';
import {cuer, type SceneProps} from './types';

const STAGE_TOP = 350;
const ITEM_W = layout.width / 4;

function Item({n, title, detail, at, active}: {n: number; title: string; detail: string; at: number; active: boolean}) {
  const frame = useCurrentFrame();
  if (frame < at) return null;
  return <Appear at={at} style={{position: 'absolute', left: layout.margin + (n - 1) * ITEM_W, top: 180, width: ITEM_W - 30}}>
    <div style={{display: 'flex', alignItems: 'baseline', gap: 16}}>
      <span style={{fontFamily: fonts.mono, fontSize: 40, color: palette.ink3}}>{n}</span>
      <span style={{fontFamily: fonts.sans, fontSize: 46, fontWeight: 650, color: active ? palette.ink : palette.ink2}}>{title}</span>
    </div>
    <div style={{fontFamily: fonts.mono, fontSize: 36, color: palette.ink3, marginTop: 6, marginLeft: 40}}>{detail}</div>
    <div style={{height: 6, marginTop: 12, marginLeft: 40, borderRadius: 3, background: active ? palette.ink : 'transparent'}}/>
  </Appear>;
}

// Cypher with the relationship patterns lit as the narration names them.
function CypherBlock({lines, marks}: {lines: string[]; marks: {token: string; at: number}[]}) {
  const frame = useCurrentFrame();
  const pattern = new RegExp(`(${marks.map((m) => m.token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`);
  return <div style={{fontFamily: fonts.mono, fontSize: 38, lineHeight: 1.7, color: palette.ink, background: palette.codeBg, borderRadius: radius, padding: '30px 40px', whiteSpace: 'pre'}}>
    {lines.map((line, i) => <div key={i}>{line.split(pattern).map((part, k) => {
      const m = marks.find((x) => x.token === part);
      if (!m) return <span key={k}>{part}</span>;
      const on = frame >= m.at;
      const t = interpolate(frame, [m.at, m.at + 8], [0, 1], clamp);
      return <span key={k} data-lit={on ? 'true' : 'false'} style={{color: on ? palette.accentInk : palette.ink, fontWeight: on ? 650 : 400, background: `rgba(45,79,208,${0.14 * t})`, borderRadius: 4, boxShadow: on ? `0 0 0 3px rgba(45,79,208,${0.14 * t})` : 'none'}}>{part}</span>;
    })}</div>)}
  </div>;
}

export function Falkordb(p: SceneProps) {
  const frame = useCurrentFrame();
  const cue = cuer(p);
  const at = [cue('a separate graph per product'), cue('a read-only user'), cue('Cypher that joins'), cue('GRAPH.COPY')];
  const active = at.filter((x) => frame >= x).length;
  const bb = facts.bug_board, wi = facts.whatif;
  const julia = cue('If Julia Smith leaves'), bugs = cue('two open bugs'), pr = cue('a pull request');
  const graphs = [...facts.products.development, ...facts.products.held_out].sort();
  // An item's row on the what-if page: its title's line, across the people table's width.
  const table = element('whatif', 'wi-head');
  const itemRows = wi.items.map((it) => { const n = needle('whatif', it.title); return rect(table.x, n.y - 4, table.w, 40); });
  const juliaRow = union(element('whatif', 'person-sum', 'Julia Smith'), ...itemRows);
  return <AbsoluteFill style={{background: palette.page}}>
    <div style={{position: 'absolute', left: layout.margin, top: layout.top, fontFamily: fonts.sans, fontSize: 40, fontWeight: 500, color: palette.ink3}}>FalkorDB doing real work</div>
    <Item n={1} title="A graph per product" detail={`herb:<product> x ${graphs.length}`} at={at[0]} active={active === 1}/>
    <Item n={2} title="A read-only user" detail="GRAPH.RO_QUERY" at={at[1]} active={active === 2}/>
    <Item n={3} title="One Cypher query" detail={bb.tool} at={at[2]} active={active === 3}/>
    <Item n={4} title="GRAPH.COPY" detail="if someone leaves" at={at[3]} active={active === 4}/>

    {active <= 1 && <div data-stage="graphs" style={{position: 'absolute', left: layout.margin, right: layout.margin, top: STAGE_TOP, display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', rowGap: 14, columnGap: 40}}>
      {graphs.map((g, i) => <div key={g} style={{fontFamily: fonts.mono, fontSize: 38, color: palette.ink2, opacity: interpolate(frame, [i * 1.5, i * 1.5 + 8], [0, 1], clamp)}}>herb:{g}</div>)}
    </div>}

    {active === 2 && <Appear at={at[1]} style={{position: 'absolute', left: layout.margin, right: layout.margin, top: STAGE_TOP + 40}}>
      <div style={{fontFamily: fonts.mono, fontSize: 56, lineHeight: 1.7, color: palette.ink, background: palette.codeBg, borderRadius: radius, padding: '40px 56px'}}>
        <div><span style={{color: palette.ink3}}>user  </span>agent_&lt;product&gt;</div>
        <div><span style={{color: palette.ink3}}>rule  </span>(~herb:&lt;product&gt; <span style={{color: palette.accentInk, fontWeight: 650}}>+GRAPH.RO_QUERY</span>)</div>
      </div>
      <div style={{fontFamily: fonts.sans, fontSize: 52, fontWeight: 500, color: palette.ink2, marginTop: 50}}>The agent connects as this user: it can read one product's graph and nothing else.</div>
    </Appear>}

    {active === 3 && <Appear at={at[2]} style={{position: 'absolute', left: layout.margin, right: layout.margin, top: STAGE_TOP}}>
      <div style={{fontFamily: fonts.sans, fontSize: 40, color: palette.ink3, marginBottom: 20}}>
        <span style={{fontFamily: fonts.mono, color: palette.ink2}}>{bb.tool}</span>, as run for <span style={{fontFamily: fonts.mono}}>{bb.key}</span>: {bb.command}, {bb.rows} rows, {bb.ms.toFixed(1)} ms
      </div>
      <CypherBlock lines={bb.lines} marks={[
        {token: '(b:Bug)', at: cue('joins a bug')},
        {token: '[fx:FIXES]', at: cue('its fix')},
        {token: '(p:Person)-[op:OPENED]', at: cue('its author')},
        {token: '[hs:HAS_EVENT]->(rs:BugEvent)', at: cue('its history')},
        {token: '[:HAS_EVENT]->(ro:BugEvent)', at: cue('its history') + 6},
      ]}/>
      <div style={{fontFamily: fonts.sans, fontSize: 46, fontWeight: 500, color: palette.ink2, marginTop: 36}}>Reopened bugs, the PRs that fixed them, who opened each PR, and the RESOLVED and REOPENED events, in one read-only query.</div>
    </Appear>}

    {active === 4 && <Appear at={at[3]} style={{position: 'absolute', left: layout.margin, right: layout.margin, top: STAGE_TOP - 10}}>
      <div style={{display: 'flex', alignItems: 'baseline', gap: 30}}>
        <span style={{fontFamily: fonts.mono, fontSize: 40, color: palette.ink, background: palette.codeBg, borderRadius: radius, padding: '6px 18px'}}>GRAPH.COPY herb:&lt;p&gt; whatif:&lt;p&gt;:&lt;uuid4 hex&gt;</span>
        <span style={{fontFamily: fonts.sans, fontSize: 38, color: palette.ink3}}>on a copy, as its own ACL user; not part of the scored run</span>
      </div>
      <div style={{marginTop: 26, display: 'flex', gap: 56}}>
        <Camera name="whatif" width={1640} height={780}
          keys={[{at: 0, rect: rect(110, 86, 1220, 570)}, {at: julia, rect: pad(juliaRow, 24, 20)}]}
          marks={[
            {rect: needle('whatif', 'Julia Smith'), at: julia + 6, kind: 'underline', color: palette.ink},
            {rect: itemRows[0], at: bugs, color: palette.bad},
            {rect: itemRows[1], at: bugs + 6, color: palette.bad},
            {rect: itemRows[2], at: pr, color: palette.bad},
          ]}/>
        <div style={{flex: 1, paddingTop: 30}}>
          <div style={{fontFamily: fonts.sans, fontSize: 40, color: palette.ink3}}>{wi.product}, what-if page</div>
          {frame >= julia && <Appear at={julia}><div style={{fontFamily: fonts.sans, fontSize: 60, fontWeight: 650, color: palette.ink, marginTop: 18, lineHeight: 1.1}}>If {wi.person.name} leaves</div></Appear>}
          {frame >= bugs && <Appear at={bugs}><div data-unowned-bugs style={{fontFamily: fonts.sans, fontSize: 92, fontWeight: 700, color: palette.bad, marginTop: 40, lineHeight: 1}}>{wi.bugs} open bugs</div></Appear>}
          {frame >= pr && <Appear at={pr}><div data-unowned-prs style={{fontFamily: fonts.sans, fontSize: 92, fontWeight: 700, color: palette.bad, marginTop: 20, lineHeight: 1}}>{wi.prs} pull request</div></Appear>}
          {frame >= pr && <Appear at={pr + 6}><div style={{fontFamily: fonts.sans, fontSize: 48, color: palette.ink2, marginTop: 26}}>have no owner left</div></Appear>}
        </div>
      </div>
    </Appear>}
  </AbsoluteFill>;
}
