import {build} from 'esbuild';
import {mkdirSync} from 'node:fs';

// Server-side rendering exercises scene logic: what is on screen at which cue, the type floor, the
// draft mark, caption coverage and the holds. It does not verify browser layout or fonts (the stills do).
mkdirSync('out/checks', {recursive: true});
await build({
  stdin: {resolveDir: process.cwd(), loader: 'tsx', contents: `
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readdirSync, readFileSync} from 'node:fs';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {Internals} from 'remotion';
import {Video, scenes as sceneMap, DRAFT_LABEL} from './src/Video';
import {Captions} from './src/components/Captions';
import {sequenceWindows, totalFrames, cueFrame} from './src/timing';
import {timeline, facts} from './src/content';

const BEATS=['cold_open','intro','walk_graph','benchmark','arms','results','audit','falkordb','check_it','close'];
const windows=sequenceWindows(timeline);
const config={id:'InheritBrain',fps:timeline.fps,width:2560,height:1440,durationInFrames:totalFrames(timeline.total_s,timeline.fps),defaultProps:{}};
globalThis.window={remotion_audioEnabled:false,remotion_videoEnabled:false,remotion_proxyPort:0,location:{origin:'http://localhost',href:'http://localhost/'}};
globalThis.document={baseURI:'http://localhost/'};
function render(element,frame) {
  return renderToStaticMarkup(React.createElement(Internals.RemotionEnvironmentContext.Provider,{value:{isRendering:true,isStudio:false,isPlayer:false,isReadOnlyStudio:false,isClientSideRendering:false}},React.createElement(Internals.CompositionManager.Provider,{value:{compositions:[config],canvasContent:{type:'composition',compositionId:'InheritBrain'},currentCompositionMetadata:config}},React.createElement(Internals.TimelineContext.Provider,{value:{frame:{InheritBrain:frame},isInsideFreeze:false}},React.createElement(Internals.CanUseRemotionHooksProvider,{},element)))));
}
const win=(id)=>windows.find(w=>w.beat.id===id);
function shot(id,frame) {
  const w=win(id);
  return render(React.createElement(sceneMap[id],{beat:w.beat,sequenceFrom:w.from,speechStart:w.speechStart,speechDuration:w.speechDuration,sceneDuration:w.durationInFrames}),frame);
}
const at=(id,phrase)=>cueFrame(win(id).beat,phrase,timeline.fps,win(id).from);
function plain(html) {return html.replace(/<[^>]+>/g,' ').replaceAll('&#x27;', "'").replaceAll('&quot;', '"').replaceAll('&gt;', '>').replaceAll('&lt;', '<').replaceAll('&amp;', '&').replace(/\\s+/g,' ');}
function fontSizes(source) {return [...source.matchAll(/fontSize\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)/g)].map(match=>Number(match[1]));}
function assertFontFloor(path,source) {
  for(const size of fontSizes(source)) assert.ok(size>=34,path+' has fontSize '+size);
}
const ALL_SECOND=(id)=>{const w=win(id); const out=[]; for(let f=0;f<w.durationInFrames;f+=Math.round(timeline.fps/2)) out.push(f); out.push(w.durationInFrames-1); return out;};

test('timeline has the ten beats in order, under 2:59',()=>{
  assert.deepEqual(timeline.beats.map(b=>b.id),BEATS);
  assert.ok(timeline.total_s<=179,'total '+timeline.total_s+'s');
  for(const id of BEATS) assert.ok(sceneMap[id],'scene for '+id);
});
test('source keeps the 34 pixel type floor',()=>{
  const files=readdirSync('src',{recursive:true}).filter(path=>path.endsWith('.tsx'));
  for(const path of files) assertFontFloor(path,readFileSync('src/'+path,'utf8'));
});
test('font floor scanner finds declarations and rejects a sub-floor fixture',()=>{
  const fixture='const safe={fontSize: 34}; const unsafe={fontSize: 20};';
  assert.deepEqual(fontSizes(fixture),[34,20]);
  assert.throws(()=>assertFontFloor('fixture.tsx',fixture),/fixture\\.tsx has fontSize 20/);
});
test('a scratch timeline stamps every frame with the draft mark, top right',()=>{
  for(const frame of [0,Math.round(config.durationInFrames/2),config.durationInFrames-1]) {
    const html=render(React.createElement(Video,{}),frame);
    const marker=html.indexOf('data-draft');
    if(!timeline.source.startsWith('scratch')) { assert.equal(marker,-1,'a recorded narration renders no draft mark'); continue; }
    assert.ok(marker>=0,'draft mark at frame '+frame);
    const tag=html.slice(marker,marker+400);
    assert.ok(tag.includes('right:64px') && tag.includes('top:48px'));
    assert.ok(plain(tag).includes(DRAFT_LABEL));
  }
  assert.equal(DRAFT_LABEL,'DRAFT, placeholder voice');
});
test('every scene renders at its start, midpoint and speech end',()=>{
  for(const w of windows) {
    const mid=Math.round((w.beat.start_s+w.beat.end_s)/2*timeline.fps)-w.from;
    for(const frame of [0,mid,w.speechStart+w.speechDuration-1]) assert.ok(shot(w.beat.id,frame).length>200,w.beat.id+' at '+frame);
  }
});
test('no em or en dash anywhere on screen, in facts.json or in scene source',()=>{
  for(const w of windows) for(const frame of ALL_SECOND(w.beat.id)) {
    const text=plain(shot(w.beat.id,frame));
    assert.ok(!/[\\u2013\\u2014]/.test(text),w.beat.id+' frame '+frame+' shows a dash: '+text.match(/.{0,40}[\\u2013\\u2014].{0,40}/)?.[0]);
  }
  assert.ok(!/[\\u2013\\u2014]/.test(readFileSync('src/content/facts.json','utf8')));
  for(const path of readdirSync('src',{recursive:true}).filter(p=>p.endsWith('.tsx'))) assert.ok(!/[\\u2013\\u2014]/.test(readFileSync('src/'+path,'utf8')),path);
  for(const b of timeline.beats) assert.ok(!/[\\u2013\\u2014]/.test(b.text),b.id+' caption text');
});
test('every spoken word has its caption on screen',()=>{
  for(const w of windows) {
    const spoken=w.beat.words?.length ? w.beat.words.map(word=>[Math.round(word.start_s*timeline.fps),Math.round(word.end_s*timeline.fps)]) : [[Math.round(w.beat.start_s*timeline.fps),Math.round(w.beat.end_s*timeline.fps)]];
    for(const [a,b] of spoken) for(let frame=a;frame<b;frame++) assert.ok(render(React.createElement(Captions),frame).length>0,'caption gap at '+frame+' in '+w.beat.id);
  }
});
test('cold open: real records stack from the first frame; the question lands on its words',()=>{
  const first=shot('cold_open',0);
  assert.ok((first.match(/data-record=/g)??[]).length>=12,'records on the first frame');
  for(const c of facts.cold_open.columns) assert.ok(plain(first).includes(c.label));
  const ask=at('cold_open','Who wrote');
  assert.ok(!first.includes('data-question'));
  assert.ok(plain(shot('cold_open',ask+20)).includes('Who wrote the technical spec, and who reviewed it?'));
});
test('intro: wordmark, then the schema; every brief label and edge is drawn by the end',()=>{
  assert.ok(!plain(shot('intro',at('intro','inherit-brain')-1)).includes('a company brain on FalkorDB'));
  assert.ok(plain(shot('intro',at('intro','a company brain')+5)).includes('a company brain on FalkorDB'));
  const end=shot('intro',win('intro').durationInFrames-1);
  for(const n of ['Person','Document','Meeting','Message','PR','Bug','BugEvent','Feature','Customer']) assert.ok(end.includes('data-node="'+n+'" data-lit="true"'),n);
  for(const e of ['AUTHORED','SUGGESTED','REVIEWED','OPENED','FIXES','HAS_EVENT','REPORTED_BY','ATTENDED','POSTED']) assert.ok(end.includes('data-edge="'+e+'"'),e);
});
test('walk_graph: the path picture arrives on "It walks the graph" and gives way to the id checklist',()=>{
  const w=at('walk_graph','It walks the graph'), t=at('walk_graph','Ten people');
  assert.ok(!shot('walk_graph',w-10).includes('data-path-picture'));
  const html=shot('walk_graph',w+12);
  assert.ok(html.includes('data-path-picture'));
  assert.ok(plain(html).includes('The path in the graph'));
  assert.ok(!shot('walk_graph',t+12).includes('data-path-picture'));
  assert.ok(plain(shot('walk_graph',t+12)).includes('All 10 ids returned'));
});
test('walk_graph: the Alice Davis rows are named with both ids on their cue',()=>{
  const a=at('walk_graph','two different people');
  assert.deepEqual(facts.walk_graph.alice_davis.slice().sort(),['eid_6e2f3e2b','eid_e5e36dc1']);
  assert.ok(!plain(shot('walk_graph',a-1)).includes('Two different people named Alice Davis'));
  const html=plain(shot('walk_graph',a+20));
  for(const eid of facts.walk_graph.alice_davis) assert.ok(html.includes(eid));
  assert.equal(facts.walk_graph.n_correct,10);
  assert.equal(facts.walk_graph.n_gold,10);
});
test('benchmark: the split and the question count on their words, then the hash and the posting time',()=>{
  assert.ok(!plain(shot('benchmark',at('benchmark','twelve hundred')-1)).includes('held-out questions'));
  assert.ok(plain(shot('benchmark',at('benchmark','twelve hundred')+15)).includes('1,203'));
  assert.equal((shot('benchmark',30).match(/data-product=/g)??[]).length,30);
  const end=plain(shot('benchmark',win('benchmark').durationInFrames-1));
  assert.ok(end.includes('d8f43632...'));
  assert.ok(end.includes('2026-10-17 02:50 UTC'));
});
test('arms: eight arms, inherit-brain in the accent, the four named ones dark by the end',()=>{
  const end=shot('arms',win('arms').durationInFrames-1);
  assert.equal((end.match(/data-arm=/g)??[]).length,8);
  for(const id of ['A0','A1f','A0v','A2']) assert.ok(end.includes('data-arm="'+id+'" data-named="true"'),id);
  for(const id of ['A4','A4c','A0f']) assert.ok(end.includes('data-arm="'+id+'" data-named="false"'),id);
  assert.ok(end.includes('color:#2d4fd0'));
  assert.ok(plain(end).includes('Qwen/Qwen3-30B-A3B-Instruct-2507'));
});
test('results: each number arrives on its spoken word',()=>{
  const scores=(f)=>[...shot('results',f).matchAll(/data-score="([0-9.]+)"/g)].map(m=>m[1]);
  assert.deepEqual(scores(at('results','eighty')-1),[]);
  assert.deepEqual(scores(at('results','eighty')),['80.0']);
  assert.deepEqual(scores(at('results','thirty-nine')),['80.0','39.2']);
  assert.deepEqual(scores(at('results','twenty-six')),['80.0','39.2','26.0']);
  const tokens=plain(shot('results',at('results','twelve times')+15));
  for(const s of ['10,117','125,947','12x']) assert.ok(tokens.includes(s),s);
  assert.ok(!plain(shot('results',at('results','forty points')-1)).includes('(35.7 to 46.2)'));
  const h1=plain(shot('results',at('results','forty points')+15));
  assert.ok(h1.includes('+40.8') && h1.includes('(35.7 to 46.2)'));
  const split=shot('results',at('results','twenty points')+15);
  assert.ok(plain(split).includes('+57.9') && plain(split).includes('(51.8 to 63.7)'));
  assert.ok(plain(split).includes('+20.1') && plain(split).includes('(12.9 to 26.5)'));
  const before=shot('results',at('results','twenty points')-1);
  assert.match(before,/opacity:0"><span data-unseen/);
});
test('audit: rounds 1 to 5 fail and round 6 passes, as in PREREG section 12',()=>{
  assert.deepEqual(facts.audit.map(r=>r.result),['failed','failed','failed','failed','failed','passed']);
  const end=shot('audit',win('audit').durationInFrames-1);
  const results=[...end.matchAll(/data-round="(\\d)" data-result="(\\w+)"/g)].map(m=>m[2]);
  assert.deepEqual(results,['failed','failed','failed','failed','failed','passed']);
  assert.ok(plain(end).includes('Failures kept on the record.'));
  const start=shot('audit',0);
  assert.equal((start.match(/data-result="pending"/g)??[]).length,6);
});
test('falkordb: the four items appear as named; the Cypher and the what-if counts are the data',()=>{
  const f=(p)=>at('falkordb',p);
  assert.ok(!plain(shot('falkordb',f('a read-only user')-1)).includes('A read-only user'));
  assert.ok(plain(shot('falkordb',f('a read-only user')+12)).includes('GRAPH.RO_QUERY'));
  const cypher=plain(shot('falkordb',f('its history')+12));
  for(const s of ['[fx:FIXES]','[op:OPENED]','[hs:HAS_EVENT]','SecurityForce/ans/21']) assert.ok(cypher.includes(s),s);
  assert.ok((shot('falkordb',f('its history')+12).match(/data-lit="true"/g)??[]).length>=5);
  const browserAt=f('in one query');
  assert.ok(!shot('falkordb',browserAt-1).includes('data-stage="browser"'));
  for(let frame=browserAt;frame<browserAt+Math.round(3.2*timeline.fps);frame++) assert.ok(shot('falkordb',frame).includes('data-stage="browser"'),'FalkorDB Browser on screen at '+frame);
  const browser=plain(shot('falkordb',browserAt+60));
  assert.ok(browser.includes('the same path in FalkorDB Browser'));
  assert.ok(browser.includes('7 nodes, 6 edges') && browser.includes('7.7 ms'));
  assert.ok(shot('falkordb',browserAt+60).includes('data-camera="falkordb_browser"'));
  const end=plain(shot('falkordb',win('falkordb').durationInFrames-1));
  assert.ok(end.includes('If Julia Smith leaves'));
  assert.ok(end.includes('2 open bugs') && end.includes('1 pull request'));
  assert.equal(facts.whatif.bugs,2); assert.equal(facts.whatif.prs,1);
});
test('check_it shows the path picture of SecurityForce/ans/21 on "Every answer"',()=>{
  const html=shot('check_it',at('check_it','Every answer')+15);
  assert.ok(html.includes('data-camera="explore_check"'));
  assert.ok(html.includes('data-mark="the answer"'));
});
test('check_it ends on the site address',()=>{
  assert.ok(plain(shot('check_it',win('check_it').durationInFrames-1)).includes('zeyusuntt.github.io/inherit-brain'));
});
test('close: line, name, site, repo, QR and track, held still through the end',()=>{
  const w=win('close');
  const last=shot('close',w.durationInFrames-1);
  for(const s of ['When you inherit a product,','inherit-brain','zeyusuntt.github.io/inherit-brain','github.com/zeyusuntt/inherit-brain','Track 3: Company Brain · Graph Hacks 2026']) assert.ok(plain(last).includes(s),s);
  assert.ok(last.includes('alt="https://zeyusuntt.github.io/inherit-brain/"'));
  const settled=at('close','inherit-brain lets you')+Math.round(timeline.fps*0.7);
  assert.equal(shot('close',settled),last,'the close card holds still from its last reveal to the end');
  assert.ok(w.durationInFrames-settled>=Math.round(timeline.fps*4),'close holds at least 4 s');
});
`},
  bundle: true, platform: 'node', format: 'esm', packages: 'external', outfile: 'out/checks/frames.mjs',
  plugins: [{name: 'omit-font-css-from-ssr', setup(b) {
    b.onResolve({filter: /\.css$/}, (args) => ({path: args.path, namespace: 'font-css'}));
    b.onLoad({filter: /.*/, namespace: 'font-css'}, () => ({contents: '', loader: 'js'}));
  }}],
});
await import('../out/checks/frames.mjs');
