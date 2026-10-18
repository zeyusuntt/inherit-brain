// Render stills of the composition with one bundle and one browser.
//   node scripts/render-stills.mjs              one still per beat (mid-beat) -> out/stills/<beat>.png
//   node scripts/render-stills.mjs --every 4    one still every 4 s, half size -> out/sheet-frames/, then a grid
//                                                at out/contact-sheet-stills.png (a quick review before a full render)
//   node scripts/render-stills.mjs --at 33.5,41   stills at those seconds, half size -> out/at/<seconds>.png
import {bundle} from '@remotion/bundler';
import {openBrowser, renderStill, selectComposition} from '@remotion/renderer';
import {spawnSync} from 'node:child_process';
import {mkdirSync, readFileSync, rmSync} from 'node:fs';
import {resolve} from 'node:path';

const args = process.argv.slice(2);
const everyIndex = args.indexOf('--every');
const every = everyIndex >= 0 ? Number(args[everyIndex + 1]) : null;
const atIndex = args.indexOf('--at');
const atSeconds = atIndex >= 0 ? args[atIndex + 1].split(',').map(Number) : null;
const timeline = JSON.parse(readFileSync('src/content/timeline.json', 'utf8'));
const serveUrl = await bundle({entryPoint: resolve('src/index.ts')});
const browser = await openBrowser('chrome');
const composition = await selectComposition({serveUrl, id: 'InheritBrain', puppeteerInstance: browser});
const jobs = [];
if (atSeconds) {
  mkdirSync('out/at', {recursive: true});
  for (const t of atSeconds) jobs.push({frame: Math.min(composition.durationInFrames - 1, Math.round(t * timeline.fps)), output: `out/at/${t.toFixed(1)}.png`, scale: 0.5});
} else if (every) {
  rmSync('out/sheet-frames', {recursive: true, force: true});
  mkdirSync('out/sheet-frames', {recursive: true});
  for (let s = 0, i = 0; s * timeline.fps < composition.durationInFrames; s += every, i++) {
    jobs.push({frame: Math.min(composition.durationInFrames - 1, Math.round(s * timeline.fps)), output: `out/sheet-frames/${String(i).padStart(3, '0')}.png`, scale: 0.5});
  }
} else {
  mkdirSync('out/stills', {recursive: true});
  for (const beat of timeline.beats) {
    jobs.push({frame: Math.round((beat.start_s + beat.end_s) / 2 * timeline.fps), output: `out/stills/${beat.id}.png`, scale: 1});
  }
  jobs.push({frame: composition.durationInFrames - 1, output: 'out/stills/zz_last_frame.png', scale: 1});
}
for (const job of jobs) {
  await renderStill({composition, serveUrl, frame: job.frame, output: job.output, scale: job.scale, puppeteerInstance: browser, overwrite: true});
  console.log(`${job.output} (frame ${job.frame}, ${(job.frame / timeline.fps).toFixed(1)} s)`);
}
await browser.close({silent: true});
if (every) {
  const r = spawnSync('node', ['scripts/contact-sheet.mjs', '--frames', 'out/sheet-frames', '--out', 'out/contact-sheet-stills.png', '--every', String(every)], {stdio: 'inherit'});
  process.exit(r.status ?? 1);
}
