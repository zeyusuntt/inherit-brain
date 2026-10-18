// A contact sheet: one frame every 4 s as a labelled grid PNG, for reviewing legibility and density.
//   node scripts/contact-sheet.mjs                         from out/inherit-brain-draft.mp4 -> out/contact-sheet.png
//   node scripts/contact-sheet.mjs --video V --out O       from another render
//   node scripts/contact-sheet.mjs --frames DIR --out O    from stills already rendered (render-stills.mjs --every)
import {spawnSync} from 'node:child_process';
import {mkdirSync, readdirSync, rmSync} from 'node:fs';

const args = process.argv.slice(2);
const opt = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const every = Number(opt('--every', '4'));
const out = opt('--out', 'out/contact-sheet.png');
const cols = Number(opt('--cols', '5'));
const run = (cmd, a) => {
  const r = spawnSync(cmd, a, {stdio: ['ignore', 'pipe', 'inherit']});
  if (r.status !== 0) process.exit(r.status ?? 1);
  return r.stdout.toString();
};

let dir = opt('--frames', null);
if (!dir) {
  const video = opt('--video', 'out/inherit-brain-draft.mp4');
  dir = 'out/sheet-frames-video';
  rmSync(dir, {recursive: true, force: true});
  mkdirSync(dir, {recursive: true});
  // Frames at 0, 4, 8 ... seconds, each 960 px wide.
  run('ffmpeg', ['-v', 'error', '-y', '-i', video, '-vf', `fps=1/${every}:start_time=0:round=down,scale=960:-1`, '-start_number', '0', `${dir}/%03d.png`]);
}
const files = readdirSync(dir).filter((f) => f.endsWith('.png')).sort();
const rows = Math.ceil(files.length / cols);
// Tiles read left to right, top to bottom: tile n is at n x `every` seconds (a row of 5 spans 20 s).
// (This ffmpeg build has no drawtext, so the tiles carry no printed time.)
run('ffmpeg', ['-v', 'error', '-y', '-pattern_type', 'glob', '-i', `${dir}/*.png`, '-vf', `scale=960:-1,tile=${cols}x${rows}:padding=8:margin=8:color=white`, '-frames:v', '1', out]);
console.log(`${out}: ${files.length} frames, one every ${every} s, ${cols} per row (${cols * every} s per row)`);
