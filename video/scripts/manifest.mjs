// Index the captures for the composition: image sizes in CSS pixels (captures are 2x), element boxes
// and text boxes, written to src/content/captures.json. Usage, from video/: node scripts/manifest.mjs
import {readdir, readFile, writeFile} from 'node:fs/promises';
const dir = 'public/captures';
const size = async (file) => {
  const b = await readFile(`${dir}/${file}`);
  return {w: b.readUInt32BE(16) / 2, h: b.readUInt32BE(20) / 2};
};
const out = {};
for (const f of (await readdir(dir)).filter((f) => f.endsWith('.json')).sort()) {
  const name = f.replace(/\.json$/, '');
  const meta = JSON.parse(await readFile(`${dir}/${f}`, 'utf8'));
  out[name] = {url: meta.url, viewport: await size(`${name}.png`), full: await size(`${name}-full.png`), boxes: meta.boxes, needles: meta.needles, missing: meta.missing};
}
await writeFile('src/content/captures.json', JSON.stringify(out, null, 1) + '\n');
console.log(Object.keys(out).length, 'captures indexed:', Object.keys(out).join(', '));
