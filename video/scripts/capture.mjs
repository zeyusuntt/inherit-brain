// Capture the demo site's real pages for the video: a viewport and a full-page screenshot at 2x (1440
// CSS pixels wide, light theme), plus the boxes of the elements and text the scenes frame, so scenes
// push in on real UI instead of redrawing it. The site is served read-only from ../site on a local port.
// Usage, from video/: node scripts/capture.mjs [name ...]   (no names = all), then node scripts/manifest.mjs
import {chromium} from 'playwright';
import {spawn} from 'node:child_process';
import {mkdir, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';

const PORT = Number(process.env.SITE_PORT ?? 8765);
const BASE = `http://127.0.0.1:${PORT}/`;
const SITE = resolve('..', 'site');

// Elements whose boxes are recorded, with their visible text (first 200 characters).
const ELEMENTS = 'h1, h2, h3, h4, .chip, .step, .step-head, .kv, pre.cypher, .qblock, .q-details, table.rows, table.rows tr, '
  + '.ans-a5, .ans-head, .chips, .gold, .arm-list, .arm-list li, .d-q, .d-tags, .d-nav, .qlist li, .ex-side, .ex-count, .filters, '
  + '.steps, .diag, .evidence li, .people > li, .person-sum, .person-body, .items li, .wi-head, .wi-foot, .wi-side, .wi-tabs, '
  + '.hero-copy, .specimen, #scores, .scores, .removal-claim, .file, .gist, .blob-wrapper, .gist-meta, table.highlight';

export const PAGES = {
  explore_q3: {path: 'explore.html#q=SupportForce/ans/3', ready: '.step', open: 'details.q-details',
    find: ['doc_people', 'AUTHORED', 'SUGGESTED', 'eid_6e2f3e2b', 'eid_e5e36dc1', 'Alice Davis', 'Gold answer', "A5's path", 'evidence', 'First 5 of 10 rows.']},
  explore_check: {path: 'explore.html#q=CoachForce/ans/11', ready: '.step', open: 'details.q-details',
    find: ['Gold answer', 'Answers', "A5's path", 'A5 inherit-brain', 'GRAPH.RO_QUERY', 'Evidence']},
  whatif: {path: 'whatif.html', ready: '.people', select: ['#wi-product', 'SupportForce'],
    find: ['Julia Smith', 'Newly unowned (3)', 'Incorrect Error Log Parsing', 'Inefficient Query Execution Plan', 'Integrate User Testing Session Scheduling', 'GRAPH.COPY']},
  index: {path: 'index.html', ready: '.specimen .chip', find: ['80.0', '39.2', '26.0', 'inherit-brain']},
  gist: {url: 'https://gist.github.com/zeyusuntt/3012ee12697e7afade6d820e95d6bd67', wait: 5000,
    find: ['d8f4363276e972bbfb9bc2714fcceae128ddede4d163b00ffe4aaac602eadffd', 'PREREG', 'sha256']},
};

async function serverUp() {
  try { return (await fetch(BASE + 'index.html')).ok; } catch { return false; }
}

const names = process.argv.slice(2);
const todo = names.length ? names : Object.keys(PAGES);
let server = null;
if (!(await serverUp())) {
  server = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1', '-d', SITE], {stdio: 'ignore'});
  for (let i = 0; i < 50 && !(await serverUp()); i++) await new Promise((r) => setTimeout(r, 100));
}
await mkdir('public/captures', {recursive: true});
const browser = await chromium.launch();
const context = await browser.newContext({viewport: {width: 1440, height: 900}, deviceScaleFactor: 2, colorScheme: 'light', reducedMotion: 'reduce'});
try {
  for (const name of todo) {
    const spec = PAGES[name];
    const page = await context.newPage();
    const url = spec.url ?? BASE + spec.path;
    let status = null;
    try {
      const resp = await page.goto(url, {waitUntil: 'networkidle', timeout: 60000});
      status = resp?.status() ?? null;
    } catch (error) {
      console.log(name, 'unreachable:', error.message.split('\n')[0]);
      await page.close();
      continue;
    }
    if (spec.select) {
      await page.waitForSelector(`${spec.select[0]} option[value="${spec.select[1]}"]`, {state: 'attached'});
      await page.selectOption(spec.select[0], spec.select[1]);
    }
    if (spec.ready) await page.waitForSelector(spec.ready, {timeout: 20000});
    if (spec.open) await page.$$eval(spec.open, (els) => els.forEach((el) => el.setAttribute('open', '')));
    await page.waitForTimeout(spec.wait ?? 600);
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.screenshot({path: `public/captures/${name}.png`});
    await page.screenshot({path: `public/captures/${name}-full.png`, fullPage: true});
    const {boxes, needles} = await page.evaluate(({selector, find}) => {
      const boxes = [];
      for (const el of document.querySelectorAll(selector)) {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        boxes.push({tag: el.tagName.toLowerCase(), cls: String(el.className || ''), text: (el.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 200),
          x: r.x + window.scrollX, y: r.y + window.scrollY, width: r.width, height: r.height});
      }
      const needles = [];
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
      while (walker.nextNode()) {
        const node = walker.currentNode;
        for (const needle of find) {
          let from = 0;
          for (;;) {
            const i = node.textContent.indexOf(needle, from);
            if (i < 0) break;
            from = i + needle.length;
            const range = document.createRange();
            range.setStart(node, i);
            range.setEnd(node, i + needle.length);
            const r = range.getBoundingClientRect();
            if (r.width === 0) continue;
            needles.push({needle, x: r.x + window.scrollX, y: r.y + window.scrollY, width: r.width, height: r.height});
          }
        }
      }
      return {boxes, needles};
    }, {selector: ELEMENTS, find: spec.find ?? []});
    const found = new Set(needles.map((b) => b.needle));
    const missing = (spec.find ?? []).filter((n) => !found.has(n));
    await writeFile(`public/captures/${name}.json`, JSON.stringify({url: spec.url ?? spec.path, status, boxes, needles, missing}, null, 1));
    console.log(name, status, `boxes ${boxes.length}, needles ${needles.length}`, missing.length ? `MISSING ${missing.join(' | ')}` : '');
    await page.close();
  }
} finally {
  await browser.close();
  server?.kill();
}
