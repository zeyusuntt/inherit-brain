import assert from 'node:assert/strict';
import { test } from 'node:test';
import { build } from 'esbuild';

const bundle = await build({entryPoints: ['src/timing.ts'], bundle: true, write: false, format: 'esm', platform: 'node'});
const timing = await import(`data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString('base64')}`);

test('sequence windows cover lead-in, gaps and final hold after retiming', () => {
  assert.equal(typeof timing.sequenceWindows, 'function');
  const windows = timing.sequenceWindows({fps: 10, total_s: 12, beats: [{id: 'cold_open', start_s: 2, end_s: 4}, {id: 'other', start_s: 5, end_s: 9}]});
  assert.deepEqual(windows.map(({from, durationInFrames}) => ({from, durationInFrames})), [{from: 0, durationInFrames: 52}, {from: 50, durationInFrames: 70}]);
});

test('caption chunks preserve sentences and every word, with contiguous proportional timing', () => {
  assert.equal(typeof timing.captionChunks, 'function');
  const chunks = timing.captionChunks('One two. Three four five six seven.', 100, 240, 3);
  assert.deepEqual(chunks.map(({text}) => text), ['One two.', 'Three four five', 'six seven.']);
  assert.deepEqual(chunks.map(({start, end}) => [start, end]), [[100, 140], [140, 200], [200, 240]]);
});

test('word timestamp cues replace the proportional fallback', () => {
  assert.equal(typeof timing.captionChunks, 'function');
  const chunks = timing.captionChunks('One two. Three.', 0, 100, 12, [{word: 'One', start: 10, end: 20}, {word: 'two.', start: 21, end: 45}, {word: 'Three.', start: 60, end: 80}]);
  // A 15-frame pause is within the default 30-frame hold, so the first caption stays until the next.
  assert.deepEqual(chunks.map(({start, end}) => [start, end]), [[10, 60], [60, 80]]);
  const strict = timing.captionChunks('One two. Three.', 0, 100, 12, [{word: 'One', start: 10, end: 20}, {word: 'two.', start: 21, end: 45}, {word: 'Three.', start: 60, end: 80}], 10);
  assert.deepEqual(strict.map(({start, end}) => [start, end]), [[10, 45], [60, 80]]);
});

test('narration cues follow relocated and lengthened beats', () => {
  assert.equal(typeof timing.cueFrame, 'function');
  const beat = {id: 'x', start_s: 20, end_s: 30, text: 'One two three. Rejected. Five.'};
  assert.equal(timing.cueFrame(beat, 'Rejected', 10, 190), 70);
  assert.equal(timing.cueFrame({...beat, start_s: 40, end_s: 60}, 'Rejected', 10, 380), 140);
});

test('word-proportional cues use the same rounded frame window as captions', () => {
  const beat = {id: 'x', start_s: 0.016, end_s: 1.048, text: 'One two three.'};
  assert.deepEqual(timing.captionChunks(beat.text, 0, 31, 1).map(({start, end}) => [start, end]), [[0, 10], [10, 21], [21, 31]]);
  assert.equal(timing.cueFrame(beat, 'two', 30, 0), 10);
});

test('aligned timeline words drive cues without a separate timestamp prop', () => {
  const beat = {id: 'aligned', start_s: 20, end_s: 26, text: 'One two. Rejected.', words: [
    {w: 'One', start_s: 20.2, end_s: 20.7},
    {w: 'two.', start_s: 21.3, end_s: 22},
    {w: 'Rejected.', start_s: 24.1, end_s: 25},
  ]};
  assert.equal(timing.cueFrame(beat, 'Rejected', 30, 600), 123);
  assert.equal(typeof timing.beatCaptionChunks, 'function');
  assert.deepEqual(timing.beatCaptionChunks(beat, 30).map(({start, end}) => [start, end]), [[606, 660], [723, 750]]);
  const shifted = {...beat, start_s: 40, end_s: 46, words: beat.words.map(w => ({...w, start_s: w.start_s + 20, end_s: w.end_s + 20}))};
  assert.equal(timing.cueFrame(shifted, 'Rejected', 30, 1200), 123);
});

test('aligned cue matching follows the recorded words when transcription differs', () => {
  const beat = {id: 'aligned', start_s: 10, end_s: 14, text: 'This time it passes.', words: [
    {w: 'And', start_s: 10, end_s: 10.2}, {w: 'this', start_s: 10.4, end_s: 10.8},
    {w: 'time', start_s: 11, end_s: 11.4}, {w: 'it', start_s: 12, end_s: 12.2}, {w: 'passes.', start_s: 13, end_s: 13.6},
  ]};
  assert.equal(timing.cueFrame(beat, 'This time', 10, 100), 4);
  assert.equal(typeof timing.beatCaptionChunks, 'function');
  assert.equal(timing.beatCaptionChunks(beat, 10)[0].text, 'And this time it passes.');
});
