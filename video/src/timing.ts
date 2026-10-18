export type AlignedWord = {w: string; start_s: number; end_s: number};
export type Beat = {id: string; text: string; start_s: number; end_s: number; words?: AlignedWord[]};
export type WordTimestamp = {word: string; start: number; end: number};
export type CaptionChunk = {text: string; start: number; end: number};

export const frames = (seconds: number, fps: number) => Math.round(seconds * fps);
export const totalFrames = (seconds: number, fps: number) => Math.ceil(seconds * fps);

export function sequenceWindows<T extends Omit<Beat, 'text'>>(timeline: {fps: number; total_s: number; beats: T[]}, overlapSeconds = 0.2) {
  const total = totalFrames(timeline.total_s, timeline.fps);
  return timeline.beats.map((beat, index) => {
    const from = index === 0 ? 0 : frames(beat.start_s, timeline.fps);
    const next = timeline.beats[index + 1];
    const end = next ? Math.min(total, frames(next.start_s + overlapSeconds, timeline.fps)) : total;
    return {beat, from, durationInFrames: end - from, speechStart: frames(beat.start_s, timeline.fps) - from, speechDuration: frames(beat.end_s, timeline.fps) - frames(beat.start_s, timeline.fps)};
  });
}

// A recorded narration pauses between sentences; a caption stays up through a pause of up to `holdFrames`
// so it does not flicker off and on, and clears for a longer silence.
export function captionChunks(text: string, start: number, end: number, maxWords = 12, timestamps?: WordTimestamp[], holdFrames = 30): CaptionChunk[] {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const chunks: CaptionChunk[] = [];
  let first = 0;
  for (let i = 0; i < words.length; i++) {
    const sentenceEnd = /[.!?]["'’”]?$/u.test(words[i]);
    if (sentenceEnd || i - first + 1 >= maxWords || i === words.length - 1) {
      chunks.push({text: words.slice(first, i + 1).join(' '), start: timestamps?.[first]?.start ?? Math.round(start + (end - start) * first / words.length), end: timestamps?.[i]?.end ?? Math.round(start + (end - start) * (i + 1) / words.length)});
      first = i + 1;
    }
  }
  if (timestamps) {
    for (let i = 0; i < chunks.length - 1; i++) {
      if (chunks[i + 1].start - chunks[i].end <= holdFrames) chunks[i].end = chunks[i + 1].start;
    }
  }
  return chunks;
}

const normalizedWord = (value: string) => value.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
export function beatWordTimestamps(beat: Beat, fps: number, fallback?: WordTimestamp[]): WordTimestamp[] | undefined {
  return beat.words?.length ? beat.words.map(({w, start_s, end_s}) => ({word: w, start: frames(start_s, fps), end: frames(end_s, fps)})) : fallback;
}

export function beatCaptionChunks(beat: Beat, fps: number, timestamps?: WordTimestamp[]) {
  const aligned = beatWordTimestamps(beat, fps, timestamps);
  return captionChunks(beat.words?.length ? beat.words.map(w => w.w).join(' ') : beat.text, frames(beat.start_s, fps), frames(beat.end_s, fps), 12, aligned);
}

// Aligned words in timeline.json take priority over legacy frame props.
export function cueFrame(beat: Beat, phrase: string, fps: number, sequenceFrom = frames(beat.start_s, fps), timestamps?: WordTimestamp[]) {
  const words = beat.words?.length ? beat.words.map(w => w.w) : beat.text.trim().split(/\s+/);
  const tokens = phrase.trim().split(/\s+/).map(normalizedWord);
  const index = words.findIndex((_, i) => tokens.every((word, j) => normalizedWord(words[i + j] ?? '') === word));
  if (index < 0) throw new Error(`Narration cue missing: ${beat.id}: ${phrase}`);
  const start = frames(beat.start_s, fps);
  const end = frames(beat.end_s, fps);
  const aligned = beatWordTimestamps(beat, fps, timestamps);
  return aligned?.[index]?.start !== undefined ? aligned[index].start - sequenceFrom : Math.round(start + (end - start) * index / words.length) - sequenceFrom;
}

export function cueAnyFrame(beat: Beat, phrases: readonly string[], fps: number, sequenceFrom: number, timestamps?: WordTimestamp[]) {
  for (const phrase of phrases) {
    try { return cueFrame(beat, phrase, fps, sequenceFrom, timestamps); } catch { /* Try the next approved narration variant. */ }
  }
  throw new Error(`Narration cue missing: ${beat.id}: ${phrases.join(', ')}`);
}
