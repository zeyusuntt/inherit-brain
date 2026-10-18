import timeline from './content/timeline.json';
import facts from './content/facts.json';
import captures from './content/captures.json';
export {timeline, facts, captures};
export type CaptureName = keyof typeof captures;
export type BeatId = typeof timeline.beats[number]['id'];
