import {timeline} from '../content';
import {cueFrame, type Beat, type WordTimestamp} from '../timing';

export type SceneProps = {beat: Beat; sequenceFrom: number; speechStart: number; speechDuration: number; sceneDuration: number; wordTimestamps?: WordTimestamp[]};

// The frame (within the scene's sequence) where a phrase of the beat's narration is spoken.
export const cuer = (p: SceneProps) => (phrase: string) => cueFrame(p.beat, phrase, timeline.fps, p.sequenceFrom, p.wordTimestamps);
export const FPS = timeline.fps;
