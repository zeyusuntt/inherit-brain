import {AbsoluteFill, Audio, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import type {ReactNode} from 'react';
import {timeline} from './content';
import {frames, sequenceWindows, totalFrames} from './timing';
import {fonts, palette, radius} from './theme';
import {Captions, type CaptionTiming} from './components/Captions';
import {clamp} from './components/Motion';
import {ColdOpen} from './scenes/cold_open';
import {Intro} from './scenes/intro';
import {WalkGraph} from './scenes/walk_graph';
import {Benchmark} from './scenes/benchmark';
import {Arms} from './scenes/arms';
import {Results} from './scenes/results';
import {Audit} from './scenes/audit';
import {Falkordb} from './scenes/falkordb';
import {CheckIt} from './scenes/check_it';
import {Close} from './scenes/close';

export type VideoProps = {musicTrack?: string | null; musicVolume?: number; wordTimestamps?: CaptionTiming; showCaptions?: boolean};
export const scenes = {cold_open: ColdOpen, intro: Intro, walk_graph: WalkGraph, benchmark: Benchmark, arms: Arms, results: Results, audit: Audit, falkordb: Falkordb, check_it: CheckIt, close: Close};
export const DRAFT_LABEL = 'DRAFT, placeholder voice';

function Transition({children, first}: {children: ReactNode; first: boolean}) {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return <AbsoluteFill style={{opacity: first ? 1 : interpolate(frame, [0, Math.round(fps * 0.25)], [0, 1], clamp)}}>{children}</AbsoluteFill>;
}

export function Video({musicTrack = null, musicVolume = 0.08, wordTimestamps, showCaptions = true}: VideoProps) {
  const total = totalFrames(timeline.total_s, timeline.fps);
  return <AbsoluteFill style={{background: palette.page, color: palette.ink, fontFamily: fonts.sans}}>
    <Audio src={staticFile(timeline.voice)}/>
    {/* Music sits under the voice at a fixed low level and fades out over the last two seconds. */}
    {musicTrack && <Audio src={staticFile(musicTrack)} loop volume={(f) => musicVolume * interpolate(f, [total - frames(2, timeline.fps), total - 1], [1, 0], clamp)}/>}
    {sequenceWindows(timeline).map(({beat, from, durationInFrames, speechStart, speechDuration}, index) => {
      const Scene = scenes[beat.id as keyof typeof scenes];
      if (!Scene) throw new Error(`no scene for beat ${beat.id}`);
      return <Sequence key={beat.id} from={from} durationInFrames={durationInFrames} name={beat.id}>
        <Transition first={index === 0}><Scene beat={beat} sequenceFrom={from} speechStart={speechStart} speechDuration={speechDuration} sceneDuration={durationInFrames} wordTimestamps={wordTimestamps?.[beat.id]}/></Transition>
      </Sequence>;
    })}
    {showCaptions && <Captions wordTimestamps={wordTimestamps}/>}
    {/* A scratch timeline is a placeholder voice: mark every frame so a draft cannot pass for the final cut. */}
    {timeline.source.startsWith('scratch') && <div data-draft style={{position: 'absolute', right: 64, top: 48, padding: '6px 18px', borderRadius: radius, background: palette.bad, color: palette.white, fontFamily: fonts.mono, fontSize: 34, zIndex: 60}}>{DRAFT_LABEL}</div>}
  </AbsoluteFill>;
}
