import type {CSSProperties, ReactNode} from 'react';
import {interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';

export const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

// Fade and rise in at `at` (a frame of the current sequence); nothing is rendered before it.
export function Appear({at = 0, children, style, rise = 28}: {at?: number; children: ReactNode; style?: CSSProperties; rise?: number}) {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  if (frame < at) return null;
  const p = spring({frame: frame - at, fps, config: {damping: 24, stiffness: 130}, durationInFrames: Math.round(fps * 0.6)});
  return <div style={{opacity: interpolate(frame - at, [0, 12], [0, 1], clamp), transform: `translateY(${(1 - p) * rise}px)`, ...style}}>{children}</div>;
}

// 0 before `at`, easing to 1 over `duration` frames.
export function useProgress(at: number, duration = 15) {
  const frame = useCurrentFrame();
  const t = interpolate(frame, [at, at + Math.max(1, duration)], [0, 1], clamp);
  return 1 - (1 - t) ** 3;
}
