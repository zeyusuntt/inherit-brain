import {Composition, continueRender, delayRender} from 'remotion';
import {useEffect, useState} from 'react';
import {Video, type VideoProps} from './Video';
import {timeline} from './content';
import {totalFrames} from './timing';

export const FPS = timeline.fps;
export const WIDTH = 2560;
export const HEIGHT = 1440;
export const SECONDS = timeline.total_s;

// Hold each frame until Geist and Geist Mono have loaded, so no frame renders in a fallback face.
function InheritBrain(props: VideoProps) {
  const [handle] = useState(() => delayRender('Loading Geist fonts'));
  useEffect(() => {
    const faces = ['400 48px "Geist Variable"', '650 48px "Geist Variable"', '700 48px "Geist Variable"', '400 48px "Geist Mono Variable"', '600 48px "Geist Mono Variable"'];
    Promise.all(faces.map((f) => document.fonts.load(f))).then(() => document.fonts.ready).then(() => continueRender(handle)).catch(() => continueRender(handle));
  }, [handle]);
  return <Video {...props}/>;
}

export const Root: React.FC = () => <>
  <Composition
    id="InheritBrain"
    component={InheritBrain}
    durationInFrames={totalFrames(SECONDS, FPS)}
    fps={FPS}
    width={WIDTH}
    height={HEIGHT}
    defaultProps={{musicTrack: 'audio/music-wallpaper-norm.mp3', musicVolume: 0.08} satisfies VideoProps}
  />
</>;
