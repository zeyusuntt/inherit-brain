import '@fontsource-variable/geist/wght.css';
import '@fontsource-variable/geist-mono/wght.css';

// The demo site's light theme (site/assets/css/site.css). The accent belongs to inherit-brain (A5)
// only; every other arm and every neutral mark is grey. One radius for every container.
export const palette = {
  page: '#f6f7f8', surface: '#fcfcfd', surface2: '#eef0f2',
  ink: '#16181c', ink2: '#454b54', ink3: '#5f6670', mark: '#858c96', markSoft: '#c4c9cf',
  line: '#dcdfe3', lineStrong: '#c3c8ce',
  accent: '#2d4fd0', accentInk: '#2442b8', accentWash: '#e9eefc',
  ok: '#2e7a4d', okWash: '#e4f1e8', bad: '#b2413a', badWash: '#f8e7e5',
  codeBg: '#eef0f2', white: '#ffffff',
};
export const fonts = {sans: '"Geist Variable", "Geist", sans-serif', mono: '"Geist Mono Variable", "Geist Mono", monospace'};
export const radius = 6;
// Side margins 128; nothing but captions below contentBottom.
export const layout = {margin: 128, width: 2304, top: 110, contentBottom: 1220, captionBottom: 56};
