import type {CSSProperties, ReactNode} from 'react';
import {fonts, layout, palette, radius} from '../theme';

// The scene heading at the top left: a small grey kicker over a title. The top right stays clear for the draft mark.
export function Heading({kicker, title, size = 72, style}: {kicker?: ReactNode; title?: ReactNode; size?: number; style?: CSSProperties}) {
  return <div style={{position: 'absolute', left: layout.margin, top: layout.top, maxWidth: 1800, ...style}}>
    {kicker && <div style={{fontFamily: fonts.sans, fontSize: 40, fontWeight: 500, color: palette.ink3, marginBottom: 12}}>{kicker}</div>}
    {title && <div style={{fontFamily: fonts.sans, fontSize: size, fontWeight: 650, color: palette.ink, letterSpacing: -1.5, lineHeight: 1.1}}>{title}</div>}
  </div>;
}

export function Chip({children, color = palette.ink2, background = palette.surface, border = palette.line, size = 38, mono = false, style}: {children: ReactNode; color?: string; background?: string; border?: string; size?: number; mono?: boolean; style?: CSSProperties}) {
  return <span style={{display: 'inline-flex', alignItems: 'center', gap: 14, fontFamily: mono ? fonts.mono : fonts.sans, fontSize: size, fontWeight: 500, color, background, border: `2px solid ${border}`, borderRadius: radius, padding: `${Math.round(size * 0.18)}px ${Math.round(size * 0.45)}px`, whiteSpace: 'nowrap', ...style}}>{children}</span>;
}

export const thousands = (n: number) => n.toLocaleString('en-US');

// One line under a camera frame that says what the highlight shows.
export function Note({children, color = palette.ink}: {children: ReactNode; color?: string}) {
  return <div style={{display: 'flex', alignItems: 'center', gap: 18, fontFamily: fonts.sans, fontSize: 44, fontWeight: 600, color}}>
    <span style={{width: 14, height: 14, borderRadius: 7, background: color, flex: 'none'}}/>{children}
  </div>;
}
