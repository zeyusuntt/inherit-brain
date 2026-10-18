import {AbsoluteFill, Img, staticFile} from 'remotion';
import {facts} from '../content';
import {fonts, layout, palette, radius} from '../theme';
import {Appear} from '../components/Motion';
import {cuer, type SceneProps} from './types';

// The closing line, then the name, the site, the repository, a QR code to the site and the track.
export function Close(p: SceneProps) {
  const cue = cuer(p);
  const name = cue('inherit-brain lets you');
  return <AbsoluteFill style={{background: palette.page}}>
    <Appear at={0} style={{position: 'absolute', left: layout.margin, right: layout.margin, top: 190}}>
      <div data-closing-line style={{fontFamily: fonts.sans, fontSize: 96, fontWeight: 650, letterSpacing: -2.5, lineHeight: 1.12, color: palette.ink}}>When you inherit a product,<br/>you inherit its history.</div>
    </Appear>
    <Appear at={name} style={{position: 'absolute', left: layout.margin, right: layout.margin, top: 560, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', borderTop: `2px solid ${palette.line}`, paddingTop: 60}}>
      <div>
        <div style={{fontFamily: fonts.sans, fontSize: 170, fontWeight: 700, letterSpacing: -6, lineHeight: 1, color: palette.ink}}>inherit-brain</div>
        <div data-site style={{fontFamily: fonts.mono, fontSize: 60, fontWeight: 600, color: palette.ink, marginTop: 44}}>{facts.site_label}</div>
        <div data-repo style={{fontFamily: fonts.mono, fontSize: 48, color: palette.ink2, marginTop: 18}}>{facts.repo_label}</div>
        <div style={{fontFamily: fonts.sans, fontSize: 44, color: palette.ink3, marginTop: 40}}>Track 3: Company Brain · Graph Hacks 2026</div>
      </div>
      <div style={{textAlign: 'center'}}>
        <Img src={staticFile('qr-site.svg')} alt={facts.site_url} style={{width: 400, height: 400, borderRadius: radius, border: `2px solid ${palette.line}`, background: palette.white}}/>
        <div style={{fontFamily: fonts.sans, fontSize: 36, color: palette.ink3, marginTop: 14}}>Scan for the demo site</div>
      </div>
    </Appear>
  </AbsoluteFill>;
}
