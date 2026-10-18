import {readFileSync, writeFileSync} from 'node:fs';
import QRCode from 'qrcode';

// The end card's QR code points at the demo site; the URL comes from facts.json.
const facts = JSON.parse(readFileSync(new URL('../src/content/facts.json', import.meta.url), 'utf8'));
const url = facts.site_url;
const svg = await QRCode.toString(url, {type: 'svg', errorCorrectionLevel: 'M', margin: 3, color: {dark: '#16181c', light: '#ffffff'}});
writeFileSync(new URL('../public/qr-site.svg', import.meta.url), svg);
console.log(`Generated offline QR for ${url}`);
