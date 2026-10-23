// Move src/sources.js's git pins (kohra, diskformer) to each repo's current main.
//   node scripts/pin-sources.mjs           rewrite the pins, print old → new
//   node scripts/pin-sources.mjs --check   change nothing; exit 1 when a pin is behind main
// transformers.js is pinned by hand (in step with LocalMind); this prints npm's latest for reference.
// After moving pins, run the browser tests (test/browser/README.md) before committing them.
import { readFileSync, writeFileSync } from 'node:fs';

const FILE = new URL('../src/sources.js', import.meta.url);
const REPOS = { kohra: 'kohra', diskformer: 'diskformer.js' };
const check = process.argv.includes('--check');

const head = async (repo) => {
  const r = await fetch(`https://api.github.com/repos/NakliTechie/${repo}/commits/main`, { headers: { accept: 'application/vnd.github.sha' } });
  if (!r.ok) throw new Error(`${repo}: GitHub returned ${r.status}`);
  return (await r.text()).trim();
};

let src = readFileSync(FILE, 'utf8');
let behind = 0;
for (const [key, repo] of Object.entries(REPOS)) {
  const pinned = new RegExp(`(\\n  ${key}: ')([0-9a-f]{40})(')`).exec(src);
  if (!pinned) throw new Error(`no ${key} pin in src/sources.js`);
  const now = await head(repo);
  if (pinned[2] === now) { console.log(`${key.padEnd(11)} ${now.slice(0, 7)} (main)`); continue; }
  behind++;
  console.log(`${key.padEnd(11)} ${pinned[2].slice(0, 7)} → ${now.slice(0, 7)}`);
  src = src.replace(pinned[0], `${pinned[1]}${now}${pinned[3]}`);
}
const latest = (await (await fetch('https://registry.npmjs.org/@huggingface/transformers/latest')).json()).version;
console.log(`transformers ${/transformers: '([^']+)'/.exec(src)[1]} pinned by hand; npm latest ${latest}`);
if (check) process.exit(behind ? 1 : 0);
if (behind) writeFileSync(FILE, src);
