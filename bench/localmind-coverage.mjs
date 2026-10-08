// LocalMind as the benchmark, step 1: which of LocalMind's chat models anyformer can plan, live against the Hub.
//   node bench/localmind-coverage.mjs [--budget GB]   (default 8: what a 24 GB Apple M-series Mac gives a tab)
// Rows come from test/fixtures/localmind-models.json (LocalMind's MODELS at a recorded commit). Caps are an Apple
// M-series Chrome (WebGPU, f16, OPFS). Prints one line per model and
// the count; exits 0 always (coverage is a measurement, not a gate yet).
import { readFileSync } from 'node:fs';
import { plan } from '../index.js';

const { models } = JSON.parse(readFileSync(new URL('../test/fixtures/localmind-models.json', import.meta.url)));
const i = process.argv.indexOf('--budget');
const budget = (i > 0 ? Number(process.argv[i + 1]) : 8) * 1e9;
const caps = { webgpu: true, f16: true, subgroups: true, opfs: true, quotaBytes: 1e11, usageBytes: 0 };
let ran = 0, hub = 0;
for (const r of models) {
  const runtime = r.runtime || 'transformers.js';
  if (r.id.startsWith('chrome-builtin/')) { console.log(`${r.key.padEnd(28)} ${runtime.padEnd(15)} → not a Hub repo`); continue; }
  hub++;
  const repo = r.id.startsWith('https://') ? r.id.split('/').slice(3, 5).join('/') : r.id;
  const p = await plan(repo, { caps, budget });
  if (p.chosen) ran++;
  console.log(`${r.key.padEnd(28)} ${runtime.padEnd(15)} → ${p.chosen || 'none: ' + p.candidates.map((c) => `${c.backend} ${c.code}`).join(', ')}`);
}
console.log(`\nanyformer plans ${ran} of ${hub} LocalMind Hub models at a ${budget / 1e9} GB budget.`);
