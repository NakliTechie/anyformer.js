// LocalMind as the benchmark, step 1: which of LocalMind's chat models anyformer can plan, live against the Hub.
//   node bench/localmind-coverage.mjs [--budget GB]
// Rows come from test/fixtures/localmind-models.json (LocalMind's MODELS at a recorded commit). Caps are an Apple
// M-series Chrome (WebGPU, f16, OPFS); Qwen3.6 35B-A3B plans at 4 GB, its gated minimum. Prints one line per model and
// the count; exits 0 always (coverage is a measurement, not a gate yet).
import { readFileSync } from 'node:fs';
import { plan } from '../index.js';

const { models } = JSON.parse(readFileSync(new URL('../test/fixtures/localmind-models.json', import.meta.url)));
const caps = { webgpu: true, f16: true, subgroups: true, opfs: true, quotaBytes: 1e11, usageBytes: 0 };
let ran = 0, hub = 0;
for (const r of models) {
  const runtime = r.runtime || 'transformers.js';
  if (r.id.startsWith('chrome-builtin/')) { console.log(`${r.key.padEnd(28)} ${runtime.padEnd(15)} → not a Hub repo`); continue; }
  hub++;
  const repo = r.id.startsWith('https://') ? r.id.split('/').slice(3, 5).join('/') : r.id;
  const p = await plan(repo, { caps, budget: r.key.includes('qwen36') ? 4e9 : 2.5e9 });
  if (p.chosen) ran++;
  console.log(`${r.key.padEnd(28)} ${runtime.padEnd(15)} → ${p.chosen || 'none: ' + p.candidates.map((c) => `${c.backend} ${c.code}`).join(', ')}`);
}
console.log(`\nanyformer plans ${ran} of ${hub} LocalMind Hub models.`);
