import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fromHubInfo, plan, localmindBackend, transformersBackend, kohraBackend, diskformerBackend, REFUSALS } from '../index.js';
import { localmindMatch } from '../src/backends/localmind.js';

const fx = (repo) => fromHubInfo(JSON.parse(readFileSync(new URL(`./fixtures/${repo.replace('/', '_')}.json`, import.meta.url))));
const full = { webgpu: true, f16: true, subgroups: true, opfs: true };
const big = { budget: 8e9 };

test('lfm2: any lfm2 GGUF, Q4_0 first, QAD and other repos skipped', () => {
  assert.deepEqual(localmindMatch(fx('LiquidAI/LFM2.5-230M-GGUF')), { engine: 'lfm2', file: 'LFM2.5-230M-Q4_0.gguf', bytes: fx('LiquidAI/LFM2.5-230M-GGUF').files.gguf.find((f) => f.name === 'LFM2.5-230M-Q4_0.gguf').size });
  assert.equal(localmindMatch(fx('LiquidAI/LFM2.5-1.2B-Instruct-GGUF')).file, 'LFM2.5-1.2B-Instruct-Q4_0.gguf');
  const r = localmindBackend.canRun(fx('LiquidAI/LFM2.5-230M-GGUF'), full, { budget: 2.5e9 });
  assert.equal(r.ok, true);
  assert.equal(r.estimate.tokPerSec, 1062.5);
  assert.equal(localmindBackend.canRun(fx('LiquidAI/LFM2.5-1.2B-Instruct-GGUF'), full, { budget: 2.5e9 }).estimate.tokPerSec, null);
});

test('gemma4: QAT-mobile exports only; E4B needs more than the default budget', () => {
  assert.equal(localmindMatch(fx('google/gemma-4-E2B-it-qat-mobile-transformers')).engine, 'gemma4');
  const e4b = localmindBackend.canRun(fx('google/gemma-4-E4B-it-qat-mobile-transformers'), full, { budget: 2.5e9 });
  assert.equal(e4b.code, 'budget');
  assert.ok(e4b.need > 3e9);
  assert.equal(localmindBackend.canRun(fx('google/gemma-4-E4B-it-qat-mobile-transformers'), full, big).ok, true);
});

test('qwen35: Prism ternary files only; mmproj never; other qwen35 GGUFs refused until tested', () => {
  assert.equal(localmindMatch(fx('prism-ml/Ternary-Bonsai-2-27B-gguf')).file, 'Ternary-Bonsai-2-27B-PTQ1_0.gguf');
  assert.equal(localmindBackend.canRun(fx('prism-ml/Ternary-Bonsai-2-27B-gguf'), { ...full, subgroups: false }, big).code, 'f16');
  assert.equal(localmindBackend.canRun(fx('unsloth/Qwen3.5-4B-GGUF'), full, big).code, 'model');
});

test('refusals use REFUSALS codes; ONNX and diskformer repos are not LocalMind\'s', () => {
  for (const repo of ['onnx-community/Qwen2.5-0.5B-Instruct', 'google/gemma-4-26B-A4B-it-qat-q4_0-gguf']) {
    const r = localmindBackend.canRun(fx(repo), full, big);
    assert.equal(r.ok, false);
    assert.ok(r.code in REFUSALS);
  }
});

test('the resolver sends LocalMind engine repos to localmind, and leaves the others where they were', async () => {
  const backends = [transformersBackend, kohraBackend, diskformerBackend, localmindBackend];
  const want = {
    'LiquidAI/LFM2.5-230M-GGUF': 'localmind', 'google/gemma-4-E2B-it-qat-mobile-transformers': 'localmind',
    'onnx-community/Qwen2.5-0.5B-Instruct': 'transformers', 'google/gemma-4-26B-A4B-it-qat-q4_0-gguf': 'diskformer',
  };
  for (const [repo, backend] of Object.entries(want)) assert.equal((await plan(repo, { model: fx(repo), caps: full, backends })).chosen, backend, repo);
});
