import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fromHubInfo, plan, transformersBackend, kohraBackend, diskformerBackend } from '../index.js';
import { budgetError } from '../src/backends/diskformer.js';

const fixture = (name) => fromHubInfo(JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url))));
const qwen = fixture('onnx-community_Qwen2.5-0.5B-Instruct');
const gemma = fixture('google_gemma-4-26B-A4B-it-qat-q4_0-gguf');
const kohra = fixture('naklitechie_Qwen3-0.6B-diffusion-mdlm-ONNX');

const full = { webgpu: true, f16: true, subgroups: true, opfs: true, quotaBytes: 1e11, usageBytes: 0 };
const noF16 = { ...full, f16: false };
const noGpu = { ...full, webgpu: false };
const B = { budget: 2.5e9 };

test('transformers: Qwen2.5 ONNX accepted with the q4 graph (f32 activations), with or without f16', () => {
  const r = transformersBackend.canRun(qwen, full, B);
  assert.equal(r.ok, true);
  assert.equal(r.estimate.gpuBytes, 786156820);                 // onnx/model_q4.onnx
  assert.equal(transformersBackend.canRun(qwen, noF16, B).estimate.gpuBytes, 786156820);
});

test('transformers: refuses GGUF, text-diffusion and no WebGPU', () => {
  assert.match(transformersBackend.canRun(gemma, full, B).reason, /no ONNX/);
  assert.match(transformersBackend.canRun(kohra, full, B).reason, /text-diffusion/);
  assert.match(transformersBackend.canRun(qwen, noGpu, B).reason, /WebGPU/);
});

test('kohra: MDLM repo accepted with the fp16 graph + data, q4 without f16; plain ONNX refused', () => {
  const r = kohraBackend.canRun(kohra, full, B);
  assert.equal(r.ok, true);
  assert.equal(r.estimate.gpuBytes, 1210740 + 1536804864);
  assert.equal(kohraBackend.canRun(kohra, noF16, B).estimate.gpuBytes, 1257258 + 680077312);
  assert.match(kohraBackend.canRun(qwen, full, B).reason, /not a text-diffusion/);
  assert.match(kohraBackend.canRun(kohra, noGpu, B).reason, /WebGPU/);
});

test('diskformer: pinned Gemma GGUF accepted at 2.5 GB with its measured speed', () => {
  const r = diskformerBackend.canRun(gemma, full, B);
  assert.deepEqual(r, { ok: true, estimate: { gpuBytes: 2.5e9, diskBytes: 14439363584, tokPerSec: 12.4 } });
  assert.equal(diskformerBackend.canRun(gemma, full, { budget: 5e9 }).estimate.tokPerSec, 14.2);
});

test('diskformer: refuses small budgets, no f16, no OPFS, ONNX and unknown GGUFs', () => {
  assert.match(diskformerBackend.canRun(gemma, full, { budget: 1e9 }).reason, /under the 2.5 GB/);
  assert.match(diskformerBackend.canRun(gemma, noF16, B).reason, /shader-f16/);
  assert.match(diskformerBackend.canRun(gemma, { ...full, opfs: false }, B).reason, /OPFS/);
  assert.match(diskformerBackend.canRun(qwen, full, B).reason, /no GGUF/);
  const other = { ...gemma, repo: 'someone/other-gguf' };
  assert.match(diskformerBackend.canRun(other, full, B).reason, /no diskformer engine/);
});

test('every back-end refusal uses a code from REFUSALS', async () => {
  const { REFUSALS } = await import('../index.js');
  const cases = [[qwen, noGpu], [gemma, full], [kohra, full], [gemma, noF16], [gemma, { ...full, opfs: false }], [qwen, full]];
  for (const b of [transformersBackend, kohraBackend, diskformerBackend])
    for (const [m, c] of cases) {
      const r = b.canRun(m, c, { budget: 1e9 });
      if (!r.ok) assert.ok(r.code in REFUSALS, `${b.id} ${m.repo}: ${r.code}`);
    }
  assert.equal(diskformerBackend.canRun(gemma, full, { budget: 1e9 }).need, 2.5e9);
});

test('resolver sends each repo to its back end', async () => {
  const backends = [transformersBackend, kohraBackend, diskformerBackend];
  for (const [model, want] of [[qwen, 'transformers'], [kohra, 'kohra'], [gemma, 'diskformer']]) {
    const p = await plan(model.repo, { model, caps: full, backends });
    assert.equal(p.chosen, want, model.repo);
    assert.deepEqual(p.ranked, [want]);
  }
});

test('diskformer: the engine\'s load-time budget refusal becomes BUDGET_TOO_SMALL with need', () => {
  const e = budgetError(new Error('GPU budget 2.50 GB is too small: weights that stay on the GPU, the KV cache and working buffers need 3.04 GB, and the expert cache at least 0.05 GB more'));
  assert.equal(e.code, 'BUDGET_TOO_SMALL');
  assert.equal(e.need, 3.09e9);
  assert.equal(budgetError(new Error('no WebGPU adapter')), null);
});

const mini = fixture('Xenova_all-MiniLM-L6-v2');
const detr = fixture('Xenova_detr-resnet-50');

test('transformers: non-generation tasks run on the fp32 graph; unknown tasks refuse with code task', () => {
  const e = transformersBackend.canRun(mini, full, B);
  assert.equal(e.ok, true);
  assert.equal(e.estimate.gpuBytes, mini.files.onnx.find((f) => f.name === 'onnx/model.onnx').size);
  assert.equal(transformersBackend.canRun(detr, full, B).ok, true);
  assert.equal(transformersBackend.canRun(mini, full, { ...B, task: 'text-to-speech' }).code, 'task');
});
