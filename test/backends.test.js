import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fromHubInfo, plan, transformersBackend, kohraBackend, diskformerBackend } from '../index.js';

const fixture = (name) => fromHubInfo(JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url))));
const qwen = fixture('onnx-community_Qwen2.5-0.5B-Instruct');
const gemma = fixture('google_gemma-4-26B-A4B-it-qat-q4_0-gguf');
const kohra = fixture('naklitechie_Qwen3-0.6B-diffusion-mdlm-ONNX');

const full = { webgpu: true, f16: true, subgroups: true, opfs: true, quotaBytes: 1e11, usageBytes: 0 };
const noF16 = { ...full, f16: false };
const noGpu = { ...full, webgpu: false };
const B = { budget: 2.5e9 };

test('transformers: Qwen2.5 ONNX accepted, q4f16 graph with f16, q4 without', () => {
  const r = transformersBackend.canRun(qwen, full, B);
  assert.equal(r.ok, true);
  assert.equal(r.estimate.gpuBytes, 483003582);                 // onnx/model_q4f16.onnx
  assert.equal(transformersBackend.canRun(qwen, noF16, B).estimate.gpuBytes, 786156820); // onnx/model_q4.onnx
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

test('resolver sends each repo to its back end', async () => {
  const backends = [transformersBackend, kohraBackend, diskformerBackend];
  for (const [model, want] of [[qwen, 'transformers'], [kohra, 'kohra'], [gemma, 'diskformer']]) {
    const p = await plan(model.repo, { model, caps: full, backends });
    assert.equal(p.chosen, want, model.repo);
    assert.deepEqual(p.ranked, [want]);
  }
});
