import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fromHubInfo, plan, BACKENDS, REFUSALS, diskformerLlamaBackend, diskformerBackend, wllamaBackend } from '../index.js';
import { pickGguf } from '../src/backends/diskformer-llama.js';

const fixture = (name) => fromHubInfo(JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url))));
const gemma = fixture('google_gemma-4-26B-A4B-it-qat-q4_0-gguf');
const qwen = fixture('onnx-community_Qwen2.5-0.5B-Instruct');
const hub = (id, architecture, files) => fromHubInfo({ id, gguf: { architecture }, pipeline_tag: 'text-generation', siblings: files.map(([rfilename, size]) => ({ rfilename, size })) });
const granite = hub('ibm-granite/granite-4.0-h-tiny-GGUF', 'granitehybrid', [
  ['granite-4.0-h-tiny-Q2_K.gguf', 2.6e9], ['granite-4.0-h-tiny-Q4_K_M.gguf', 4230976352], ['granite-4.0-h-tiny-Q8_0.gguf', 7.4e9]]);
const flint = hub('RecursiveSI/flint-int4-gguf', 'flint', [['flint-int4-q4_1.gguf', 25992966976]]);
const dense = hub('Qwen/Qwen3-4B-GGUF', 'qwen3', [['Qwen3-4B-Q4_K_M.gguf', 2.5e9]]);
const split = hub('Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF', 'qwen3next', [
  ['Q4_K_M/Qwen3-Next-80B-A3B-Instruct-Q4_K_M-00001-of-00002.gguf', 4e10], ['Q4_K_M/Qwen3-Next-80B-A3B-Instruct-Q4_K_M-00002-of-00002.gguf', 8e9]]);
const vlm = hub('unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF', 'qwen3vlmoe', [['mmproj-F16.gguf', 1e9], ['Qwen3-VL-30B-A3B-Instruct-Q3_K_M.gguf', 14.7e9]]);

const full = { webgpu: true, f16: true, subgroups: true, opfs: true, memory64: true, jspi: true, quotaBytes: 1e11, usageBytes: 0 };
const B = { budget: 2.5e9 };

test('diskformer-llama: picks the preferred single-file quant, never a vision tower or a split part', () => {
  assert.equal(pickGguf(granite).name, 'granite-4.0-h-tiny-Q4_K_M.gguf');
  assert.equal(pickGguf(flint).name, 'flint-int4-q4_1.gguf');            // lower-case q4_1
  assert.equal(pickGguf(gemma).name, 'gemma-4-26B_q4_0-it.gguf');         // the quant mid-name
  assert.equal(pickGguf(vlm).name, 'Qwen3-VL-30B-A3B-Instruct-Q3_K_M.gguf');
  assert.equal(pickGguf(split), null);
});

test('diskformer-llama: accepts a checked MoE architecture, with the file size as disk bytes', () => {
  const r = diskformerLlamaBackend.canRun(granite, full, B);
  assert.equal(r.ok, true);
  assert.deepEqual(r.estimate, { gpuBytes: 2.5e9, diskBytes: 4230976352, tokPerSec: null });
});

test('diskformer-llama: refuses dense architectures, ONNX, split sets and browsers without JSPI, memory64 or OPFS', () => {
  const cases = [
    [dense, full, 'model', /architecture qwen3 is not one/],
    [qwen, full, 'format', /no GGUF/],
    [split, full, 'model', /no single-file GGUF/],
    [granite, { ...full, jspi: false }, 'webgpu', /memory64 and JSPI/],
    [granite, { ...full, memory64: false }, 'webgpu', /memory64 and JSPI/],
    [granite, { ...full, opfs: false }, 'opfs', /OPFS/],
    [granite, { ...full, webgpu: false }, 'webgpu', /no WebGPU/],
  ];
  for (const [m, caps, code, reason] of cases) {
    const r = diskformerLlamaBackend.canRun(m, caps, B);
    assert.equal(r.ok, false, m.repo);
    assert.equal(r.code, code, m.repo);
    assert.ok(code in REFUSALS);
    assert.match(r.reason, reason);
  }
  assert.equal(diskformerLlamaBackend.canRun(granite, full, { ...B, task: 'feature-extraction' }).code, 'task');
});

test('resolver: a diskformer hand engine outranks the llama.cpp path for its model; other MoE models go to the llama.cpp path', async () => {
  const g = await plan(gemma.repo, { model: gemma, caps: full });
  assert.equal(g.chosen, 'diskformer');
  assert.deepEqual(g.ranked.slice(0, 2), ['diskformer', 'diskformer-llama']);
  const t = await plan(granite.repo, { model: granite, caps: full });
  assert.equal(t.chosen, 'diskformer-llama');
  // wllama could hold a small model whole, but a checked MoE architecture pages experts through diskformer-llama first.
  const small = hub('allenai/OLMoE-1B-7B-0125-Instruct-GGUF', 'olmoe', [['OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf', 3e9]]);
  assert.equal((await plan(small.repo, { model: small, caps: full, budget: 4e9, backends: [diskformerLlamaBackend, wllamaBackend] })).chosen, 'diskformer-llama');
  assert.equal((await plan(dense.repo, { model: dense, caps: full, budget: 4e9 })).candidates.find((c) => c.backend === 'diskformer-llama').code, 'model');
  assert.ok(BACKENDS.indexOf(diskformerBackend) < BACKENDS.indexOf(diskformerLlamaBackend));
});

test('diskformer-llama: prefers the file diskformer checked; a caller can choose another', async () => {
  const glm = hub('unsloth/GLM-4.7-Flash-GGUF', 'deepseek2', [['GLM-4.7-Flash-Q4_K_M.gguf', 18312339808], ['GLM-4.7-Flash-Q3_K_M.gguf', 14614624608]]);
  assert.equal(pickGguf(glm).name, 'GLM-4.7-Flash-Q3_K_M.gguf');                 // checked, though Q4_K_M ranks first
  assert.equal(pickGguf(glm, 'GLM-4.7-Flash-Q4_K_M.gguf').name, 'GLM-4.7-Flash-Q4_K_M.gguf');
  assert.equal(diskformerLlamaBackend.canRun(glm, full, B).estimate.diskBytes, 14614624608);
  assert.equal(diskformerLlamaBackend.canRun(glm, full, { ...B, file: 'GLM-4.7-Flash-Q4_K_M.gguf' }).estimate.diskBytes, 18312339808);
  const r = diskformerLlamaBackend.canRun(glm, full, { ...B, file: 'nope.gguf' });
  assert.equal(r.code, 'model');
  assert.match(r.reason, /nope\.gguf is not/);
  const p = await plan(glm.repo, { model: glm, caps: full, file: 'GLM-4.7-Flash-Q4_K_M.gguf' });
  assert.equal(p.file, 'GLM-4.7-Flash-Q4_K_M.gguf');
  assert.equal(p.candidates.find((c) => c.backend === 'diskformer-llama').estimate.diskBytes, 18312339808);
});
