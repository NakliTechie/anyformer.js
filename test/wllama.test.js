import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fromHubInfo, plan, wllamaBackend, localmindBackend, diskformerBackend, transformersBackend, kohraBackend } from '../index.js';

const fx = (repo) => fromHubInfo(JSON.parse(readFileSync(new URL(`./fixtures/${repo.replace('/', '_')}.json`, import.meta.url))));
const full = { webgpu: true, f16: true, subgroups: true, opfs: true };
const minicpm = fx('openbmb/MiniCPM5-2B-GGUF');

test('wllama: any single-file GGUF up to 2 GB, Q4_K_M first; on the GPU when it fits the budget, else the CPU', () => {
  const r = wllamaBackend.canRun(minicpm, full, { budget: 2.5e9 });
  assert.equal(r.ok, true);
  assert.equal(r.estimate.diskBytes, minicpm.files.gguf.find((f) => f.name === 'MiniCPM5-2B-Q4_K_M.gguf').size);
  assert.equal(r.estimate.gpuBytes, r.estimate.diskBytes);
  assert.equal(wllamaBackend.canRun(minicpm, full, { budget: 1e9 }).estimate.gpuBytes, 0);          // CPU
  assert.equal(wllamaBackend.canRun(minicpm, { webgpu: false }, { budget: 2.5e9 }).ok, true);       // no WebGPU needed
});

test('wllama: refuses ONNX repos, files over 2 GB, and other tasks', () => {
  assert.equal(wllamaBackend.canRun(fx('onnx-community/Qwen2.5-0.5B-Instruct'), full, { budget: 8e9 }).code, 'format');
  assert.equal(wllamaBackend.canRun(fx('prism-ml/Ternary-Bonsai-2-27B-gguf'), full, { budget: 8e9 }).code, 'model');   // no K-quant ≤ 2 GB
  assert.equal(wllamaBackend.canRun(minicpm, full, { budget: 8e9, task: 'feature-extraction' }).code, 'task');
});

test('an architecture-specific engine wins over wllama for the same GGUF', async () => {
  const backends = [transformersBackend, kohraBackend, diskformerBackend, localmindBackend, wllamaBackend];
  assert.equal((await plan('x', { model: fx('LiquidAI/LFM2.5-230M-GGUF'), caps: full, backends })).chosen, 'localmind');
  assert.equal((await plan('x', { model: minicpm, caps: full, backends })).chosen, 'wllama');
  const gemma = await plan('x', { model: fx('google/gemma-4-26B-A4B-it-qat-q4_0-gguf'), caps: full, backends, budget: 2.5e9 });
  assert.equal(gemma.chosen, 'diskformer');
});
