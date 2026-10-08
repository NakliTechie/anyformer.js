import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fromHubInfo, plan, runntimeBackend, RUNNTIME_MODELS, transformersBackend, kohraBackend, diskformerBackend, REFUSALS } from '../index.js';

const hub = (id, pipeline_tag) => fromHubInfo({ id, pipeline_tag, siblings: [{ rfilename: 'onnx/model.onnx', size: 1 }] });
const full = { webgpu: true, f16: true, opfs: true };
const B = { budget: 2.5e9 };

test('runntime: covered ids accepted for their own task, with the weight size', () => {
  const r = runntimeBackend.canRun(hub('Xenova/all-MiniLM-L6-v2', 'feature-extraction'), full, B);
  assert.deepEqual(r, { ok: true, estimate: { gpuBytes: 45e6, diskBytes: 45e6, tokPerSec: null } });
  assert.equal(runntimeBackend.canRun(hub('sentence-transformers/all-MiniLM-L6-v2', 'sentence-similarity'), full, B).ok, true);
  assert.equal(runntimeBackend.canRun(hub('onnx-community/yolo26n-ONNX', null), full, B).ok, true);  // no Hub tag
});

test('runntime: refusals for other ids, other tasks, no WebGPU, and f16-only tasks without f16', () => {
  assert.equal(runntimeBackend.canRun(hub('onnx-community/Qwen2.5-0.5B-Instruct', 'text-generation'), full, B).code, 'model');
  assert.equal(runntimeBackend.canRun(hub('Xenova/all-MiniLM-L6-v2', 'feature-extraction'), full, { ...B, task: 'text-generation' }).code, 'task');
  assert.equal(runntimeBackend.canRun(hub('Xenova/all-MiniLM-L6-v2', null), { ...full, webgpu: false }, B).code, 'webgpu');
  assert.equal(runntimeBackend.canRun(hub('onnx-community/yolo26n-ONNX', null), { ...full, f16: false }, B).code, 'f16');
  assert.equal(runntimeBackend.canRun(hub('Xenova/all-MiniLM-L6-v2', null), { ...full, f16: false }, B).ok, true);
});

test('runntime: every entry names a task the back end lists; tasks cover 6 kinds', () => {
  for (const m of Object.values(RUNNTIME_MODELS)) assert.ok(runntimeBackend.tasks.includes(m.task));
  assert.equal(runntimeBackend.tasks.length, 6);
});

test('task routing: the requested task picks the back end; a task nobody runs refuses with code task', async () => {
  const backends = [transformersBackend, kohraBackend, diskformerBackend, runntimeBackend];
  const mini = hub('Xenova/all-MiniLM-L6-v2', 'feature-extraction');
  assert.equal((await plan(mini.repo, { model: mini, caps: full, backends })).chosen, 'runntime');
  const qwen = fromHubInfo(JSON.parse(readFileSync(new URL('./fixtures/onnx-community_Qwen2.5-0.5B-Instruct.json', import.meta.url))));
  const p = await plan(qwen.repo, { model: qwen, caps: full, backends, task: 'feature-extraction' });
  assert.equal(p.chosen, null);
  assert.equal(p.candidates.find((c) => c.backend === 'transformers').code, 'task');
  assert.ok('task' in REFUSALS);
});
