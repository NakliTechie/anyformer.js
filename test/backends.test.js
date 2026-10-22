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

test('transformers multimodal: gemma4 and qwen3_5 repos run as chat at q4f16, every module counted, budget enforced', () => {
  const q35 = fixture('onnx-community_Qwen3.5-4B-ONNX-OPT');
  const e2b = fixture('onnx-community_gemma-4-E2B-it-ONNX');
  const r = transformersBackend.canRun(q35, full, { budget: 8e9 });
  assert.equal(r.ok, true);
  const q4f16 = q35.files.all.filter((f) => /^onnx\/[a-z_]+_q4f16\.onnx/.test(f.name)).reduce((n, f) => n + f.size, 0);
  assert.equal(r.estimate.gpuBytes, q4f16);
  assert.equal(transformersBackend.canRun(e2b, full, { budget: 8e9, task: 'any-to-any' }).ok, true);
  assert.equal(transformersBackend.canRun(e2b, full, { budget: 8e9, task: 'feature-extraction' }).code, 'task');
  const tight = transformersBackend.canRun(e2b, full, { budget: 2.5e9 });
  assert.equal(tight.code, 'budget');
  assert.ok(tight.need > 3e9);
});

test('diskformer: Ling 3.0 Flash needs both parts at their sizes and a 6.8 GB budget; the store size is its disk bytes', async () => {
  const P1 = 'Q4_K_M/Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf', P2 = 'Q4_K_M/Ling-3.0-flash-Q4_K_M-00002-of-00002.gguf';
  const ling = (files) => fromHubInfo({ id: 'inclusionAI/Ling-3.0-flash-GGUF', gguf: { architecture: 'bailingmoe3' }, siblings: files.map(([rfilename, size]) => ({ rfilename, size })) });
  const both = ling([[P1, 49995534208], [P2, 27014610464]]);
  const caps = { ...full, memory64: true, jspi: true };
  const r = diskformerBackend.canRun(both, caps, { budget: 7e9 });
  assert.equal(r.ok, true);
  assert.deepEqual(r.estimate, { gpuBytes: 7e9, diskBytes: 80.7e9, tokPerSec: 5.53 });
  const small = diskformerBackend.canRun(both, caps, { budget: 4e9 });
  assert.equal(small.code, 'budget');
  assert.equal(small.need, 6.8e9);
  assert.equal(diskformerBackend.canRun(ling([[P1, 49995534208]]), caps, { budget: 7e9 }).code, 'model');
  assert.equal(diskformerBackend.canRun(ling([[P1, 49995534208], [P2, 1]]), caps, { budget: 7e9 }).code, 'model');
  // The default back ends: the hand engine; the llama.cpp path refuses a split set.
  const p = await plan(both.repo, { model: both, caps, budget: 7e9 });
  assert.equal(p.chosen, 'diskformer');
  assert.equal(p.candidates.find((c) => c.backend === 'diskformer-llama').code, 'model');
});

test('diskformer: generate() turns thinking off by default and on with { thinking: true }', async () => {
  const saved = diskformerBackend.source;
  diskformerBackend.source = new URL('./fixtures/fake-engines', import.meta.url).href;
  try {
    const { calls } = await import('./fixtures/fake-engines/gemma4_moe_ssd.js');
    const s = await diskformerBackend.load(gemma, { budget: 2.5e9 });
    for await (const _ of s.generate([{ role: 'user', content: 'hi' }])) { /* drain */ }
    for await (const _ of s.generate([{ role: 'user', content: 'hi' }], { thinking: true })) { /* drain */ }
    assert.deepEqual(calls.map((o) => o.enableThinking), [false, true]);
  } finally { diskformerBackend.source = saved; }
});

test('diskformer through load(): token ids, think markers, prime, poolBytes and localFile reach the caller and the engine', async () => {
  const { load } = await import('../index.js');
  const saved = diskformerBackend.source;
  diskformerBackend.source = new URL('./fixtures/fake-engines', import.meta.url).href;
  try {
    const { loads, primed } = await import('./fixtures/fake-engines/gemma4_moe_ssd.js');
    const caps = { ...full, memory64: true, jspi: true };
    const file = { name: 'gemma-4-26B_q4_0-it.gguf', size: 14439363584 };
    const s = await load(gemma.repo, { model: gemma, caps, backends: [diskformerBackend], budget: 7e9, poolBytes: 4 * 2 ** 30, localFile: file });
    const opts = loads.at(-1);
    assert.equal(opts.poolBytes, 4 * 2 ** 30);
    assert.equal(opts.gpuBudgetBytes, undefined);          // with poolBytes the budget only admits the model
    assert.equal(opts.localFile, file);
    assert.deepEqual(s.think, { openId: 100, closeId: 101, inPrompt: false, preamble: 'thought\n', always: null });
    await s.prime('You are helpful.');
    assert.deepEqual(primed, ['You are helpful.']);
    const chunks = [];
    for await (const c of s.generate([{ role: 'user', content: 'hi' }])) chunks.push(c);
    assert.deepEqual(chunks, [{ text: 'ok', token: 42, tokens: 1 }]);
    await load(gemma.repo, { model: gemma, caps, backends: [diskformerBackend], budget: 7e9 });
    assert.equal(loads.at(-1).gpuBudgetBytes, 7e9);        // no poolBytes: the engine sizes its pool from the budget
    await s.dispose();
    await assert.rejects(s.prime('x'), /disposed/);
  } finally { diskformerBackend.source = saved; }
});

test('diskformer: MiniCPM-V 4.7 goes to its hand engine (the only one that reads its images), not the llama.cpp path', async () => {
  const mini = fromHubInfo({ id: 'openbmb/MiniCPM-V-4.7-35B-A3B-gguf', pipeline_tag: 'image-text-to-text', gguf: { architecture: 'qwen35moe' },
    siblings: [{ rfilename: 'MiniCPM-V-4_7-35B-A3B-Q4_0.gguf', size: 19714550688 }, { rfilename: 'mmproj-model-f16.gguf', size: 1119575840 }] });
  const caps = { ...full, memory64: true, jspi: true };
  for (const budget of [4e9, 7e9]) {
    const p = await plan(mini.repo, { model: mini, caps, budget });
    assert.equal(p.chosen, 'diskformer', `budget ${budget}`);
    assert.ok(p.ranked.includes('diskformer-llama'));      // it could run the text, but ranks after
  }
  assert.equal((await plan(mini.repo, { model: mini, caps, budget: 7e9 })).candidates.find((c) => c.backend === 'diskformer').estimate.tokPerSec, 14.1);
  assert.equal(diskformerBackend.canRun(mini, caps, { budget: 3e9 }).need, 4e9);
});
