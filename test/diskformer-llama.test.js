import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fromHubInfo, plan, BACKENDS, MODELS, REFUSALS, diskformerLlamaBackend, diskformerBackend, wllamaBackend } from '../index.js';
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
const partial = hub('Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF', 'qwen3next', [
  ['Q4_K_M/Qwen3-Next-80B-A3B-Instruct-Q4_K_M-00001-of-00003.gguf', 4e10], ['Q4_K_M/Qwen3-Next-80B-A3B-Instruct-Q4_K_M-00002-of-00003.gguf', 8e9]]);
const vlm = hub('unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF', 'qwen3vlmoe', [['mmproj-F16.gguf', 1e9], ['Qwen3-VL-30B-A3B-Instruct-Q3_K_M.gguf', 14.7e9]]);

const full = { webgpu: true, f16: true, subgroups: true, opfs: true, memory64: true, jspi: true, quotaBytes: 1e11, usageBytes: 0 };
const B = { budget: 2.5e9 };

test('diskformer-llama: picks the preferred quant, never a vision tower; a split set by its first part, with every part', () => {
  assert.equal(pickGguf(granite).name, 'granite-4.0-h-tiny-Q4_K_M.gguf');
  assert.equal(pickGguf(flint).name, 'flint-int4-q4_1.gguf');            // lower-case q4_1
  assert.equal(pickGguf(gemma).name, 'gemma-4-26B_q4_0-it.gguf');         // the quant mid-name
  assert.equal(pickGguf(vlm).name, 'Qwen3-VL-30B-A3B-Instruct-Q3_K_M.gguf');
  const s = pickGguf(split);
  assert.equal(s.name, 'Q4_K_M/Qwen3-Next-80B-A3B-Instruct-Q4_K_M-00001-of-00002.gguf');
  assert.equal(s.size, 4.8e10);
  assert.equal(s.parts.length, 2);
  assert.equal(pickGguf(split, s.name).size, 4.8e10);
  assert.equal(pickGguf(split, 'Q4_K_M/Qwen3-Next-80B-A3B-Instruct-Q4_K_M-00002-of-00002.gguf'), null);  // not a first part
  assert.equal(pickGguf(partial), null);                                    // 2 of 3 parts
  assert.equal(diskformerLlamaBackend.canRun(split, full, B).estimate.diskBytes, 4.8e10);
});

test('diskformer-llama: accepts a checked MoE architecture, with the file size as disk bytes', () => {
  const r = diskformerLlamaBackend.canRun(granite, full, B);
  assert.equal(r.ok, true);
  assert.deepEqual(r.estimate, { gpuBytes: 2.5e9, diskBytes: 4230976352, tokPerSec: null });
});

test('diskformer-llama: refuses dense architectures, ONNX, incomplete split sets and browsers without JSPI, memory64 or OPFS', () => {
  const cases = [
    [dense, full, 'model', /architecture qwen3 is not one/],
    [qwen, full, 'format', /no GGUF/],
    [partial, full, 'model', /split set needs every part/],
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

test('diskformer-llama: load downloads each part of a split set under its own name and loads the first', async () => {
  const fake = await import('./fixtures/fake-engines/llama.js');
  const was = { source: diskformerLlamaBackend.source, navigator: globalThis.navigator };
  const files = new Map();
  const handle = (name) => ({
    getFile: async () => ({ text: async () => files.get(name) ?? (() => { throw new Error('none'); })() }),
    createWritable: async () => ({ write: async (t) => files.set(name, t), close: async () => {} }),
  });
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { storage: { getDirectory: async () => ({ getFileHandle: async (n) => handle(n) }) } } });
  diskformerLlamaBackend.source = new URL('./fixtures/fake-engines/llama.js', import.meta.url).href;
  try {
    const seen = [];
    const s = await diskformerLlamaBackend.load(split, { budget: 6e9, onProgress: (e) => seen.push(e) });
    const downloads = fake.calls.filter((c) => c.type === 'download');
    assert.deepEqual(downloads.map((d) => [d.name, d.size]), [
      ['Qwen3-Next-80B-A3B-Instruct-Q4_K_M-00001-of-00002.gguf', 4e10], ['Qwen3-Next-80B-A3B-Instruct-Q4_K_M-00002-of-00002.gguf', 8e9]]);
    assert.match(downloads[1].url, /resolve\/main\/Q4_K_M\/Qwen3-Next-80B-A3B-Instruct-Q4_K_M-00002-of-00002\.gguf$/);
    assert.equal(fake.calls.find((c) => c.type === 'load').name, 'Qwen3-Next-80B-A3B-Instruct-Q4_K_M-00001-of-00002.gguf');
    assert.equal(seen.filter((e) => e.status === 'download').at(-1).loaded, 4.8e10);   // progress over the whole set
    assert.deepEqual(Object.values(JSON.parse(files.get('anyformer-llama.json'))), [split.repo, split.repo]);
    s.dispose();
  } finally {
    diskformerLlamaBackend.source = was.source;
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: was.navigator });
  }
});

const deepseek = fixture('ISTA-DASLab_DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF');

test('diskformer-llama: DeepSeek V4.1 Flash picks the IQ1_M set with its Engram side file, never the side file alone', () => {
  const f = pickGguf(deepseek);
  assert.equal(f.name, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf');
  assert.equal(f.parts.length, 4);
  assert.deepEqual(f.extra.map((x) => x.name), ['DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf']);
  assert.equal(f.size, 231316411072);
  assert.equal(pickGguf(deepseek, 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf'), null);
  const r = diskformerLlamaBackend.canRun(deepseek, full, { budget: 8e9 });
  assert.equal(r.ok, true);
  assert.equal(r.estimate.diskBytes, 231316411072);
  // The architecture is accepted for the demo file only: another quant of the same repo is refused.
  assert.equal(diskformerLlamaBackend.canRun(deepseek, full, { budget: 8e9, file: 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf' }).code, 'model');
  // Without the side file the set cannot load.
  const noEngram = hub(deepseek.repo, 'deepseek41-engram', deepseek.files.gguf.filter((x) => !/Engram/.test(x.name)).map((x) => [x.name, x.size]));
  assert.equal(diskformerLlamaBackend.canRun(noEngram, full, { budget: 8e9 }).ok, false);
});

test('diskformer-llama: load downloads the side file with the parts and passes it to llama.load as extra', async () => {
  const fake = await import('./fixtures/fake-engines/llama.js');
  fake.calls.length = 0;
  const was = { source: diskformerLlamaBackend.source, navigator: globalThis.navigator };
  const files = new Map();
  const handle = (name) => ({
    getFile: async () => ({ text: async () => files.get(name) ?? (() => { throw new Error('none'); })() }),
    createWritable: async () => ({ write: async (t) => files.set(name, t), close: async () => {} }),
  });
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { storage: { getDirectory: async () => ({ getFileHandle: async (n) => handle(n) }) } } });
  diskformerLlamaBackend.source = new URL('./fixtures/fake-engines/llama.js', import.meta.url).href;
  try {
    const s = await diskformerLlamaBackend.load(deepseek, { budget: 8e9 });
    const downloads = fake.calls.filter((c) => c.type === 'download').map((d) => d.name);
    assert.equal(downloads.length, 5);
    assert.equal(downloads.at(-1), 'DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf');
    const load = fake.calls.find((c) => c.type === 'load');
    assert.equal(load.name, 'DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf');
    assert.deepEqual(load.extra, ['DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf']);
    assert.equal(load.budget, 8e9);
    assert.equal(Object.keys(JSON.parse(files.get('anyformer-llama.json'))).length, 5);
    s.dispose();
  } finally {
    diskformerLlamaBackend.source = was.source;
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: was.navigator });
  }
});

const lfm24 = fixture('LiquidAI_LFM2.5-24B-A2B-GGUF');

test('an adapter routes to the engines that apply one: diskformer\'s LFM2.5 engine first, then diskformer-llama', async () => {
  const pi = MODELS.chat.LFM2_5_24B_PI;
  const p = await plan(pi, { model: lfm24, caps: full });
  assert.equal(p.chosen, 'diskformer');
  assert.deepEqual(p.candidates.map((c) => c.backend), ['diskformer', 'diskformer-llama']);
  assert.equal(p.adapter.file, 'LFM2.5-24B-A2B-harbor-pi-lora-F16.gguf');
  // Without the hand engine, the llama.cpp path applies it.
  assert.equal((await plan(pi, { model: lfm24, caps: full, backends: [diskformerLlamaBackend, wllamaBackend] })).chosen, 'diskformer-llama');
  // Gemma 4's hand engine applies one too (diskformer v0.6.20); Clover's has no adapter support, so it drops diskformer.
  const g = await plan(gemma.repo, { model: gemma, caps: full, budget: 4e9, adapter: pi.adapter });
  assert.equal(g.chosen, 'diskformer');
  const clover = fixture('Diffuspace_Clover-35B-A3B-Preview-GGUF');
  const c = await plan(clover.repo, { model: clover, caps: full, budget: 4e9, adapter: pi.adapter });
  assert.ok(!c.candidates.some((x) => x.backend === 'diskformer'));
});

test('diskformer: load applies the adapter at run time; setAdapter switches it off and on through load()', async () => {
  const { load } = await import('../index.js');
  const fake = await import('./fixtures/fake-engines/lfm2_moe_ssd.js');
  fake.calls.length = 0;
  const saved = { source: diskformerBackend.source, fetch: globalThis.fetch, caches: globalThis.caches };
  diskformerBackend.source = new URL('./fixtures/fake-engines', import.meta.url).href;
  // The adapter download: a fake 37,437,888-byte body, and a Cache Storage stand-in keyed by URL.
  const { size } = MODELS.chat.LFM2_5_24B_PI.adapter;
  const fetched = [], stored = new Map();
  globalThis.fetch = async (url) => { fetched.push(url); return new Response(new Uint8Array(size), { headers: { 'content-length': String(size) } }); };
  globalThis.caches = { open: async () => ({
    match: async (u) => stored.get(u)?.clone(), put: async (u, r) => { stored.set(u, r); }, delete: async (u) => stored.delete(u) }) };
  try {
    const progress = [];
    const s = await load(MODELS.chat.LFM2_5_24B_PI, { model: lfm24, caps: full, onProgress: (e) => { if (e.status === 'adapter') progress.push(e); } });
    assert.equal(s.backend, 'diskformer');
    const url = `https://huggingface.co/harness-hub/LFM2.5-24B-A2B-pi-GGUF/resolve/${MODELS.chat.LFM2_5_24B_PI.adapter.revision}/LFM2.5-24B-A2B-harbor-pi-lora-F16.gguf`;
    assert.deepEqual(fetched, [url]);
    const la = fake.calls.find((c) => c.type === 'loadAdapter');
    assert.ok(la.url instanceof Blob && la.url.size === size, 'the engine gets the adapter bytes, not a URL');
    assert.deepEqual(la.opts, { scale: 1 });
    assert.deepEqual([progress[0].loaded, progress.at(-1).loaded, progress.at(-1).total], [0, size, size]);
    assert.ok(stored.has(url), 'the adapter is kept in Cache Storage');
    const chunks = [];
    for await (const c of s.generate([{ role: 'user', content: 'hi' }])) chunks.push(c);
    assert.deepEqual(chunks.map((c) => [c.text, c.reasoning]), [['', '\nPlan.'], ['Answer.', '\nPlan.\n']]);
    let raw;
    for await (const c of s.generate([{ role: 'user', content: 'hi' }], { thinking: true })) raw = c;
    assert.equal(raw.text, '\nPlan.\n</think>\n\nAnswer.');
    assert.equal(raw.reasoning, undefined);
    await s.setAdapter(false);
    await s.setAdapter(true);
    assert.deepEqual(fake.calls.filter((c) => c.type === 'setAdapter').map((c) => c.adapter?.id ?? null), ['pi', null, 'pi']);
    await s.dispose();
    // A second load reads the cached copy: no download.
    const cachedProgress = [];
    const s2 = await load(MODELS.chat.LFM2_5_24B_PI, { model: lfm24, caps: full, onProgress: (e) => { if (e.status === 'adapter') cachedProgress.push(e); } });
    assert.equal(fetched.length, 1);
    assert.equal(cachedProgress.at(-1).cached, true);
    await s2.dispose();
  } finally { diskformerBackend.source = saved.source; globalThis.fetch = saved.fetch; globalThis.caches = saved.caches; }
});

test('diskformer-llama: load downloads the adapter beside the model and passes it to llama.load as lora', async () => {
  const fake = await import('./fixtures/fake-engines/llama.js');
  fake.calls.length = 0;
  const was = { source: diskformerLlamaBackend.source, navigator: globalThis.navigator };
  const files = new Map();
  const handle = (name) => ({
    getFile: async () => ({ text: async () => files.get(name) ?? (() => { throw new Error('none'); })() }),
    createWritable: async () => ({ write: async (t) => files.set(name, t), close: async () => {} }),
  });
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { storage: { getDirectory: async () => ({ getFileHandle: async (n) => handle(n) }) } } });
  diskformerLlamaBackend.source = new URL('./fixtures/fake-engines/llama.js', import.meta.url).href;
  try {
    const { adapter } = MODELS.chat.LFM2_5_24B_PI;
    const seen = [];
    const s = await diskformerLlamaBackend.load(lfm24, { budget: 4e9, adapter: { ...adapter, scale: 0.5 }, onProgress: (e) => seen.push(e) });
    const downloads = fake.calls.filter((c) => c.type === 'download');
    assert.deepEqual(downloads.map((d) => d.name), ['LFM2.5-24B-A2B-QAD-Q4_0.gguf', 'LFM2.5-24B-A2B-harbor-pi-lora-F16.gguf']);
    assert.equal(downloads[1].url, `https://huggingface.co/harness-hub/LFM2.5-24B-A2B-pi-GGUF/resolve/${adapter.revision}/LFM2.5-24B-A2B-harbor-pi-lora-F16.gguf`);
    const load = fake.calls.find((c) => c.type === 'load');
    assert.equal(load.lora, 'LFM2.5-24B-A2B-harbor-pi-lora-F16.gguf');
    assert.equal(load.loraScale, 0.5);
    assert.equal(seen.filter((e) => e.status === 'download').at(-1).loaded, 13578190880 + 37437888);
    assert.equal(JSON.parse(files.get('anyformer-llama.json'))['LFM2.5-24B-A2B-harbor-pi-lora-F16.gguf'], adapter.repo);
    s.dispose();
  } finally {
    diskformerLlamaBackend.source = was.source;
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: was.navigator });
  }
});
