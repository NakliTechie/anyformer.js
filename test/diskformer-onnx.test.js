import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fromHubInfo, plan, diskformerOnnxBackend, MODELS } from '../index.js';

const omni = fromHubInfo(JSON.parse(readFileSync(new URL('./fixtures/LiquidAI_LFM3-Omni-2B-ONNX.json', import.meta.url))));
const caps = { webgpu: true, f16: true, subgroups: true, opfs: true };

test('diskformer-onnx: LFM3-Omni routes to the hand engine, counting the three graphs it reads and the small files', async () => {
  const p = await plan(MODELS.chat.LFM3_OMNI_2B, { model: omni, caps });
  assert.equal(p.chosen, 'diskformer-onnx');
  assert.equal(p.candidates.find((c) => c.backend === 'transformers').code, 'task');
  assert.equal(p.candidates.find((c) => c.backend === 'diskformer-onnx').estimate.diskBytes, 2138391580);
  assert.equal(diskformerOnnxBackend.canRun(omni, { ...caps, f16: false }, {}).code, 'f16');
  const other = { ...omni, architecture: 'lfm3' };
  assert.equal(diskformerOnnxBackend.canRun(other, caps, {}).code, 'model');
  const thin = { ...omni, files: { ...omni.files, all: omni.files.all.filter((f) => !f.name.startsWith('onnx/audio_encoder.onnx')) } };
  assert.match(diskformerOnnxBackend.canRun(thin, caps, {}).reason, /onnx\/audio_encoder\.onnx/);
});

test('diskformer-onnx: generate streams the engine text cumulatively, thinking off by default, tools and signal passed through', async () => {
  const fake = await import('./fixtures/fake-engines/lfm3_omni.js');
  const was = diskformerOnnxBackend.source;
  diskformerOnnxBackend.source = new URL('./fixtures/fake-engines', import.meta.url).href;
  try {
    const events = [];
    const s = await diskformerOnnxBackend.load(omni, { caps, onProgress: (e) => events.push(e) });
    assert.equal(fake.calls[0].opts.baseUrl, `https://huggingface.co/LiquidAI/LFM3-Omni-2B-ONNX/resolve/${omni.revision}/`);
    assert.equal(s.baseUrl, fake.calls[0].opts.baseUrl);
    assert.equal(events[0].backend, 'diskformer-onnx');
    const chunks = [];
    for await (const c of s.generate([{ role: 'user', content: 'Capital of France?' }])) chunks.push(c);
    assert.deepEqual(chunks, [{ text: 'The capital', tokens: 1 }, { text: 'The capital is Paris.', tokens: 2 }]);
    const ac = new AbortController(), tools = [{ type: 'function', function: { name: 'f' } }], wave = new Float32Array(16);
    for await (const _ of s.generate([{ role: 'user', content: [{ type: 'audio', audio: wave }] }], { thinking: true, tools, signal: ac.signal, maxNewTokens: 8 }));
    const [g1, g2] = fake.calls.filter((c) => c.type === 'generate');
    assert.equal(g1.opts.enable_thinking, false);
    assert.equal(g1.opts.do_sample, false);
    assert.equal(g2.opts.enable_thinking, true);
    assert.equal(g2.opts.tools, tools);
    assert.equal(g2.opts.signal, ac.signal);
    assert.equal(g2.opts.max_new_tokens, 8);
    assert.equal(g2.messages[0].content[0].audio, wave);
    await s.dispose();
    assert.ok(fake.calls.some((c) => c.type === 'dispose'));
  } finally {
    diskformerOnnxBackend.source = was;
  }
});
