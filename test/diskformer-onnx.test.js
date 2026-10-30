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
    assert.ok(g2.opts.signal instanceof AbortSignal); // linked to ac.signal; the early-stop test covers the abort
    assert.equal(g2.opts.max_new_tokens, 8);
    assert.equal(g2.messages[0].content[0].audio, wave);
    const spoken = [];
    for await (const c of s.generate([{ role: 'user', content: 'Capital of France?' }], { speech: true, seed: 7, audioTemperature: 0 })) spoken.push(c);
    const g3 = fake.calls.filter((c) => c.type === 'generate')[2];
    assert.deepEqual(g3.opts.output_modalities, ['text', 'audio']);
    assert.equal(g3.opts.seed, 7);
    assert.equal(g3.opts.audio_temperature, 0);
    assert.deepEqual(spoken.filter((c) => c.audio).map((c) => [c.text, c.audio.length, c.sampleRate]), [['The capital', 4, 24000], ['The capital is Paris.', 4, 24000]]);
    assert.equal(g1.opts.output_modalities, undefined);
    await s.dispose();
    assert.ok(fake.calls.some((c) => c.type === 'dispose'));
    await diskformerOnnxBackend.load(omni, { caps, speech: true });
    assert.equal(fake.calls.filter((c) => c.type === 'load').at(-1).opts.speech, true);
  } finally {
    diskformerOnnxBackend.source = was;
  }
});

test('diskformer-onnx: a caller that stops reading early stops the engine before the next generate starts', async () => {
  const fake = await import('./fixtures/fake-engines/lfm3_omni.js');
  const was = diskformerOnnxBackend.source;
  diskformerOnnxBackend.source = new URL('./fixtures/fake-engines', import.meta.url).href;
  try {
    const s = await diskformerOnnxBackend.load(omni, { caps });
    const caller = new AbortController();
    for await (const c of s.generate([{ role: 'user', content: 'long' }], { signal: caller.signal })) if (c.text.length > 4) break;
    const end = fake.calls.filter((c) => c.type === 'end').at(-1);
    assert.equal(end.aborted, true);
    assert.ok(end.pieces < 1000);
    assert.equal(caller.signal.aborted, false);
    // The caller's own abort still reaches the engine.
    const ac = new AbortController();
    let n = 0;
    for await (const _ of s.generate([{ role: 'user', content: 'long' }], { signal: ac.signal })) if (++n === 3) ac.abort();
    assert.equal(fake.calls.filter((c) => c.type === 'end').at(-1).aborted, true);
  } finally {
    diskformerOnnxBackend.source = was;
  }
});
