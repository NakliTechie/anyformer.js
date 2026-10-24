import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fromHubInfo, plan, transformersBackend } from '../index.js';

const fixture = (name) => fromHubInfo(JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url))));
const bolt = fixture('danyalmalik_NVIDIA-Nemotron-3.5-Bolt-E4B-ONNX-WebGPU');
const caps = { webgpu: true, f16: true, subgroups: true, opfs: true };

test('transformers: a text-only Gemma 4 export loads through Gemma4ForCausalLM with external-data counts from its files', async () => {
  const p = await plan(bolt.repo, { model: bolt, caps, budget: 5.3e9 });
  assert.equal(p.chosen, 'transformers');
  assert.equal(p.candidates.find((c) => c.backend === 'transformers').estimate.diskBytes, 4905482405);
  const fake = await import('./fixtures/fake-engines/transformers.js');
  const was = transformersBackend.source;
  transformersBackend.source = new URL('./fixtures/fake-engines/transformers.js', import.meta.url).href;
  try {
    const s = await transformersBackend.load(bolt, { caps });
    const load = fake.calls.find((c) => c.type === 'load');
    assert.equal(load.cls, 'Gemma4ForCausalLM');
    assert.equal(load.opts.dtype, 'q4f16');
    assert.deepEqual(load.opts.use_external_data_format, { 'embed_tokens_q4f16.onnx': 1, 'decoder_model_merged_q4f16.onnx': 2 });
    let text = '';
    for await (const c of s.generate([{ role: 'user', content: [{ type: 'text', text: '17 x 23?' }] }])) text = c.text;
    assert.equal(text, '391');
    assert.deepEqual(fake.calls.find((c) => c.type === 'template').messages, [{ role: 'user', content: '17 x 23?' }]);
    await assert.rejects(async () => { for await (const _ of s.generate([{ role: 'user', content: [{ type: 'image', image: null }] }])); }, /reads text only/);
  } finally {
    transformersBackend.source = was;
  }
});
