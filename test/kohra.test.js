import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fromHubInfo, plan, kohraBackend } from '../index.js';

const fixture = (name) => fromHubInfo(JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url))));
const nemotron = fixture('naklitechie_Nemotron-Labs-Diffusion-3B-ONNX');
const caps = { webgpu: true, f16: true, subgroups: true, opfs: true };

test('kohra: Nemotron-Labs-Diffusion-3B loads its seeded KV graph with its own tokens and template', async () => {
  const p = await plan(nemotron.repo, { model: nemotron, caps });
  assert.equal(p.chosen, 'kohra');
  assert.equal(p.candidates.find((c) => c.backend === 'kohra').estimate.diskBytes, 1067632 + 1995503616 + 738852864);
  const fake = await import('./fixtures/fake-engines/kohra.js');
  const was = kohraBackend.source;
  kohraBackend.source = new URL('./fixtures/fake-engines/kohra.js', import.meta.url).href;
  try {
    const seen = [];
    const s = await kohraBackend.load(nemotron, { caps, onProgress: (e) => seen.push(e) });
    assert.deepEqual(seen.at(-1), { backend: 'kohra', status: 'progress', file: 'model_kv_q4f16.onnx_data', loaded: 1, total: 2, progress: 50 });
    const { opts } = fake.calls.find((c) => c.type === 'load');
    assert.match(opts.model, /\/resolve\/[^/]+\/onnx\/model_kv_q4f16\.onnx$/);
    assert.equal(opts.kvCache, true);
    assert.equal(opts.maskId, 100n);
    assert.equal(opts.eosId, 11n);
    assert.equal(opts.ortVersion, undefined);
    let out;
    for await (const c of s.generate([{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'Hello.' }, { role: 'user', content: [{ type: 'text', text: '17 x 23?' }] }])) out = c;
    assert.equal(out.text, '391');
    const g = fake.calls.find((c) => c.type === 'generate');
    // The one-turn case equals kohra's demo template (index.html nemotronChat), extended over the conversation.
    assert.equal(g.prompt, '<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think></think>Hello.<|im_end|>\n<|im_start|>user\n17 x 23?<|im_end|>\n<|im_start|>assistant\n<think></think>');
    assert.equal(g.o.blockCausal, false);
    assert.equal(g.o.chatTemplate, false);
    assert.equal(g.o.threshold, 0.9);
    for await (const _ of s.generate([{ role: 'user', content: 'hi' }], { threshold: null }));
    assert.equal(fake.calls.filter((c) => c.type === 'generate').at(-1).o.threshold, null);
  } finally {
    kohraBackend.source = was;
  }
});
