import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pipeline, isAnyformerError } from '../index.js';

const caps = { webgpu: true, f16: true, opfs: true };
const model = { repo: 'a/b', formats: ['onnx'], task: 'text-generation' };
const echo = {
  id: 'echo', canRun: (m, c, { task }) => ({ ok: true, estimate: {}, task }),
  load: async (m, o) => ({ task: o.task, async *generate(msgs) { yield { text: `re: ${msgs.at(-1).content}`, tokens: 2 }; }, dispose() {} }),
};

test('text-generation: messages in, messages + reply out; a string in, the reply string out', async () => {
  const gen = await pipeline('text-generation', 'a/b', { caps, model, backends: [echo] });
  assert.equal(gen.backend, 'echo');
  const [{ generated_text }] = await gen([{ role: 'user', content: 'hi' }], { max_new_tokens: 8 });
  assert.deepEqual(generated_text, [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 're: hi' }]);
  assert.deepEqual(await gen('yo'), [{ generated_text: 're: yo' }]);
  await gen.dispose();
});

test('the task reaches the plan; other tasks return run() output; no repo is INVALID_ARGUMENT', async () => {
  let seen;
  const emb = { id: 'emb', canRun: (m, c, { task }) => { seen = task; return { ok: true, estimate: {} }; },
    load: async () => ({ task: 'feature-extraction', run: async (x) => [[x.length]], dispose() {} }) };
  const fe = await pipeline('feature-extraction', 'a/b', { caps, model, backends: [emb] });
  assert.equal(seen, 'feature-extraction');
  assert.deepEqual(await fe('four'), [[4]]);
  await assert.rejects(pipeline('text-generation'), (e) => isAnyformerError(e, 'INVALID_ARGUMENT'));
});
