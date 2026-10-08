import { test } from 'node:test';
import assert from 'node:assert/strict';
import { load, plan, NoBackendError, DEFAULT_BUDGET } from '../index.js';

const caps = { webgpu: true, f16: true };
const model = { repo: 'a/b', formats: ['onnx'] };

function fakeBackend(id, { ok = true, reason = 'no' } = {}) {
  const calls = [];
  return {
    id, calls,
    canRun: () => (ok ? { ok: true, estimate: { gpuBytes: 1, diskBytes: 0, tokPerSec: null } } : { ok: false, reason }),
    load: async (m, o) => {
      calls.push({ m, o });
      return {
        async *generate(messages) { yield { text: 'hi', tokens: 1 }; yield { text: 'hi ' + messages.length, tokens: 2 }; },
        dispose() { calls.push('dispose'); },
      };
    },
  };
}

test('plan() uses the default budget and passes caps and model through', async () => {
  const p = await plan('a/b', { caps, model, backends: [fakeBackend('t')] });
  assert.equal(p.budget, DEFAULT_BUDGET);
  assert.equal(p.chosen, 't');
  assert.equal(p.model, model);
});

test('plan() calls describe() with the injected fetch when no model is given', async () => {
  const fetch = async () => ({ ok: true, json: async () => ({ id: 'a/b', siblings: [{ rfilename: 'm.onnx', size: 1 }] }) });
  const p = await plan('a/b', { caps, fetch, backends: [] });
  assert.deepEqual(p.model.formats, ['onnx']);
});

test('load() loads the chosen back end and streams its output', async () => {
  const b = fakeBackend('t');
  const onProgress = () => {};
  const s = await load('a/b', { caps, model, backends: [fakeBackend('x', { ok: false }), b], budget: 3e9, onProgress });
  assert.equal(s.backend, 't');
  assert.equal(b.calls[0].o.budget, 3e9);
  assert.equal(b.calls[0].o.onProgress, onProgress);
  const out = [];
  for await (const c of s.generate([{ role: 'user', content: 'x' }])) out.push(c.text);
  assert.deepEqual(out, ['hi', 'hi 1']);
  s.dispose();
  assert.equal(b.calls.at(-1), 'dispose');
});

test('load() throws NoBackendError listing every refusal', async () => {
  const err = await load('a/b', { caps, model, backends: [fakeBackend('x', { ok: false, reason: 'needs gguf' })] }).catch((e) => e);
  assert.ok(err instanceof NoBackendError);
  assert.match(err.message, /no back end can run a\/b at a 2\.5 GB budget/);
  assert.match(err.message, /x \[model\]: needs gguf/);
  const none = await load('a/b', { caps, model, backends: [] }).catch((e) => e);
  assert.match(none.message, /no back ends registered/);
});
