import { test } from 'node:test';
import assert from 'node:assert/strict';
import { load, plan, DEFAULT_BUDGET, isAnyformerError, ERROR_CODES } from '../index.js';

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
  const s = await load('a/b', { caps, model, backends: [fakeBackend('x', { ok: false }), b], budget: 3e9, maxCtx: 2048, onProgress });
  assert.equal(s.backend, 't');
  assert.equal(b.calls[0].o.budget, 3e9);
  assert.equal(b.calls[0].o.onProgress, onProgress);
  assert.equal(b.calls[0].o.maxCtx, 2048);
  const out = [];
  for await (const c of s.generate([{ role: 'user', content: 'x' }])) out.push(c.text);
  assert.deepEqual(out, ['hi', 'hi 1']);
  await s.dispose();
  assert.equal(b.calls.at(-1), 'dispose');
});

test('load() throws NO_BACKEND listing every refusal', async () => {
  const err = await load('a/b', { caps, model, backends: [fakeBackend('x', { ok: false, reason: 'needs gguf' })] }).catch((e) => e);
  assert.ok(isAnyformerError(err, 'NO_BACKEND'));
  assert.equal(err.plan.candidates[0].code, 'model');
  assert.match(err.message, /no back end can run a\/b at a 2\.5 GB budget/);
  assert.match(err.message, /x \[model\]: needs gguf/);
  const none = await load('a/b', { caps, model, backends: [] }).catch((e) => e);
  assert.match(none.message, /no back ends registered/);
});

const slowBackend = (log) => ({
  id: 'slow',
  canRun: () => ({ ok: true, estimate: {} }),
  load: async () => ({
    async *generate(m) {
      log.push(`start ${m[0].content}`);
      await new Promise((r) => setTimeout(r, 20));
      yield { text: m[0].content, tokens: 1 };
      log.push(`end ${m[0].content}`);
    },
    dispose() { log.push('dispose'); },
  }),
});
const drain = async (it) => { let last; for await (const c of it) last = c; return last; };
const msg = (content) => [{ role: 'user', content }];

test('a session runs one generate at a time, in call order', async () => {
  const log = [];
  const s = await load('a/b', { caps, model, backends: [slowBackend(log)] });
  await Promise.all([drain(s.generate(msg('1'))), drain(s.generate(msg('2')))]);
  assert.deepEqual(log, ['start 1', 'end 1', 'start 2', 'end 2']);
});

test('a caller that stops iterating early releases the session', async () => {
  const log = [];
  const s = await load('a/b', { caps, model, backends: [slowBackend(log)] });
  for await (const c of s.generate(msg('1'))) break;
  assert.equal((await drain(s.generate(msg('2')))).text, '2');
});

test('dispose waits for the running call, then later calls throw DISPOSED', async () => {
  const log = [];
  const s = await load('a/b', { caps, model, backends: [slowBackend(log)] });
  const running = drain(s.generate(msg('1')));
  await new Promise((r) => setTimeout(r, 0));
  await s.dispose();
  await running;
  assert.deepEqual(log, ['start 1', 'end 1', 'dispose']);
  await assert.rejects(drain(s.generate(msg('2'))), (e) => isAnyformerError(e, 'DISPOSED'));
  await s.dispose();                                            // idempotent
  assert.equal(log.filter((l) => l === 'dispose').length, 1);
});

test('bad messages throw INVALID_ARGUMENT; a back-end failure is EXECUTION_FAILED with the cause', async () => {
  const s = await load('a/b', { caps, model, backends: [slowBackend([])] });
  await assert.rejects(drain(s.generate('hi')), (e) => isAnyformerError(e, 'INVALID_ARGUMENT'));
  const broken = { id: 'b', canRun: () => ({ ok: true, estimate: {} }),
    load: async () => ({ async *generate() { throw new Error('device lost'); } }) };
  const t = await load('a/b', { caps, model, backends: [broken] });
  await assert.rejects(drain(t.generate(msg('x'))), (e) => isAnyformerError(e, 'EXECUTION_FAILED') && e.cause.message === 'device lost');
});

test('load errors: LOAD_FAILED with the cause, ABORTED when the signal fired, lookup failures named', async () => {
  const failing = { id: 'f', canRun: () => ({ ok: true, estimate: {} }), load: async () => { throw new Error('404'); } };
  await assert.rejects(load('a/b', { caps, model, backends: [failing] }),
    (e) => isAnyformerError(e, 'LOAD_FAILED') && /^f: 404/.test(e.message));
  const ctl = new AbortController();
  const aborting = { id: 'f', canRun: () => ({ ok: true, estimate: {} }),
    load: async () => { ctl.abort('user'); throw new TypeError('fetch failed'); } };
  await assert.rejects(load('a/b', { caps, model, backends: [aborting], signal: ctl.signal }),
    (e) => isAnyformerError(e, 'ABORTED') && e.cause === 'user');
  await assert.rejects(load('x/y', { caps, backends: [], fetch: async () => ({ ok: false, status: 404 }) }),
    (e) => isAnyformerError(e, 'LOAD_FAILED') && /x\/y: .*404/.test(e.message));
});

test('isAnyformerError checks name and code, so a second copy of the library still matches', () => {
  const foreign = Object.assign(new Error('x'), { name: 'AnyformerError', code: 'DISPOSED' });
  assert.ok(isAnyformerError(foreign, 'DISPOSED'));
  assert.ok(!isAnyformerError(Object.assign(new Error('x'), { name: 'AnyformerError', code: 'NOPE' })));
  assert.ok(Object.keys(ERROR_CODES).length === 7);
});
