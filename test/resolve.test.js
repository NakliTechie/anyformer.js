import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from '../index.js';

const yes = (id, tokPerSec = null) => ({ id, canRun: () => ({ ok: true, estimate: { gpuBytes: 1, diskBytes: 1, tokPerSec } }) });
const no = (id, reason) => ({ id, canRun: () => ({ ok: false, reason }) });
const model = { repo: 'a/b' };

test('no back ends: nothing chosen', async () => {
  assert.deepEqual(await resolve(model, {}, []), { chosen: null, ranked: [], candidates: [] });
});

test('refusals are kept with their reasons and never chosen', async () => {
  const r = await resolve(model, {}, [no('x', 'no gguf'), yes('y')]);
  assert.equal(r.chosen, 'y');
  assert.deepEqual(r.ranked, ['y']);
  assert.deepEqual(r.candidates[0], { backend: 'x', order: 0, ok: false, code: 'model', reason: 'no gguf' });
});

test('a throwing canRun becomes a refusal with the message', async () => {
  const boom = { id: 'boom', canRun: () => { throw new Error('bad caps'); } };
  const r = await resolve(model, {}, [boom]);
  assert.equal(r.chosen, null);
  assert.equal(r.candidates[0].reason, 'canRun threw: bad caps');
});

test('an async canRun is awaited', async () => {
  const r = await resolve(model, {}, [{ id: 'a', canRun: async () => ({ ok: true, estimate: {} }) }]);
  assert.equal(r.chosen, 'a');
});

test('ranking: tok/s estimate beats none; higher tok/s first; ties keep listed order', async () => {
  const r = await resolve(model, {}, [yes('none1'), yes('slow', 5), yes('none2'), yes('fast', 20), yes('fast2', 20)]);
  assert.deepEqual(r.ranked, ['fast', 'fast2', 'slow', 'none1', 'none2']);
});

test('budget reaches canRun', async () => {
  let seen;
  await resolve(model, {}, [{ id: 'a', canRun: (m, c, o) => { seen = o.budget; return { ok: false, reason: '' }; } }], { budget: 3e9 });
  assert.equal(seen, 3e9);
});

test('refusal codes come from the closed set; budget refusals carry need', async () => {
  const { REFUSALS } = await import('../index.js');
  const tight = { id: 't', canRun: () => ({ ok: false, code: 'budget', need: 4e9, reason: 'small' }) };
  const r = await resolve(model, {}, [tight, no('x', 'r')]);
  assert.equal(r.candidates[0].need, 4e9);
  for (const c of r.candidates) assert.ok(c.code in REFUSALS, c.code);
});
