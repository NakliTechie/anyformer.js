import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TOOLS, call, registerTools, journal, toolOptions } from '../index.js';

const hub = JSON.parse(readFileSync(new URL('./fixtures/onnx-community_Qwen2.5-0.5B-Instruct.json', import.meta.url)));
const caps = { webgpu: true, f16: true, opfs: true };
const fake = {
  id: 'fake',
  canRun: () => ({ ok: true, estimate: { gpuBytes: 1, diskBytes: 1000, tokPerSec: null } }),
  load: async () => ({ async *generate(m) { yield { text: `echo ${m.at(-1).content}`, tokens: 2 }; }, dispose() {} }),
};
Object.assign(toolOptions, { caps, backends: [fake], fetch: async () => ({ ok: true, json: async () => hub }) });

test('manifest lint: closed consequence set, object schemas, required inputs declared, unique names', () => {
  assert.equal(new Set(TOOLS.map((t) => t.name)).size, TOOLS.length);
  for (const t of TOOLS) {
    assert.ok(['read', 'stages', 'mutates'].includes(t.consequence), t.name);
    assert.equal(t.inputSchema.type, 'object', t.name);
    for (const r of t.inputSchema.required || []) assert.ok(r in t.inputSchema.properties, `${t.name}.${r}`);
    assert.ok(t.description.length > 20 && typeof t.handler === 'function', t.name);
  }
});

test('parity: the manifest covers every command of the public API', () => {
  const commands = ['models', 'probe', 'plan', 'load', 'generate', 'sessions', 'unload', 'stored', 'remove'];
  assert.deepEqual(TOOLS.map((t) => t.name.replace('anyformer_', '')).sort(), commands.sort());
});

test('plan is bounded: no file list, the chosen back end and disk bytes', async () => {
  const p = await call('anyformer_plan', { repo: 'onnx-community/Qwen2.5-0.5B-Instruct' });
  assert.equal(p.chosen, 'fake');
  assert.equal(p.diskBytes, 1000);
  assert.equal(JSON.stringify(p).includes('model_q4'), false);
});

test('load stages until the caller accepts the disk bytes, then loads; generate and unload by session', async () => {
  const staged = await call('anyformer_load', { repo: 'onnx-community/Qwen2.5-0.5B-Instruct' });
  assert.equal(staged.loaded, false);
  assert.equal(staged.staged, true);
  const l = await call('anyformer_load', { repo: 'onnx-community/Qwen2.5-0.5B-Instruct', acceptDiskBytes: 1000 });
  assert.equal(l.loaded, true);
  const g = await call('anyformer_generate', { session: l.session, messages: [{ role: 'user', content: 'hi' }] });
  assert.deepEqual(g, { text: 'echo hi', tokens: 2 });
  assert.equal((await call('anyformer_sessions')).sessions.length, 1);
  assert.deepEqual(await call('anyformer_unload', { session: l.session }), { unloaded: true });
  assert.match((await call('anyformer_generate', { session: l.session, messages: [] })).error, /no session/);
});

test('errors come back as data; the journal records the door and stays at 50', async () => {
  assert.match((await call('nope')).error, /no tool nope/);
  for (let i = 0; i < 60; i++) await call('anyformer_sessions', {}, 'window');
  assert.equal(journal.length, 50);
  assert.equal(journal.at(-1).door, 'window');
});

test('registerTools publishes the same entries through WebMCP and window', async () => {
  const registered = [], target = {};
  const names = registerTools({ modelContext: { registerTool: (t) => registered.push(t) }, target });
  assert.deepEqual(registered.map((t) => t.name), names);
  assert.deepEqual(Object.keys(target.anyformer.tools), names);
  await registered.find((t) => t.name === 'anyformer_sessions').execute({});
  assert.equal(journal.at(-1).door, 'modelContext');
});

test('remove stages without confirm; stored and models read', async () => {
  const r = await call('anyformer_remove', { repo: 'a/b' });
  assert.deepEqual(r, { removed: false, staged: true, would: [] });
  assert.deepEqual(await call('anyformer_stored'), []);
  assert.ok((await call('anyformer_models')).chat.DEFAULT.repo);
});
