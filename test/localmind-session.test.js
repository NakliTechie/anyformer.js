import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fromHubInfo, localmindBackend } from '../index.js';

const hub = (id, architecture, files) => fromHubInfo({ id, gguf: architecture && { architecture }, config: !architecture ? { model_type: 'gemma4' } : undefined,
  siblings: files.map(([rfilename, size]) => ({ rfilename, size })) });
const bonsai = hub('prism-ml/Ternary-Bonsai-2-27B-gguf', 'qwen35', [['Ternary-Bonsai-2-27B-PTQ1_0.gguf', 5.94e9]]);
const gemma = hub('google/gemma-4-E2B-it-qat-mobile-transformers', null, [['model.safetensors', 2.46e9]]);
const fakes = new URL('./fixtures/fake-engines', import.meta.url).href;

// Cache Storage and a Range-serving fetch, as a browser has them.
function browserish(bytes) {
  const stores = new Map();
  globalThis.caches = { open: async (n) => { if (!stores.has(n)) stores.set(n, new Map()); const m = stores.get(n);
    return { match: async (k) => (m.has(k) ? new Response(m.get(k)) : undefined), put: async (k, r) => m.set(k, await r.blob()) }; } };
  let requests = 0;
  const fetch = async (url, { headers } = {}) => {
    requests++;
    const [, a, b] = /bytes=(\d+)-(\d+)/.exec(headers.Range);
    return new Response(bytes.slice(+a, +b + 1), { status: 206, headers: { 'content-range': `bytes ${a}-${b}/${bytes.length}` } });
  };
  return { fetch, requests: () => requests };
}

test('localmind: Bonsai thinking per request, assistantPrefix and token ids reach the engine; reset passes through', async () => {
  const saved = localmindBackend.source;
  localmindBackend.source = fakes;
  try {
    const { log } = await import('./fixtures/fake-engines/ternary_bonsai_2_27b.js');
    const s = await localmindBackend.load(bonsai, { maxCtx: 8192 });
    assert.equal(log.loads.at(-1).maxLength, 8192);
    const out = [];
    for await (const c of s.generate([{ role: 'user', content: 'hi' }], { thinking: true, assistantPrefix: 'x' })) out.push(c);
    assert.deepEqual(out, [{ text: 'a', token: 7, tokens: 1 }, { text: 'ab', token: 8, tokens: 2 }]);
    assert.deepEqual(log.gens.at(-1).args, { enable_thinking: true, reasoning_effort: 'medium' });
    assert.equal(log.gens.at(-1).opts.assistantPrefix, 'x');
    for await (const _ of s.generate([{ role: 'user', content: 'hi' }])) { /* drain */ }
    assert.equal(log.gens.at(-1).args.enable_thinking, false);
    assert.equal('assistantPrefix' in log.gens.at(-1).opts, false);
    s.reset();
    assert.equal(s.engine.resets, 1);
    assert.equal(s.speculative.status, 'off');
  } finally { localmindBackend.source = saved; }
});

test('localmind: Bonsai DFlash attaches after load from a cached drafter; a second load does not download it again', async () => {
  const saved = localmindBackend.source;
  localmindBackend.source = fakes;
  const env = browserish(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]));
  try {
    const { created } = await import('./fixtures/fake-engines/ternary_bonsai_2_dflash.js');
    const wait = async (s) => { for (let i = 0; i < 100 && s.speculative.status === 'loading'; i++) await new Promise((r) => setTimeout(r, 5)); };
    const s = await localmindBackend.load(bonsai, { fetch: env.fetch, speculative: { dflash: { block: 5 } } });
    await wait(s);
    assert.equal(s.speculative.status, 'ready', s.speculative.error);
    assert.deepEqual(s.speculative.stats, { cycles: 3, tokens: 9, accepted: 6 });
    assert.notEqual(s.engine.model.dflashRunner, null);
    assert.deepEqual(created.at(-1), { block: 5, head: [1, 2, 3, 4] });
    const firstRequests = env.requests();
    const s2 = await localmindBackend.load(bonsai, { fetch: env.fetch, speculative: { dflash: {} } });
    await wait(s2);
    assert.equal(s2.speculative.status, 'ready');
    assert.equal(env.requests(), firstRequests, 'the drafter came from Cache Storage');
    s2.dispose();
    assert.equal(s2.engine.model.dflashRunner, null);
  } finally { localmindBackend.source = saved; delete globalThis.caches; }
});

test('localmind: Gemma 4 with ple streams the table from OPFS and reports pleMode; without it the table stays resident', async () => {
  const saved = localmindBackend.source;
  localmindBackend.source = fakes;
  try {
    const { made } = await import('./fixtures/fake-engines/ple-opfs.js');
    const withPle = await localmindBackend.load(gemma, { ple: true });
    assert.deepEqual(made, ['google/gemma-4-E2B-it-qat-mobile-transformers']);
    assert.equal(withPle.pleMode, 'opfs');
    assert.equal((await localmindBackend.load(gemma, {})).pleMode, 'resident');
  } finally { localmindBackend.source = saved; }
});

test('localmind: Bonsai MTP maps the head\'s absolute offsets onto the downloaded ranges', async () => {
  const saved = localmindBackend.source;
  localmindBackend.source = fakes;
  const env = browserish(Uint8Array.from({ length: 20 }, (_, i) => i));
  try {
    const { created } = await import('./fixtures/fake-engines/ternary_bonsai_2_mtp.js');
    const s = await localmindBackend.load(bonsai, { fetch: env.fetch, speculative: { mtp: { depth: 2, adaptive: true } } });
    for (let i = 0; i < 100 && s.speculative.status === 'loading'; i++) await new Promise((r) => setTimeout(r, 5));
    assert.equal(s.speculative.status, 'ready', s.speculative.error);
    assert.deepEqual(created.at(-1), { depth: 2, adaptive: true, a: [3], b: [12, 13] });
    assert.equal(s.speculative.stats.drafted, 2);
  } finally { localmindBackend.source = saved; delete globalThis.caches; }
});

test('load(): session.speculative is read live, so its status moves from loading to ready after load returns', async () => {
  const { load } = await import('../index.js');
  const saved = localmindBackend.source;
  localmindBackend.source = fakes;
  const env = browserish(new Uint8Array([9, 9, 9, 9]));
  try {
    const caps = { webgpu: true, f16: true, subgroups: true, opfs: true, memory64: true, jspi: true };
    const s = await load(bonsai.repo, { model: bonsai, caps, backends: [localmindBackend], budget: 7e9, fetch: env.fetch, speculative: { dflash: {} } });
    assert.equal(s.speculative.status, 'loading');
    for (let i = 0; i < 100 && s.speculative.status === 'loading'; i++) await new Promise((r) => setTimeout(r, 5));
    assert.equal(s.speculative.status, 'ready');
    assert.equal(typeof s.engine.reset, 'function');
  } finally { localmindBackend.source = saved; delete globalThis.caches; }
});
