// What anyformer's back ends have downloaded into this browser, and removing it. Each back end keeps its own store;
// this module reads them all into one shape: { backend, repo, bytes }. Modelled on LocalMind's
// inventoryModelCaches() (index.html), which covers the same engines' stores.
//   Cache Storage   'transformers-cache' (transformers.js; kohra's tokenizer too); 'gguf-v1', 'gguf-v1-headers' and
//                   'bonsai-pipeline-v1' (LocalMind's engines; 'gguf-v1' holds the LFM2
//                   engine): one entry per file URL (…/<repo>/resolve/<rev>/<file>). kohra's ONNX graph sits in the
//                   browser's HTTP cache, which a page cannot list.
//   IndexedDB       'gguf-cache-v1' (Bonsai 2) and 'safetensors-cache-v1' (Gemma 4 QAT-mobile): a store keyed by URL
//                   with { size } and a chunk store keyed [url, start, end]; 'webgpu-prefix-snapshots' (Bonsai 2's
//                   system-prompt snapshot).
//   OPFS            'localmind-ssd/<key>/' (diskformer's MoE stores, Gemma 4's PLE table) with manifest.json naming the
//                   source; 'cache/' (wllama: '__metadata__<name>' names each file's URL); 'anyformer-gguf/<key>/'
//                   (GGUFs over 2 GB as shards, meta.json or done.json naming the repo; src/gguf-split.js);
//                   'gguf/<name>' (diskformer-llama: llama.cpp's GGUFs, with <name>.part and .part.json while downloading;
//                   anyformer-llama.json maps each name to its repo).

import { DISKFORMER_MODELS } from './backends/diskformer.js';
import { SPLIT_ROOT } from './gguf-split.js';
import { readLlamaIndex, LLAMA_INDEX } from './backends/diskformer-llama.js';

const CACHE_BACKEND = { 'transformers-cache': 'transformers', 'gguf-v1': 'localmind', 'gguf-v1-headers': 'localmind', 'bonsai-pipeline-v1': 'localmind' };
const IDB_STORES = ['gguf-cache-v1', 'safetensors-cache-v1'];
const BONSAI2 = 'prism-ml/Ternary-Bonsai-2-27B-gguf';

const repoOf = (url) => /^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\//.exec(url || '')?.[1] ?? null;
const req = (r) => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
const openDb = (name) => new Promise((res, rej) => {
  const r = indexedDB.open(name);
  r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
  r.onupgradeneeded = () => { r.transaction.abort(); rej(new Error(`${name} does not exist`)); };
});

async function opfsDir(path) {
  if (!globalThis.navigator?.storage?.getDirectory) return null;
  let dir = await navigator.storage.getDirectory();
  for (const name of path) {
    try { dir = await dir.getDirectoryHandle(name); } catch { return null; }
  }
  return dir;
}
async function dirBytes(dir) {
  let n = 0;
  for await (const [, h] of dir.entries()) n += h.kind === 'file' ? (await h.getFile()).size : await dirBytes(h);
  return n;
}
const readJson = async (dir, name) => { try { return JSON.parse(await (await (await dir.getFileHandle(name)).getFile()).text()); } catch { return null; } };

// Every stored piece: { backend, repo, bytes, remove }.
async function inventory() {
  const out = [];
  const add = (backend, repo, bytes, remove) => { if (repo) out.push({ backend, repo, bytes: bytes || 0, remove }); };

  if (typeof caches !== 'undefined') for (const name of await caches.keys()) {
    const cache = await caches.open(name);
    for (const r of await cache.keys()) {
      // Content-Length is in the stored headers; reading the body to size it would load every file into memory.
      const res = await cache.match(r);
      add(CACHE_BACKEND[name] || name, repoOf(r.url), Number(res?.headers.get('content-length')) || 0, () => cache.delete(r));
    }
  }

  let dbs = [];
  try { dbs = (await indexedDB.databases()).map((d) => d.name); } catch { /* no IndexedDB */ }
  for (const dbName of IDB_STORES.filter((n) => dbs.includes(n))) {
    const db = await openDb(dbName);
    let meta = null, chunks = null;
    for (const s of db.objectStoreNames) {
      const [k] = await req(db.transaction(s, 'readonly').objectStore(s).getAllKeys(null, 1));
      if (Array.isArray(k)) chunks = s; else if (typeof k === 'string') meta = s;
    }
    if (meta) for (const url of await req(db.transaction(meta, 'readonly').objectStore(meta).getAllKeys())) {
      const m = await req(db.transaction(meta, 'readonly').objectStore(meta).get(url));
      add('localmind', repoOf(url), m?.size, async () => {
        if (chunks) await req(db.transaction(chunks, 'readwrite').objectStore(chunks).delete(IDBKeyRange.bound([url], [url, [], []])));
        await req(db.transaction(meta, 'readwrite').objectStore(meta).delete(url));
      });
    }
  }
  if (dbs.includes('webgpu-prefix-snapshots')) {
    const db = await openDb('webgpu-prefix-snapshots');
    const stores = [...db.objectStoreNames];
    const slot = stores.includes('meta') ? await req(db.transaction('meta', 'readonly').objectStore('meta').get('prefix-snapshot-slot')) : null;
    if (slot) add('localmind', BONSAI2, (slot.sizes || []).reduce((n, s) => n + s, 0), async () => {
      for (const s of stores) await req(db.transaction(s, 'readwrite').objectStore(s).clear());
    });
  }

  const ssd = await opfsDir(['localmind-ssd']);
  if (ssd) for await (const [key, h] of ssd.entries()) {
    if (h.kind !== 'directory') continue;
    const src = (await readJson(h, 'manifest.json'))?.source;
    const repo = src?.repo || repoOf(src?.url) || DISKFORMER_MODELS.find((m) => m.file.toLowerCase().startsWith(key))?.repo;
    add(DISKFORMER_MODELS.some((m) => m.repo === repo) ? 'diskformer' : 'localmind', repo, await dirBytes(h), () => ssd.removeEntry(key, { recursive: true }));
  }
  const wcache = await opfsDir(['cache']);
  if (wcache) for await (const [name, h] of wcache.entries()) {
    if (h.kind !== 'file' || name.startsWith('__metadata__')) continue;
    add('wllama', repoOf((await readJson(wcache, `__metadata__${name}`))?.originalURL), (await h.getFile()).size, async () => {
      await wcache.removeEntry(name);
      await wcache.removeEntry(`__metadata__${name}`).catch(() => {});
    });
  }
  const split = await opfsDir([SPLIT_ROOT]);
  if (split) for await (const [key, h] of split.entries()) {
    if (h.kind !== 'directory') continue;
    const repo = (await readJson(h, 'meta.json'))?.repo || (await readJson(h, 'done.json'))?.repo;
    add('wllama', repo, await dirBytes(h), () => split.removeEntry(key, { recursive: true }));
  }
  const llama = await opfsDir(['gguf']);
  if (llama) {
    const index = await readLlamaIndex();
    for await (const [name, h] of llama.entries()) {
      if (h.kind !== 'file' || name.endsWith('.part.json') || name.endsWith('.crswap')) continue;
      const base = name.replace(/\.part$/, '');
      add('diskformer-llama', index[base], (await h.getFile()).size, async () => {
        for (const n of [base, `${base}.part`, `${base}.part.json`]) await llama.removeEntry(n).catch(() => {});
        const rest = await readLlamaIndex();
        delete rest[base];
        const w = await (await (await navigator.storage.getDirectory()).getFileHandle(LLAMA_INDEX, { create: true })).createWritable();
        await w.write(JSON.stringify(rest));
        await w.close();
      });
    }
  }
  return out;
}

// Every stored model, one row per back end and repo. A store of 0 bytes (an interrupted first write) is left out.
export async function stored() {
  const rows = new Map();
  for (const e of await inventory()) {
    const k = `${e.backend} ${e.repo}`;
    const r = rows.get(k) || { backend: e.backend, repo: e.repo, bytes: 0 };
    r.bytes += e.bytes;
    rows.set(k, r);
  }
  return [...rows.values()].filter((r) => r.bytes > 0);
}

// Remove a repo from every store that holds it. Returns the bytes freed. The next load downloads it again.
export async function remove(repo) {
  let bytes = 0;
  for (const e of await inventory()) if (e.repo === repo) { bytes += e.bytes; await e.remove(); }
  return { repo, bytes };
}
