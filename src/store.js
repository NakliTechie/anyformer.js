// What anyformer has downloaded into this browser, and removing it. Each back end keeps its own store; this module
// reads them all through one shape: { backend, repo, bytes }.
//   transformers.js: the Cache API cache 'transformers-cache', one entry per file URL (…/<repo>/resolve/<rev>/<file>).
//     kohra's tokenizer lands here too; its ONNX graph is in the browser's HTTP cache, which a page cannot list.
//   diskformer: OPFS folder 'localmind-ssd' (the engines' default root), one folder per GGUF, named storeKey(file).

import { DISKFORMER_MODELS } from './backends/diskformer.js';

const TJS_CACHE = 'transformers-cache';
export const DISKFORMER_ROOT = 'localmind-ssd';
// diskformer's storeKey (src/ingest.js): the GGUF name without .gguf, lowercased, other characters to '-'.
const storeKey = (file) => file.replace(/\.gguf$/i, '').toLowerCase().replace(/[^a-z0-9._-]+/g, '-');

const repoOf = (url) => /^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\//.exec(url)?.[1] ?? null;

async function cacheEntries() {
  if (typeof caches === 'undefined' || !(await caches.has(TJS_CACHE))) return [];
  const cache = await caches.open(TJS_CACHE);
  const out = [];
  for (const req of await cache.keys()) {
    const repo = repoOf(req.url);
    if (!repo) continue;
    // Content-Length is in the stored headers; reading the body to size it would load every file into memory.
    const res = await cache.match(req);
    out.push({ repo, url: req.url, bytes: Number(res?.headers.get('content-length')) || 0, cache });
  }
  return out;
}

async function opfsDir(path, create = false) {
  if (!globalThis.navigator?.storage?.getDirectory) return null;
  let dir = await navigator.storage.getDirectory();
  for (const name of path) {
    try { dir = await dir.getDirectoryHandle(name, { create }); } catch { return null; }
  }
  return dir;
}

async function dirBytes(dir) {
  let n = 0;
  for await (const [, h] of dir.entries()) n += h.kind === 'file' ? (await h.getFile()).size : await dirBytes(h);
  return n;
}

async function diskformerEntries() {
  const root = await opfsDir([DISKFORMER_ROOT]);
  if (!root) return [];
  const out = [];
  for (const m of DISKFORMER_MODELS) {
    const dir = await opfsDir([DISKFORMER_ROOT, storeKey(m.file)]);
    if (dir) out.push({ repo: m.repo, key: storeKey(m.file), bytes: await dirBytes(dir) });
  }
  return out;
}

// Every stored model, one row per back end and repo. A store of 0 bytes (an interrupted first write) is left out.
export async function stored() {
  const rows = new Map();
  for (const e of await cacheEntries()) {
    const k = `transformers ${e.repo}`;
    const r = rows.get(k) || { backend: 'transformers', repo: e.repo, bytes: 0, files: 0 };
    r.bytes += e.bytes; r.files++;
    rows.set(k, r);
  }
  for (const e of await diskformerEntries()) rows.set(`diskformer ${e.repo}`, { backend: 'diskformer', repo: e.repo, bytes: e.bytes });
  return [...rows.values()].filter((r) => r.bytes > 0);
}

// Remove a repo from every store that holds it. Returns the bytes freed. The next load downloads it again.
export async function remove(repo) {
  let bytes = 0;
  for (const e of await cacheEntries()) if (e.repo === repo) { bytes += e.bytes; await e.cache.delete(e.url); }
  const m = DISKFORMER_MODELS.find((x) => x.repo === repo);
  const root = m && (await opfsDir([DISKFORMER_ROOT]));
  if (root) {
    const dir = await opfsDir([DISKFORMER_ROOT, storeKey(m.file)]);
    if (dir) { bytes += await dirBytes(dir); await root.removeEntry(storeKey(m.file), { recursive: true }); }
  }
  return { repo, bytes };
}
