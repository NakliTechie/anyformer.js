// A GGUF too big for one ArrayBuffer (wllama reads files of at most 2 GB) becomes shards in llama.cpp's gguf-split
// layout, the same files `llama-gguf-split --split-max-size` writes: shard 1 carries every metadata key, each shard
// carries split.no / split.count / split.tensors.count and its own tensor table, and each shard's data section holds
// its tensors with offsets relative to it. Tensor bytes are copied, never decoded, so any quant type works.

const MAGIC = 0x46554747;                   // 'GGUF'
const DEFAULT_ALIGNMENT = 32;
// GGUF metadata value types → byte size (fixed-size types only; 8 = string, 9 = array).
const SIZE = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

class Short extends Error {}                // the header runs past the bytes read so far

// Parse a GGUF header from its first bytes. Returns the raw metadata bytes (copied verbatim into shard 1), the tensor
// table, the alignment and where the data section starts. Throws Short when `u8` stops inside the header.
export function parseHeader(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let p = 0;
  const need = (n) => { if (p + n > u8.byteLength) throw new Short(); };
  const u32 = () => { need(4); const v = dv.getUint32(p, true); p += 4; return v; };
  const u64 = () => { need(8); const v = dv.getBigUint64(p, true); p += 8; return Number(v); };
  const str = () => { const n = u64(); need(n); const s = new TextDecoder().decode(u8.subarray(p, p + n)); p += n; return s; };
  const skip = (type) => {
    if (type === 8) { const n = u64(); need(n); p += n; return; }
    if (type === 9) { const t = u32(), n = u64(); if (t in SIZE) { need(n * SIZE[t]); p += n * SIZE[t]; } else for (let i = 0; i < n; i++) skip(t); return; }
    need(SIZE[type]); p += SIZE[type];
  };
  if (u32() !== MAGIC) throw new Error('not a GGUF file');
  const version = u32();
  if (version < 2) throw new Error(`GGUF version ${version} is not supported`);
  const tensorCount = u64(), kvCount = u64();
  const kvStart = p;
  let alignment = DEFAULT_ALIGNMENT;
  for (let i = 0; i < kvCount; i++) {
    const key = str(), type = u32();
    if (key === 'general.alignment' && type === 4) { need(4); alignment = dv.getUint32(p, true); }
    if (key.startsWith('split.')) throw new Error(`already a shard (${key})`);
    skip(type);
  }
  const kvBytes = u8.slice(kvStart, p);
  const tensors = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = str(), nDims = u32(), dims = [];
    for (let d = 0; d < nDims; d++) dims.push(u64());
    tensors.push({ name, dims, type: u32(), offset: u64() });
  }
  const dataStart = Math.ceil(p / alignment) * alignment;
  return { version, kvCount, kvBytes, tensors, alignment, dataStart };
}

// Read the header through `read(start, end)` (bytes [start, end)), widening the window until it parses.
export async function readHeader(read, fileSize) {
  for (let n = 8 << 20; ; n *= 2) {
    try { return parseHeader(await read(0, Math.min(n, fileSize))); } catch (e) {
      if (!(e instanceof Short) || n >= fileSize) throw e;
    }
  }
}

// Group tensors (in data order) into shards of at most maxBytes of data each. A tensor's bytes run to the next
// tensor's offset, so the original alignment padding travels with it and every copied offset stays aligned.
export function planShards(header, fileSize, maxBytes) {
  const t = [...header.tensors].sort((a, b) => a.offset - b.offset);
  const dataSize = fileSize - header.dataStart;
  const sized = t.map((x, i) => ({ ...x, bytes: (i + 1 < t.length ? t[i + 1].offset : dataSize) - x.offset }));
  const shards = [];
  let cur = null;
  for (const x of sized) {
    if (!cur || (cur.bytes + x.bytes > maxBytes && cur.tensors.length)) shards.push(cur = { tensors: [], bytes: 0 });
    cur.tensors.push({ ...x, newOffset: cur.bytes });
    cur.bytes += x.bytes;
  }
  for (const s of shards) { s.start = header.dataStart + s.tensors[0].offset; s.end = s.start + s.bytes; }
  return shards;
}

// The header bytes of shard i of n: magic, version, counts, metadata (all of it in shard 0, split keys only in the
// rest), this shard's tensor table, padding to the alignment.
export function shardHeader(header, shards, i) {
  const out = [];
  const enc = new TextEncoder();
  const u32 = (v) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); out.push(b); };
  const u64 = (v) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); out.push(b); };
  const u16 = (v) => { const b = new Uint8Array(2); new DataView(b.buffer).setUint16(0, v, true); out.push(b); };
  const i32 = (v) => { const b = new Uint8Array(4); new DataView(b.buffer).setInt32(0, v, true); out.push(b); };
  const str = (s) => { const b = enc.encode(s); u64(b.length); out.push(b); };
  const shard = shards[i];
  u32(MAGIC); u32(header.version);
  u64(shard.tensors.length);
  u64((i === 0 ? header.kvCount : 0) + 3);
  if (i === 0) out.push(header.kvBytes);
  str('split.no'); u32(2); u16(i);
  str('split.count'); u32(2); u16(shards.length);
  str('split.tensors.count'); u32(5); i32(header.tensors.length);
  for (const t of shard.tensors) {
    str(t.name); u32(t.dims.length);
    for (const d of t.dims) u64(d);
    u32(t.type); u64(t.newOffset);
  }
  // Shards after the first carry no general.alignment, so llama.cpp reads them at the default 32.
  const align = i === 0 ? header.alignment : DEFAULT_ALIGNMENT;
  const len = out.reduce((n, b) => n + b.length, 0);
  out.push(new Uint8Array(Math.ceil(len / align) * align - len));
  const all = new Uint8Array(out.reduce((n, b) => n + b.length, 0));
  let p = 0;
  for (const b of out) { all.set(b, p); p += b.length; }
  return all;
}

export const shardName = (base, i, n) => `${base}-${String(i + 1).padStart(5, '0')}-of-${String(n).padStart(5, '0')}.gguf`;

// Split the GGUF at `url` into shards of at most maxBytes in OPFS (anyformer-gguf/<key>/), reading it with HTTP range
// requests, and return the shards as Files in order. Crash-safe and resumable: a shard of the expected size is kept,
// anything else is rewritten, and done.json is written last; with done.json present nothing is fetched.
export const SPLIT_ROOT = 'anyformer-gguf';
export async function splitToOpfs({ url, size, key, repo, file, maxBytes = 512 << 20, onProgress, signal, fetch: f = globalThis.fetch }) {
  const root = await (await navigator.storage.getDirectory()).getDirectoryHandle(SPLIT_ROOT, { create: true });
  const dir = await root.getDirectoryHandle(key, { create: true });
  const files = async (names) => Promise.all(names.map(async (n) => (await dir.getFileHandle(n)).getFile()));
  // meta.json first, so an interrupted split still names its repo (src/store.js lists and removes it).
  const meta = await (await dir.getFileHandle('meta.json', { create: true })).createWritable();
  await meta.write(JSON.stringify({ repo, file, url, size }));
  await meta.close();
  try {
    const done = JSON.parse(await (await (await dir.getFileHandle('done.json')).getFile()).text());
    if (done.url === url && done.size === size) return files(done.shards);
  } catch { /* not split yet, or interrupted */ }
  const range = async (s, e) => {
    const r = await f(url, { headers: { Range: `bytes=${s}-${e - 1}` }, signal });
    if (r.status !== 206 && !(r.ok && s === 0 && e >= size)) throw new Error(`${url}: range request returned ${r.status}`);
    return r;
  };
  const header = await readHeader(async (s, e) => new Uint8Array(await (await range(s, e)).arrayBuffer()), size);
  const shards = planShards(header, size, maxBytes);
  const base = file.replace(/\.gguf$/i, '');
  const names = shards.map((_, i) => shardName(base, i, shards.length));
  let copied = 0;
  for (const [i, s] of shards.entries()) {
    const head = shardHeader(header, shards, i);
    const expect = head.length + s.bytes;
    const h = await dir.getFileHandle(names[i], { create: true });
    if ((await h.getFile()).size === expect) { copied += s.bytes; continue; }
    const w = await h.createWritable();           // commits atomically on close(); an abort leaves the old file
    try {
      await w.write(head);
      const body = (await range(s.start, s.end)).body.getReader();
      for (;;) {
        const { done, value } = await body.read();
        if (done) break;
        await w.write(value);
        copied += value.length;
        onProgress?.({ status: 'split', file: names[i], loaded: copied, total: size - header.dataStart });
      }
      await w.close();
    } catch (e) { await w.abort().catch(() => {}); throw e; }
  }
  const marker = await (await dir.getFileHandle('done.json', { create: true })).createWritable();
  await marker.write(JSON.stringify({ url, size, repo, file, shards: names }));
  await marker.close();
  return files(names);
}
