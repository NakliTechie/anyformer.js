import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseHeader, readHeader, planShards, shardHeader, shardName } from '../src/gguf-split.js';

// A small GGUF: three metadata keys (a string, an array of strings, alignment 32) and five tensors of different
// sizes, each tensor's bytes a distinct pattern so a misplaced copy shows.
function makeGguf() {
  const parts = [];
  const enc = new TextEncoder();
  const num = (bytes, set) => { const b = new Uint8Array(bytes); set(new DataView(b.buffer)); parts.push(b); };
  const u32 = (v) => num(4, (d) => d.setUint32(0, v, true));
  const u64 = (v) => num(8, (d) => d.setBigUint64(0, BigInt(v), true));
  const str = (s) => { const b = enc.encode(s); u64(b.length); parts.push(b); };
  const sizes = [100, 3000, 64, 5000, 777];
  const aligned = (n) => Math.ceil(n / 32) * 32;
  u32(0x46554747); u32(3); u64(sizes.length); u64(3);
  str('general.architecture'); u32(8); str('llama');
  str('tokenizer.ggml.tokens'); u32(9); u32(8); u64(3); str('a'); str('bb'); str('ccc');
  str('general.alignment'); u32(4); u32(32);
  let off = 0;
  sizes.forEach((n, i) => { str(`blk.${i}.weight`); u32(1); u64(n); u32(0); u64(off); off += aligned(n); });
  const headerLen = parts.reduce((n, b) => n + b.length, 0);
  parts.push(new Uint8Array(aligned(headerLen) - headerLen));
  sizes.forEach((n, i) => { const b = new Uint8Array(aligned(n)); b.fill(i + 1, 0, n); parts.push(b); });
  const all = new Uint8Array(parts.reduce((n, b) => n + b.length, 0));
  let p = 0; for (const b of parts) { all.set(b, p); p += b.length; }
  return { bytes: all, sizes };
}

test('parseHeader reads the metadata, tensor table, alignment and data start', () => {
  const { bytes, sizes } = makeGguf();
  const h = parseHeader(bytes);
  assert.equal(h.version, 3);
  assert.equal(h.kvCount, 3);
  assert.equal(h.alignment, 32);
  assert.equal(h.tensors.length, sizes.length);
  assert.equal(h.dataStart % 32, 0);
  assert.equal(bytes[h.dataStart + h.tensors[1].offset], 2);
});

test('readHeader widens its window until the header parses', async () => {
  const { bytes } = makeGguf();
  let calls = 0;
  const h = await readHeader(async (s, e) => { calls++; return bytes.slice(s, e); }, bytes.length);
  assert.equal(h.tensors.length, 5);
  assert.equal(calls, 1);
});

test('shards: every tensor lands once, bytes identical, split keys present, metadata only in shard 1', () => {
  const { bytes, sizes } = makeGguf();
  const h = parseHeader(bytes);
  const shards = planShards(h, bytes.length, 4000);
  assert.ok(shards.length >= 3);
  const seen = new Map();
  shards.forEach((s, i) => {
    const file = new Uint8Array([...shardHeader(h, shards, i), ...bytes.slice(s.start, s.end)]);
    const sh = parseHeader2(file);
    assert.equal(sh.split.no, i);
    assert.equal(sh.split.count, shards.length);
    assert.equal(sh.split.tensorsCount, sizes.length);
    assert.equal(sh.keys.includes('general.architecture'), i === 0);
    for (const t of sh.tensors) {
      const k = Number(t.name.split('.')[1]);
      const at = sh.dataStart + t.offset;
      assert.ok(file.slice(at, at + sizes[k]).every((b) => b === k + 1), t.name);
      seen.set(t.name, (seen.get(t.name) || 0) + 1);
    }
  });
  assert.equal(seen.size, sizes.length);
  assert.ok([...seen.values()].every((n) => n === 1));
});

test('a shard is refused as input; names follow llama.cpp\'s pattern', () => {
  const { bytes } = makeGguf();
  const h = parseHeader(bytes);
  const shards = planShards(h, bytes.length, 4000);
  const file = new Uint8Array([...shardHeader(h, shards, 0), ...bytes.slice(shards[0].start, shards[0].end)]);
  assert.throws(() => parseHeader(file), /already a shard/);
  assert.equal(shardName('m', 0, 3), 'm-00001-of-00003.gguf');
});

// A reader that also returns the split keys and key names (parseHeader refuses shards by design).
function parseHeader2(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let p = 8;
  const u32 = () => { const v = dv.getUint32(p, true); p += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(p, true)); p += 8; return v; };
  const str = () => { const n = u64(); const s = new TextDecoder().decode(u8.subarray(p, p + n)); p += n; return s; };
  const nt = u64(), nk = u64(), keys = [], split = {};
  let align = 32;
  for (let i = 0; i < nk; i++) {
    const k = str(), t = u32(); keys.push(k);
    if (t === 2) { const v = dv.getUint16(p, true); p += 2; if (k === 'split.no') split.no = v; if (k === 'split.count') split.count = v; }
    else if (t === 5) { split.tensorsCount = dv.getInt32(p, true); p += 4; }
    else if (t === 4) { const v = u32(); if (k === 'general.alignment') align = v; }
    else if (t === 8) str();
    else if (t === 9) { u32(); const n = u64(); for (let j = 0; j < n; j++) str(); }
  }
  const tensors = [];
  for (let i = 0; i < nt; i++) { const name = str(), nd = u32(); for (let d = 0; d < nd; d++) u64(); u32(); tensors.push({ name, offset: u64() }); }
  return { keys, split, tensors, dataStart: Math.ceil(p / align) * align };
}
