// Speculative decoding for Ternary Bonsai 2 27B: DFlash 2 (a 5-layer drafter) or MTP (one extra decoder layer that
// drafts ahead). Both are greedy-identical to the engine's own decode; they only change how many tokens one verify
// pass accepts. The runners live in diskformer (engines/ternary_bonsai_2_dflash.js and ternary_bonsai_2_mtp.js) and
// attach to the engine through its specDecodeRunner() seam (engine.model.dflashRunner).
//
// Their weights are separate downloads, kept in Cache Storage under the names and keys LocalMind uses, so a browser
// that already holds them from LocalMind reuses them:
//   'dflash-drafter-v1'  the drafter GGUF (1.14 GB), keyed by DRAFTER_URL
//   'mtp-head-v1'        the MTP head's byte ranges (451 MB) of a grafted GGUF, keyed by MTP_HEAD.url

// LocalMind's drafter: naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2 (index.html DRAFTER_URL, main febbc93).
export const DRAFTER_URL = 'https://huggingface.co/naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2/resolve/main/Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf';
export const DRAFTER_CACHE = 'dflash-drafter-v1';
export const MTP_CACHE = 'mtp-head-v1';

// One file's byte ranges ([[start, length], …], or the whole file) as a Blob, from Cache Storage or downloaded in 32 MB
// pieces, 8 in flight (Hugging Face's CDN throttles each connection), then stored. A Blob keeps a 1 GB file on disk
// rather than in the JS heap.
export async function cachedBlob({ cacheName, url, ranges = null, fileSize = null, fetch = globalThis.fetch, onProgress, signal }) {
  const cache = await caches.open(cacheName);
  const want = ranges ? ranges.reduce((n, [, len]) => n + len, 0) : null;
  const hit = await cache.match(url);
  if (hit) { const b = await hit.blob(); if (want === null || b.size === want) return b; }
  if (!ranges) {
    const probe = await fetch(url, { headers: { Range: 'bytes=0-0' }, signal });
    if (probe.status !== 206) throw new Error(`${url}: HTTP ${probe.status}; the server must support Range`);
    fileSize = Number((probe.headers.get('content-range') || '').split('/')[1]) || 0;
    try { await probe.body?.cancel(); } catch { /* nothing to cancel */ }
    if (!fileSize) throw new Error(`${url}: unknown size`);
    ranges = [[0, fileSize]];
  }
  const PIECE = 32 << 20, STREAMS = 8;
  const jobs = [];
  for (const [start, len] of ranges) for (let o = 0; o < len; o += PIECE) jobs.push([start + o, Math.min(PIECE, len - o)]);
  const total = jobs.reduce((n, [, len]) => n + len, 0);
  const pieces = new Array(jobs.length);
  let next = 0, loaded = 0;
  const worker = async () => {
    for (;;) {
      const i = next++;
      if (i >= jobs.length) return;
      const [start, len] = jobs[i];
      const r = await fetch(url, { headers: { Range: `bytes=${start}-${start + len - 1}` }, signal });
      if (r.status !== 206) throw new Error(`${url}: HTTP ${r.status} on bytes ${start}+${len}`);
      if (fileSize && Number((r.headers.get('content-range') || '').split('/')[1]) !== fileSize) throw new Error(`${url}: the file is not ${fileSize} bytes`);
      const buf = new Uint8Array(await r.arrayBuffer());
      if (buf.byteLength !== len) throw new Error(`${url}: short read on bytes ${start}+${len}`);
      pieces[i] = buf;
      loaded += len;
      onProgress?.({ loaded, total });
    }
  };
  await Promise.all(Array.from({ length: STREAMS }, worker));
  const blob = new Blob(pieces, { type: 'application/octet-stream' });
  // A failed write only costs a download next time.
  await cache.put(url, new Response(blob, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(blob.size) } })).catch(() => {});
  return blob;
}

const blobSource = (blob) => async (off, len) => new Uint8Array(await blob.slice(off, off + len).arrayBuffer());

// Attaches a runner to a loaded Ternary Bonsai 2 engine. spec: { dflash: { block = 5 } } or
// { mtp: { depth = 2, adaptive = false } }. engines: the diskformer engines/ URL. Returns the runner; its .stats
// counts cycles, tokens and accepted drafts (MTP adds drafted and plainTokens).
export async function attachSpeculative(engine, spec, { engines, fetch, onProgress, signal } = {}) {
  if (spec.dflash) {
    const { DFlashRunner } = await import(`${engines}/ternary_bonsai_2_dflash.js`);
    const blob = await cachedBlob({ cacheName: DRAFTER_CACHE, url: DRAFTER_URL, fetch, signal,
      onProgress: (p) => onProgress?.({ status: 'download', kind: 'dflash', ...p }) });
    signal?.throwIfAborted();
    const runner = await DFlashRunner.create(engine, { drafter: blobSource(blob), block: spec.dflash.block ?? 5,
      onProgress: (p) => onProgress?.({ status: 'load', kind: 'dflash', loaded: p.done, total: p.total }) });
    engine.model.dflashRunner = runner;
    return runner;
  }
  if (spec.mtp) {
    const { MTPRunner, MTP_HEAD } = await import(`${engines}/ternary_bonsai_2_mtp.js`);
    const blob = await cachedBlob({ cacheName: MTP_CACHE, url: MTP_HEAD.url, ranges: MTP_HEAD.ranges, fileSize: MTP_HEAD.fileSize, fetch, signal,
      onProgress: (p) => onProgress?.({ status: 'download', kind: 'mtp', ...p }) });
    signal?.throwIfAborted();
    // Absolute file offsets → offsets in the blob, which holds the ranges back to back.
    const at = [];
    let acc = 0;
    for (const [start, len] of MTP_HEAD.ranges) { at.push([start, len, acc]); acc += len; }
    const head = async (off, len) => {
      const r = at.find(([s, l]) => off >= s && off + len <= s + l);
      if (!r) throw new Error(`MTP head: bytes ${off}+${len} are outside the downloaded ranges`);
      return blobSource(blob)(r[2] + off - r[0], len);
    };
    const runner = await MTPRunner.create(engine, { head, depth: spec.mtp.depth ?? 2, adaptive: !!spec.mtp.adaptive,
      onProgress: (p) => onProgress?.({ status: 'load', kind: 'mtp', loaded: p.done, total: p.total }) });
    engine.model.dflashRunner = runner;
    return runner;
  }
  throw new Error('speculative: pass { dflash: { block } } or { mtp: { depth, adaptive } }');
}
