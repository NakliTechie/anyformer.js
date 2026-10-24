// Where each back end's runtime loads from: exact versions and commits, so a given anyformer behaves the same tomorrow.
// kohra and diskformer move fast on their main branches; `node scripts/pin-sources.mjs` moves these pins
// to each repo's current main, and the browser tests then prove the new pins (test/browser/README.md).
// transformers.js and wllama move only by hand, in step with LocalMind's pins (plan/history.md, 2026-10-08).

export const PINS = {
  transformers: '4.3.1',
  wllama: '3.4.1',      // LocalMind's pin; its stream-drain workaround (src/backends/wllama.js) is for this version
  kohra: 'f678fac7342c79b58b640540b8b3a7d078b7ab1a',
  // v0.6.7: every engine (the disk-tier MoE engines, Ling 3.0 Flash, and LocalMind's whole-model engines LFM2.5,
  // Gemma 4 E2B/E4B and Ternary Bonsai 2) under engines/, and the llama.cpp path under ggml/, which loads split GGUFs.
  // The hand engines' ingest streams the GGUF header with progress and resumes a dropped range.
  diskformer: '663b2b01cfbed95ae8d319ce845d8aea759d60cf',
};

const gh = (repo, sha, path) => `https://cdn.jsdelivr.net/gh/NakliTechie/${repo}@${sha}/${path}`;

export const SOURCES = {
  transformers: `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${PINS.transformers}/+esm`,
  wllama: `https://cdn.jsdelivr.net/npm/@wllama/wllama@${PINS.wllama}/esm`,
  kohra: gh('kohra', PINS.kohra, 'kohra.js'),
  diskformer: gh('diskformer.js', PINS.diskformer, 'engines'),
  diskformerLlama: gh('diskformer.js', PINS.diskformer, 'ggml/llama.js'),
};
