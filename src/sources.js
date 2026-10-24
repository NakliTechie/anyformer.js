// Where each back end's runtime loads from: exact versions and commits, so a given anyformer behaves the same tomorrow.
// kohra and diskformer move fast on their main branches; `node scripts/pin-sources.mjs` moves these pins
// to each repo's current main, and the browser tests then prove the new pins (test/browser/README.md).
// transformers.js and wllama move only by hand, in step with LocalMind's pins (plan/history.md, 2026-10-08).

export const PINS = {
  transformers: '4.3.1',
  wllama: '3.4.1',      // LocalMind's pin; its stream-drain workaround (src/backends/wllama.js) is for this version
  // kohra main after v0.2.1: seeded block-KV graphs (Nemotron-Labs-Diffusion-3B), chatFormat, split external data.
  kohra: '14e430c5b70a0da80086432769be0ee0f40f3882',
  // v0.6.8: every engine (the disk-tier MoE engines, Ling 3.0 Flash, and LocalMind's whole-model engines LFM2.5,
  // Gemma 4 E2B/E4B and Ternary Bonsai 2) under engines/, and the llama.cpp path under ggml/, which loads split GGUFs
  // and drafters, and reads past 2 GiB (Nemotron 3.5 Bolt E4B needs v0.6.8 or later). The hand engines' ingest
  // streams the GGUF header with progress and resumes a dropped range.
  diskformer: '3aad103e47cac76c1471e92a438edfe1c80dd86e',
};

const gh = (repo, sha, path) => `https://cdn.jsdelivr.net/gh/NakliTechie/${repo}@${sha}/${path}`;

export const SOURCES = {
  transformers: `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${PINS.transformers}/+esm`,
  wllama: `https://cdn.jsdelivr.net/npm/@wllama/wllama@${PINS.wllama}/esm`,
  kohra: gh('kohra', PINS.kohra, 'kohra.js'),
  diskformer: gh('diskformer.js', PINS.diskformer, 'engines'),
  diskformerLlama: gh('diskformer.js', PINS.diskformer, 'ggml/llama.js'),
};
