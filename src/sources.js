// Where each back end's runtime loads from: exact versions and commits, so a given anyformer behaves the same tomorrow.
// kohra and diskformer move fast on their main branches; `node scripts/pin-sources.mjs` moves these pins
// to each repo's current main, and the browser tests then prove the new pins (test/browser/README.md).
// transformers.js and wllama move only by hand, in step with LocalMind's pins (plan/history.md, 2026-10-08).

export const PINS = {
  transformers: '4.3.2', // 4.3.2 adds model_type lfm3 (LFM3 1.7B)
  wllama: '3.4.1',      // LocalMind's pin; its stream-drain workaround (src/backends/wllama.js) is for this version
  // v0.3.1: seeded block-KV graphs (Nemotron-Labs-Diffusion-3B), chatFormat, split external data; download progress
  // (progress_callback) and the graph files cached in Cache Storage 'kohra-models'.
  kohra: 'bf95d44c3e3cfc001201baa03fa121c2bf9c7d73',
  // 48f2c2a (main after v0.6.9): engines/lfm3_omni.js, LFM3-Omni-2B on onnxruntime-web 1.30.0 (text, image and speech
  // in; text out; files in Cache Storage 'diskformer-models'). Adds files only: no other engine or ggml/ changed.
  // v0.6.9: llama.load({ extra }) mounts side files beside the GGUF (DeepSeek V4.1 Flash's Engram tables).
  // v0.6.8: every engine (the disk-tier MoE engines, Ling 3.0 Flash, and LocalMind's whole-model engines LFM2.5,
  // Gemma 4 E2B/E4B and Ternary Bonsai 2) under engines/, and the llama.cpp path under ggml/, which loads split GGUFs
  // and drafters, and reads past 2 GiB (Nemotron 3.5 Bolt E4B needs v0.6.8 or later). The hand engines' ingest
  // streams the GGUF header with progress and resumes a dropped range.
  diskformer: '48f2c2a2f5bf59b16db8f3d57653e7c44705ce6c',
};

const gh = (repo, sha, path) => `https://cdn.jsdelivr.net/gh/NakliTechie/${repo}@${sha}/${path}`;

export const SOURCES = {
  transformers: `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${PINS.transformers}/+esm`,
  wllama: `https://cdn.jsdelivr.net/npm/@wllama/wllama@${PINS.wllama}/esm`,
  kohra: gh('kohra', PINS.kohra, 'kohra.js'),
  diskformer: gh('diskformer.js', PINS.diskformer, 'engines'),
  diskformerLlama: gh('diskformer.js', PINS.diskformer, 'ggml/llama.js'),
};
