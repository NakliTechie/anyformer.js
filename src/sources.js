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
  // v0.6.20 (7962912): run-time LoRA in every native MoE engine (engines/lora.js): Gemma 4 26B, Qwen3.6 35B, LFM2.5
  // 24B and Qwen3 (untested); adapters on routed experts are refused. Engines expose addBos from the GGUF.
  // v0.6.19 (cefebc7): run-time LoRA in lfm2_moe_ssd.js (engine.loadAdapter / setAdapter, no reload to switch).
  // v0.6.18 (317d2a0): with thinking off, the llama.cpp chat closes a template's always-open <think> (LFM2.5-24B) and
  // bans the <think> token for the reply.
  // v0.6.17 (314362e): LoRA adapters on the llama.cpp path (llama.load({ lora, loraScale })), and lfm2_moe_ssd.js runs
  // a LoRA-merged LFM2.5-24B-A2B with Q8_0 dense matrices.
  // v0.6.16 (467052f): engines/subgroup-matrix-guard.js. Chrome 156 rejects mixed f16/f32 subgroup-matrix kernels
  // on Apple GPUs; the guard hides the feature when one tiny mixed kernel fails to compile, so Ternary Bonsai 2 27B,
  // Gemma 4 E2B, LFM2.5 and qwen3_moe_ssd.js take their non-matrix prefill (slower prefill, same output path).
  // v0.6.15 (f17009c): engines/clover_moe_ssd.js, Clover-35B-A3B-Preview (block diffusion on Qwen3.6-35B-A3B; kohra's
  // denoise loop vendored in engines/vendor/). qwen35_moe_ssd.js and qwen3_moe_ssd.js only learn to accept clover_moe.
  // v0.6.14 (93193d0): LFM3-Omni speech out's inverse STFT uses an FFT (same samples), 0.54x → 1.5–1.8x real time on
  // an M4 Pro. Changes only engines/lfm3_omni.js.
  // v0.6.13 (7ffbea5): LFM3-Omni serves one reply at a time; overlapping runs broke onnxruntime-web WebGPU.
  // v0.6.12 (1fe19f8): LFM3-Omni speech out (interleaved text and 24 kHz speech; depth and codec decoders). Changes
  // only engines/lfm3_omni.js; its text, image and speech-in gate (3/3) is unchanged.
  // 48f2c2a (main after v0.6.9): engines/lfm3_omni.js, LFM3-Omni-2B on onnxruntime-web 1.30.0 (text, image and speech
  // in; text out; files in Cache Storage 'diskformer-models'). Adds files only: no other engine or ggml/ changed.
  // v0.6.9: llama.load({ extra }) mounts side files beside the GGUF (DeepSeek V4.1 Flash's Engram tables).
  // v0.6.8: every engine (the disk-tier MoE engines, Ling 3.0 Flash, and LocalMind's whole-model engines LFM2.5,
  // Gemma 4 E2B/E4B and Ternary Bonsai 2) under engines/, and the llama.cpp path under ggml/, which loads split GGUFs
  // and drafters, and reads past 2 GiB (Nemotron 3.5 Bolt E4B needs v0.6.8 or later). The hand engines' ingest
  // streams the GGUF header with progress and resumes a dropped range.
  diskformer: '79629125d7a39b13ddcaf74df58262458d8d470e',
};

const gh = (repo, sha, path) => `https://cdn.jsdelivr.net/gh/NakliTechie/${repo}@${sha}/${path}`;

export const SOURCES = {
  transformers: `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${PINS.transformers}/+esm`,
  wllama: `https://cdn.jsdelivr.net/npm/@wllama/wllama@${PINS.wllama}/esm`,
  kohra: gh('kohra', PINS.kohra, 'kohra.js'),
  diskformer: gh('diskformer.js', PINS.diskformer, 'engines'),
  diskformerLlama: gh('diskformer.js', PINS.diskformer, 'ggml/llama.js'),
};
