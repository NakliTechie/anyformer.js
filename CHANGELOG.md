# Changelog

## 0.1.0 — 2026-10-14

The first release with code. `anyformer@0.0.1` reserved the name and held none.

### Added
- `plan(repo)` and `load(repo)`: read the Hub repo, probe the browser, ask every back end, run the best yes. A refusal
  carries a closed code (`format`, `model`, `task`, `webgpu`, `f16`, `opfs`, `budget`, `threw`) and one remedy in `REFUSALS`.
- Five back ends:
  - `transformers`: transformers.js 4.3.1. Text generation, image and audio chat (Gemma 4, Qwen3.5), embeddings,
    speech to text, detection, classification, depth, segmentation.
  - `kohra`: text diffusion (MDLM, BD3LM) over onnxruntime-web.
  - `localmind`: LocalMind's hand-written WebGPU kernels (LFM2 / LFM2.5, Gemma 4 E2B / E4B, Ternary Bonsai 2).
  - `wllama`: llama.cpp in WebAssembly for any other GGUF up to about 3.6 GB, on WebGPU or the CPU; files over 2 GB
    split into OPFS shards once.
  - `diskformer`: weights in OPFS, only what a step needs on the GPU (Gemma 4 26B-A4B from 2.5 GB, Qwen3.6 35B-A3B from 4 GB).
- `MODELS`: models that pass a browser test, loadable by entry.
- `pipeline(task, repo)`: transformers.js's call shape over any back end.
- `stored()` and `remove(repo)`: what is downloaded in this browser, across every back end.
- Coded errors: `AnyformerError` with a code from `ERROR_CODES`, checked with `isAnyformerError(e, code)`.
- One `generate()` at a time per session, in call order.
- Options: `load()` takes `task`, `maxCtx`, `dtype` and `signal`; `generate()` takes `maxNewTokens`, `repetitionPenalty`
  and `signal`; each back end's `.source` (for example `kohraBackend.source`) overrides where its runtime loads from.
- Pinned runtimes in `src/sources.js` (exact versions and commits); `npm run pin` moves them.
- Agent face: 9 tools over one manifest, through WebMCP's `navigator.modelContext` or `window.anyformer.tools`.
- The workbench (`examples/chat/`): pick a model, see the plan, then chat, compare embeddings or detect objects.

### Measured (Chrome, M4 Pro, 24 GB)
- 13 of 13 LocalMind Hub models plan at an 8 GB budget; 12 ran in a browser test.
- 10 of 10 replies byte-identical with LocalMind direct on the same transformers.js pin.
- Speed 0.99–1.02× LocalMind direct on the steady back ends.

### Not tested yet
Audio input blocks, abort mid-reply, speech / classification / depth / segmentation in a browser, any machine other
than the M4 Pro, and Qwen3.6 35B-A3B (it needs more than this machine's 24 GB).
