# Changelog

## 0.3.2 — 2026-10-23

### Changed
- Nothing loads from the LocalMind repo any more. diskformer v0.6.4 (095ab20) holds LocalMind's whole-model
  engines too (LFM2.5, Gemma 4 E2B/E4B, Ternary Bonsai 2 27B; MIT, confirmed by their author), and the `localmind`
  back end loads them from diskformer's `engines/`. The id stays `localmind`. `PINS.localmind` and
  `SOURCES.localmind` are gone.
- LFM2.5 230M through the `localmind` back end from diskformer's copy: the same reply as from LocalMind's commit
  203288a (the engine file is byte-identical), on topic.

## 0.3.1 — 2026-10-23

### Added
- MiniCPM-V 4.7 35B-A3B (openbmb/MiniCPM-V-4.7-35B-A3B-gguf, Q4_0, 19.7 GB) on the `diskformer` back end, through
  diskformer's `MiniCpmV47Ssd`: text, and images through `ensureVision()` and `{ type: 'image', rgb, width, height }`
  blocks. Floor 4 GB (Qwen3.6's; the smallest working budget is unmeasured); 14.1 tok/s over short continuations at a
  4 GiB pool, ~6.1 GB on the GPU (LocalMind's gate). It ranks before `diskformer-llama`, which has no image path.
  Not yet run through anyformer in a browser; LocalMind's live check is that check.

## 0.3.0 — 2026-10-23

What LocalMind's worker uses from its disk-tier engines, so its models can move onto anyformer with nothing lost.

### Added
- `generate()` chunks on the `diskformer` back end carry `token`, the engine's token id.
- `session.think`: `{ openId, closeId, inPrompt, preamble, always }`, where a model's reasoning starts and ends as
  token ids (Gemma 4's thought channel, Qwen3.6 and Ling's `<think>`); null where the back end has none.
- `session.prime(system)`, where the engine has it (Qwen3.6 35B): checkpoints a system prompt after load.
- `session.ensureVision({ onProgress })`, where the engine has it; image blocks `{ type: 'image', rgb, width, height }`
  pass through to such an engine. No vision model is in `DISKFORMER_MODELS` yet.
- `load(repo, { poolBytes })`: the expert cache size, passed to the engine; the budget then only admits the model.
- `load(repo, { localFile })`: a GGUF the user picked, ingested instead of downloaded.

## 0.2.5 — 2026-10-23

### Changed
- diskformer pinned to v0.6.3 (1d8b418): Ling 3.0 Flash loads a complete store without reading the parts' headers
  from Hugging Face; the browser llama.cpp build is rebuilt (split GGUFs still refused there). Granite 4.0 H Tiny
  through `diskformer-llama` on the new build: 9/9, chat path = reference. The hand engines anyformer gated are
  unchanged (`qwen3_moe_ssd.js` only adds an export).

## 0.2.4 — 2026-10-23

### Changed
- The `diskformer` back end's `generate()` honours `thinking`, as `diskformer-llama` does since 0.2.1: off by default,
  `{ thinking: true }` lets Gemma 4, Qwen3.6 or Ling 3.0 Flash reason first (the engines' `enableThinking`).

## 0.2.3 — 2026-10-23

### Added
- Ling 3.0 Flash (inclusionAI/Ling-3.0-flash-GGUF, two-part Q4_K_M, 77 GB) on the `diskformer` back end, through
  diskformer's `LingMoeSsd` hand engine: from a 6.8 GB budget, 5.53 tok/s and an 80.7 GB OPFS store (LocalMind's gate).
  Not yet run through anyformer in a browser: LocalMind's live check through anyformer is that check.
- `describe()` keeps the Hub's last answer per repo in Cache Storage (`anyformer-describe-v1`) and uses it when the
  network fails, so a downloaded model loads offline.

### Changed
- diskformer pinned to v0.6.2 (f5f1177): adds `ling_moe_ssd.js` and two quant formats to `iq_quants.js`, which no
  engine gated earlier imports; the earlier gates stand.

## 0.2.2 — 2026-10-23

### Added
- `load(repo, { file })` and `plan(repo, { file })`: load a chosen GGUF from the repo; a back end that cannot load
  that file refuses with code `model`.

### Changed
- `diskformer-llama` loads the file diskformer checked when the repo has it (`CHECKED_FILES`, from diskformer's
  catalog and coverage table), else the preferred quant. GLM-4.7-Flash now defaults to Q3_K_M (14.6 GB, gated 8/9 by
  diskformer), not Q4_K_M (18.3 GB, not gated). Granite 4.0 H Tiny and Gemma 4 26B keep the files their gates used.

## 0.2.1 — 2026-10-23

### Changed
- diskformer pinned to v0.6.1 (5bccf32): its llama.cpp build takes a thinking switch. `diskformer-llama`'s
  `generate()` now turns thinking off by default (`thinking: true` turns it on), so a model that reasons first answers
  directly, as on the other back ends.

### Measured (Chrome, M4 Pro; correctness only)
- Granite 4.0 H Tiny via `diskformer-llama`: 9/9; the chat path matches the reference.
- Gemma 4 26B-A4B via `diskformer-llama`: 8/9 (miss: system-pirate, as diskformer's own run); the chat path matches the
  reference, with no thought block. The hand engines are unchanged in v0.6.1, so their 9/9 at 0.2.0 stands.

## 0.2.0 — 2026-10-22

### Added
- `diskformer-llama` back end: mixture-of-experts GGUFs through diskformer's llama.cpp path (`diskformer/llama`,
  llama.cpp with WebGPU in a worker, experts paged from OPFS), for the 17 architectures diskformer has checked
  (`LLAMA_ARCHITECTURES`). A diskformer hand-written engine still ranks first where it covers the model.
- `probe()` reports `jspi` (WebAssembly JSPI), which the llama.cpp build needs.
- `stored()` and `remove()` cover the llama.cpp path's GGUFs in OPFS (`gguf/`).

### Changed
- diskformer pinned to v0.6.0 (439ad4e) for both its hand engines and the llama.cpp path.

### Measured (Chrome, M4 Pro)
- Granite 4.0 H Tiny through `diskformer-llama`: 9/9 replies identical to diskformer's reference
  (`npm run test:browser -- llama-gate`).
- Gemma 4 26B-A4B through `diskformer-llama`: 8/9 identical to llama.cpp b9830, the same 8/9 (miss: system-pirate)
  diskformer records for its own Chrome run. The resolver still sends this repo to the hand engine.
- Gemma 4 26B-A4B on the hand engine at the new pin: 9/9 identical (`diskformer-chat`).

### Known gaps
- `diskformer-llama`'s chat path does not turn thinking off: a model that reasons first (Gemma 4, Qwen3.6, gpt-oss)
  begins its reply with its thought block. Closed in 0.2.1.

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
