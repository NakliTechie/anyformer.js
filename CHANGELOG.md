# Changelog

## 0.4.18 — 2026-10-31

### Added
- LoRA adapters: `load(repo, { adapter: { repo, file, revision, size, scale } })` (or an entry carrying `adapter`)
  downloads the adapter GGUF into OPFS beside the model and applies it at `scale` on `diskformer-llama`, the only back
  end with `adapters: true`; the others drop out of the plan. Needs diskformer v0.6.17.
- `MODELS.chat.LFM2_5_24B_PI`: LFM2.5-24B-A2B (QAD-Q4_0) with harness-hub's Pi coding-agent LoRA
  (`harness-hub/LFM2.5-24B-A2B-pi-GGUF` at a994cf7, 37.4 MB), 4 GB budget, `experimental`, `license: 'lfm1.0'`.
  diskformer's gate (v0.6.17): 8/9 replies identical to llama-server --lora (7/9 without the adapter, the path's own
  baseline). The adapter was trained inside the Pi agent; elsewhere expect base-model behaviour or worse. Hub fixture
  at 0912d33.

### Changed
- diskformer pin f17009c → 314362e (v0.6.17, which includes v0.6.16). v0.6.16: Chrome 156 rejects mixed f16/f32 subgroup-matrix kernels on Apple GPUs,
  so Ternary Bonsai 2 27B, Gemma 4 E2B, LFM2.5 and the IQ-quant dense path failed to load ("Mixed precision subgroup
  matrix multiply is not supported by the device"). diskformer's new `engines/subgroup-matrix-guard.js` hides the
  feature when a tiny mixed kernel fails to compile, and those engines take their non-matrix prefill (slower prefill).
- Browser tests: `chat` takes `?turns=N` (up to 4 follow-up turns, each checked on topic); `vision` takes `?backend=`
  and `?local=1` (MiniCPM-V 4.7 on diskformer, image as packed RGB after `ensureVision()`); `run.mjs` prints each
  failed network request with its reason. `vision?local=1` serves `cats.jpg` from `--models` as well.

### Verified
- MiniCPM-V 4.7 image round trip through anyformer (browser `vision`, Chrome 156, this Mac, 11:12 IST, GGUF, vision
  encoder and image from local files, 6.5 GB budget, diskformer f17009c): "There are **two cats** in this picture",
  then a correct description, 40 tokens at 18.3 tok/s.

## 0.4.17 — 2026-10-30

### Added
- `MODELS.chat.CLOVER_35B_A3B`: Diffuspace's Clover-35B-A3B-Preview (Apache-2.0), block diffusion on Qwen3.6-35B-A3B,
  text only, on diskformer's `clover_moe_ssd.js` (Q4_0 GGUF, 19.6 GB, from OPFS; 4 GB budget). `diffusion` and
  `experimental`; speed unmeasured. generate() yields once per committed 32-token block. Hub fixture at revision
  32e22a0. Browser test `chat` passed (Chrome, this Mac, 23:17 IST, GGUF from Models2, 4 GB budget): load 55.6 s
  including the copy into OPFS, an on-topic 25-token reply in 25.4 s.
- Browser test `chat` takes `?local=1` for a diskformer GGUF served from `--models`.

### Changed
- diskformer pin 93193d0 → f17009c (v0.6.15: the Clover engine; `qwen35_moe_ssd.js` and `qwen3_moe_ssd.js` only learn
  to accept the `clover_moe` architecture).

## 0.4.16 — 2026-10-30

### Changed
- diskformer pin 7ffbea5 → 93193d0 (v0.6.14: LFM3-Omni speech out runs at 1.5–1.8x real time, up from 0.54x, on an
  M4 Pro; the inverse STFT uses an FFT with the same output; changes only `engines/lfm3_omni.js`).
- Browser test `omni-speech` on the new pin (Chrome, this Mac, local files, about 22:50 IST): 13.52 s of speech in
  7.1 s (1.9x real time), first sound after 0.98 s; abort at the first speech chunk; text reply after it equal.

## 0.4.15 — 2026-10-30

### Fixed
- A caller that stops reading a reply early (a `break` out of `for await`, as after a tool call) without aborting
  now stops the reply on `diskformer-onnx` and `transformers`: the back end aborts the engine call (a linked
  AbortController for diskformer-onnx, the stopping criteria's interrupt for transformers.js) and waits for it before
  the next generate() starts. Before, the engine kept generating, and the next reply ran on the same onnxruntime
  sessions beside it: "failed to call OrtRun() … Failed to download data from buffer" or a hang of over 10 minutes in
  Chrome (reported by NakliOS, root-caused by LocalMind). `diskformer-llama` already did this.

### Changed
- diskformer pin 1fe19f8 → 7ffbea5 (v0.6.13: LFM3-Omni serves one reply at a time; changes only
  `engines/lfm3_omni.js`).
- Browser tests `omni` and `omni-speech` break out of a reply (text, and spoken at its first speech chunk) without
  aborting, then require the next reply to equal the text reference within 10 s. Passed 2026-10-30 (Chrome, this Mac,
  local files), with `transformers` and the LFM3 1.7B `chat` check.

## 0.4.14 — 2026-10-30

### Added
- `diskformer-onnx`: LFM3-Omni speech out. `load(MODELS.chat.LFM3_OMNI_2B, { speech: true })` also fetches the depth
  and codec decoders (515 MB, Cache Storage 'diskformer-models'); `generate(messages, { speech: true })` answers in
  interleaved text and speech, and a chunk with new speech carries `audio` (24 kHz Float32 PCM since the last chunk)
  and `sampleRate`. `seed`, `audioTemperature`, `audioRepetitionPenalty` and `audioRepetitionWindow` pass through.
  Catalog flag `audioOut: true` on `MODELS.chat.LFM3_OMNI_2B`.
- Browser test `omni-speech` (Chrome, this Mac, 2026-10-30, local files): the reference speech clip in, a spoken reply
  that names Mount Everest, 13.5 s of 24 kHz speech in 15 chunks over 9.3 s (first sound after 1.1 s), RMS 0.037
  (the reference's 0.048); an abort from the first speech chunk ended the reply after 0.9 s of speech; a text reply
  after it still equals the text reference. Code-level exactness is diskformer's own gate.

### Changed
- diskformer pin 48f2c2a → 1fe19f8 (v0.6.12, LFM3-Omni speech out; changes only `engines/lfm3_omni.js`). The `omni`
  test passes again on it (3/3 equal to the reference, streaming, abort, tool call).

## 0.4.13 — 2026-10-30

### Added
- `diskformer-onnx` back end: ONNX exports that transformers.js has no model class for, run by a diskformer hand
  engine on onnxruntime-web WebGPU (the engine pins onnxruntime-web 1.30.0). Ranks second, after `transformers`, so
  transformers.js takes a model over once it adds the class. Streams text, passes `signal` (an abort ends the reply
  quietly), `thinking` (off by default), `tools` (tool calls stay raw text) and `repetitionPenalty`; image blocks take
  ImageData, ImageBitmap, a URL or `{ data, width, height }`, audio blocks a 16 kHz Float32Array of at most 30 s.
  `session.baseUrl` names the URLs its Cache Storage entries ('diskformer-models', listed by `stored()`) are keyed by.
- `MODELS.chat.LFM3_OMNI_2B`: `LiquidAI/LFM3-Omni-2B-ONNX` (task `any-to-any`, 2.14 GB: decoder_q4f16,
  vision_encoder_q8, audio_encoder) on `diskformer-onnx`, asked for by LocalMind for Chirag. Browser test `omni`
  (Chrome, this Mac, 2026-10-30), from local files and from Hugging Face: diskformer's three gate cases through
  `session.generate()` equal the text Liquid's runner gives on onnxruntime's CPU EP (text 7 tokens, image 32, speech
  32); the image reply streams in 32 chunks; an abort after 3 chunks ends the reply; a `get_weather` tool list gives a
  Qwen3-Coder XML tool call.

### Changed
- diskformer pin 53c976e (v0.6.9) → 48f2c2a (`engines/lfm3_omni.js`; adds files only, no other engine changed).
- `test/browser/run.mjs --models` serves subfolders (a repo layout), still only inside the named folders.

## 0.4.12 — 2026-10-29

### Changed
- transformers.js pin 4.3.1 → 4.3.2 (in step with LocalMind), which adds `model_type` `lfm3` and `lfm3_vl`.
- transformers back end: a graph whose last-position input is named `logits_to_keep` (LiquidAI's LFM3 ONNX export)
  gets it fed at the session's run; transformers.js 4.3.2 feeds only `num_logits_to_keep` and stopped with "Missing
  the following inputs". `LiquidAI/LFM3-1.7B-ONNX` now loads and generates through `chat.html` (64 tokens at
  67 tok/s, Chrome, this Mac, 2026-10-28).
- transformers back end, text pipeline path: `thinking: false` by default, as on every other back end. The template
  renders with `enable_thinking` off (through `tokenizer_encode_kwargs`), so LFM3 answers directly; `thinking: true`
  per request turns it on. Templates without the switch ignore it.
- `test/browser/chat.html` takes `?dtype=` and fails a reply that contains `<think>` text.

### Added
- `MODELS.chat.LFM3_1_7B`: `LiquidAI/LFM3-1.7B-ONNX` on `transformers` (asked for by LocalMind for Chirag). Browser
  chat check on 2026-10-29, Chrome, this Mac: on topic, no `<think>` text, on q4 (anyformer's pick, 1.63 GB; 82.4
  tok/s) and on q4f16 (`dtype: 'q4f16'`, LocalMind's precision, 1.38 GB; 76.6 tok/s), the same 61-token reply.

## 0.4.11 — 2026-10-24

### Added
- `MODELS.chat.DEEPSEEK_V4_1_FLASH`: DeepSeek V4.1 Flash (763B MoE, ISTA-DASLab GSQ-RCO IQ1_M) on `diskformer-llama`,
  budget 8 GB, `experimental: true`, so LocalMind shows it as an opt-in tech demo (Chirag's approval, relayed by the
  LocalMind session). 231,316,411,072 bytes on disk: 4 parts plus the 110.6 GB Engram file. Very slow: diskformer
  measured about 0.4 tok/s natively from a USB SSD.
- `diskformer-llama`: side files (`SIDE_FILES`). A model's side file downloads with its parts under its own name,
  counts toward the disk bytes, goes to llama.cpp as `load({ extra })`, and is never picked as the model. The
  `deepseek41-engram` architecture is accepted for the demo file only; another quant in the repo is refused.

### Changed
- diskformer pinned to v0.6.9 (53c976e), which adds `load({ extra })`. diskformer reports Granite and Bolt 9/9 on it.

### Verified
- npm test 92/92 (fixture of the repo at cb02422c: the IQ1_M set with its side file, no Engram file alone, a missing
  side file refuses, `extra` reaches the fake llama.js). `plan()` against the live Hub picks `diskformer-llama` at 8 GB
  with 231,316,411,072 disk bytes.
- Not run, on purpose (Chirag: no live runs or benchmarks of this model): any browser load through anyformer.
  diskformer checked one hand-written prompt through `complete()` in Chrome; the chat-template path that `generate()`
  uses, and so LocalMind's chat, has not run in any browser.

## 0.4.10 — 2026-10-24

### Changed
- kohra pinned to v0.3.1 (bf95d44): `from_pretrained` reports download progress, which anyformer's `onProgress`
  receives (forwarded since npm 0.4.9), and caches the graph files in Cache Storage 'kohra-models', which `stored()`
  lists and `remove()` deletes. A second load of Nemotron-Labs-Diffusion-3B reads them from there.

### Verified
- npm test 90/90. kohra's own Chrome check on v0.3.1 (kohra session): the first Nemotron load cached all 3 files with
  their content-length; a reload was ready in 7.4 s with no graph fetches; output unchanged. Through anyformer
  (`npm run test:browser -- kohra-cache`, Chrome, GPU shared, after the npm publish): the first load sent 35,416
  progress events to `onProgress` and answered on topic in 200 s; `stored()` listed 2,735,424,112 bytes under kohra;
  the second load took 5.1 s with the same reply; `remove()` freed 2,752,678,776 bytes (graph and tokenizer).

## 0.4.9 — 2026-10-24

npm 0.4.9 was published from main 69bed78, one commit after tag v0.4.9 (cce1e1a), so it also carries the change
below. With the kohra v0.3.0 pin that change does nothing yet: v0.3.0 ignores `progress_callback` and caches nothing.

### Added (69bed78)
- `kohra`: `onProgress` is forwarded to kohra's `from_pretrained` as `progress_callback` (kohra v0.3.1+), so the
  2.7 GB Nemotron download reports bytes. `stored()` and `remove()` list kohra's Cache Storage 'kohra-models'.

### Added
- Catalog flags: `diffusion: true` on the two kohra entries (Qwen3-0.6B MDLM, Nemotron-Labs-Diffusion-3B), and
  `experimental: true` on Nemotron-Labs-Diffusion-3B, so LocalMind can show it as opt-in, "experimental · diffusion",
  until its multi-turn chat check passes (Chirag's conditions for LocalMind, 2026-10-24). `anyformer_models` returns
  them.

### Verified
- npm test 90/90. No browser check: catalog fields only.

## 0.4.8 — 2026-10-24

Skips 0.4.7 on npm: this release carries its Nemotron-Labs-Diffusion-3B entry too.

### Changed
- kohra pinned to v0.3.0 (85e01f0); kohra.js is identical to 0.4.7's pin, 14e430c.
- Nemotron-Labs-Diffusion-3B decodes with threshold 0.9 by default (Fast-dLLM: unmask every position at that
  confidence per step); `generate(messages, { threshold: null })` runs fixed steps. kohra measured 34 forward passes
  against 64 with the same answer.

### Verified
- Browser (Chrome, M4 Pro, GPU shared with another session): one page, the CPU prompt, 32 tokens, threshold
  null / 0.9 / null / 0.9: identical text in every run; 7.1 s and 5.8 s at fixed steps, 4.9 s at 0.9. npm test 89/89.

## 0.4.7 — 2026-10-24

### Added
- Nemotron-Labs-Diffusion-3B (`MODELS.chat.NEMOTRON_DIFFUSION_3B`, naklitechie/Nemotron-Labs-Diffusion-3B-ONNX) on
  `kohra`: the q4f16 seeded block-KV graph (onnx/model_kv_q4f16.onnx, 2.73 GB with its two data files), mask id 100,
  EOS 11, and Nemotron's chat template over the whole conversation (a system turn, an empty think block before each
  assistant reply). Content blocks are joined as text.

### Changed
- kohra pinned to main 14e430c (after v0.2.1; kohra.js changed in afee5f4): seeded KV graphs, `chatFormat`, graphs
  whose data is split into `_data`, `_data_1`.

### Verified
- Browser (Chrome, M4 Pro): `npm run test:browser -- kohra --query "repo=naklitechie/Nemotron-Labs-Diffusion-3B-ONNX"`
  answered on topic (32 tokens, 5.0 tok/s, load 156 s with the download); `npm run test:browser -- kohra` (Qwen3-0.6B
  MDLM) on the new pin answered on topic. npm test 89/89.

## 0.4.6 — 2026-10-24

Skips 0.4.5 on npm: this release carries its diskformer v0.6.8 pin too.

### Added
- Nemotron 3.5 Bolt E4B (`MODELS.chat.NEMOTRON_3_5_BOLT_E4B`) on `transformers` at a 5 GB budget:
  danyalmalik/NVIDIA-Nemotron-3.5-Bolt-E4B-ONNX-WebGPU, a community q4f16 export (Bolt's weights in onnx-community's
  Gemma 4 E4B graph), text only, 4.91 GB.
- `transformers`: a Gemma 4 repo in the multimodal layout with no vision or audio encoder loads through
  `Gemma4ForCausalLM` and the tokenizer. Its `use_external_data_format` comes from the repo's `_data` files, because
  that repo's config.json has no transformers.js_config (without it: "Module.MountedFiles is not available"). Image
  and audio blocks are refused with `INVALID_ARGUMENT`.

### Verified
- Browser (Chrome, M4 Pro): `npm run test:browser -- transformers --query "repo=danyalmalik/NVIDIA-Nemotron-3.5-Bolt-E4B-ONNX-WebGPU&budget=5.3"`
  answered on topic, 34.9 tok/s, load 231 s with the download. npm test 88/88.

## 0.4.5 — 2026-10-24

### Changed
- diskformer pinned to v0.6.8 (3aad103). Its llama.cpp path (ggml/) reads past 2 GiB in a browser, adds head-512
  attention and loads drafter GGUFs (`draft`). The hand engines (engines/, src/) are unchanged since v0.6.7.

### Verified
- `llama-gate` on Granite 4.0 H Tiny (Chrome, M4 Pro, v0.6.8 pin): 9/9 identical to diskformer's refs, chat path
  equal, 19.1 tok/s median. npm test 87/87. Not rerun: the hand-engine gates, because engines/ did not change.

## 0.4.4 — 2026-10-24

### Changed
- diskformer pinned to v0.6.7 (663b2b0). Its hand engines' ingest (src/ingest.js) now streams the GGUF header with
  progress, so a slow link no longer goes silent for minutes (Flint at 0.1 MB/s stalled LocalMind's watchdog for 3 min).
  A range that drops mid-stream, or returns 5xx or 429, resumes from the last byte, up to 6 times. The new events reach
  `onProgress` as status `weights` with phase `header` or `retry`. diskformer changed only src/ingest.js and its test
  between v0.6.6 and v0.6.7.

### Verified
- npm test 87/87. Not run here in a browser: LocalMind's throttled production load of Flint is that check.

## 0.4.3 — 2026-10-24

npm skips 0.4.2: this release carries it (Flint 40B and LFM2.5 24B on the hand engines).

### Added
- `diskformer-llama` loads split GGUFs (`name-00001-of-0000n.gguf`): `pickGguf` returns a set named by its first part,
  with every part and their summed size; a set missing a part is refused. Each part downloads into OPFS under its own
  file name and `load` names the first, as diskformer v0.6.5's worker expects. Ling 3.0 Flash (77 GB, 2 parts) now
  routes to the llama.cpp path under the hand engine's 6.8 GB floor (diskformer v0.6.5: 6 GB, 4.1–4.9 tok/s, 6/9 vs
  llama.cpp on the CPU) and stays on the hand engine above it. Ling 3.0 Flash through anyformer in a browser is not yet run.

### Changed
- diskformer pinned to v0.6.6 (e07e129), whose llama.cpp worker loads split GGUFs.
- OPFS names for llama.cpp's GGUFs drop the repo's folders (gguf/<file name>); a file in a repo subfolder failed before.

### Verified
- Granite 4.0 H Tiny through `diskformer-llama` on v0.6.6, Chrome on the M4 Pro: 9/9 replies identical to llama.cpp, chat path equal to the reference.
- LocalMind's checks: through 0.4.1, LFM2.5 230M, Gemma 4 E2B and Ternary Bonsai 2 27B gave identical replies on 21/21 rows in both run orders, with decode within run-to-run spread; through 0.4.2, Flint 40B and LFM2.5 24B gave identical replies 3/3 in four runs, with equal GPU memory and equal or faster decode.

## 0.4.2 — 2026-10-23

### Added
- Flint 40B-A2.5B (RecursiveSI/flint-int4-gguf, Q4_1, 26.0 GB) and LFM2.5 24B-A2B (LiquidAI/LFM2.5-24B-A2B-GGUF,
  QAD-Q4_0, 13.6 GB) on the `diskformer` back end, through diskformer's `FlintMoeSsd` and `Lfm2MoeSsd` hand engines.
  Before, both routed to the llama.cpp path (Flint 1.8–3.4 tok/s there against 8.66 on its engine). Numbers from
  LocalMind's gates: Flint ~7.3 GB on the GPU with a 4 GiB pool, 8.66 tok/s; LFM2.5 24B ~16 tok/s at a 4 GiB pool.
  Floors of 4 GB are unmeasured. Not yet run through anyformer in a browser; LocalMind's live check is that check.

### Changed
- README: the `localmind` back end's kernels ship in diskformer.

## 0.4.1 — 2026-10-23

### Fixed
- `session.speculative` and `session.pleMode` were copied once when `load()` returned, because the session object was
  built with spreads, which read getters. A Bonsai session's `speculative.status` stayed `'loading'` after DFlash or
  MTP attached. They are now read live. A test through `load()` failed on 0.4.0 and passes now.

### Added
- `session.engine`: the engine instance the `diskformer` and `localmind` back ends built, read-only, for token-level
  work a caller already does with it (Bonsai 2's tokenizer and think-marker ids in LocalMind's worker). Calls on it
  bypass the session's one-call-at-a-time queue.

## 0.4.0 — 2026-10-23

What LocalMind's own workers do with its last three engine families (LFM2.5, Gemma 4 E2B/E4B, Ternary Bonsai 2 27B),
on the `localmind` back end, so those models can move onto anyformer with nothing lost.

### Added
- `load(repo, { fetch })`: the engines' fetch, for a caller's retrying fetch on multi-GB range reads.
- `load(repo, { ple: true })` for Gemma 4 E2B/E4B: the per-layer-embedding table streams from OPFS (diskformer's
  `ple-opfs.js`); `session.pleMode` reports the mode.
- `load(repo, { speculative })` for Ternary Bonsai 2 27B: `{ dflash: { block } }` (DFlash 2) or
  `{ mtp: { depth, adaptive } }`, attached after the model is ready so the first reply never waits for it. The drafter
  and the MTP head download once into Cache Storage, under LocalMind's names and keys, so a browser that holds them
  reuses them. `session.speculative` gives `{ status, error, stats }`. Output is greedy-identical either way.
- `generate(messages, { thinking, assistantPrefix })` on the `localmind` back end: thinking per request (Bonsai's
  template, `reasoning_effort: 'medium'`) and a raw prefix to continue from; chunks carry `token`.
- `session.reset()` where the engine has it, queued with `generate()`.

### Measured
- LFM2.5 230M through the `localmind` back end: the same reply as before these changes. Node tests cover each option
  with stand-in engines (83/83). Gemma's PLE table and Bonsai's DFlash and MTP have not run through anyformer in a
  browser; LocalMind's A/B per model is that check.

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
