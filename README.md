<h1 align="center">anyformer.js</h1>

**One call to run a model in a browser tab: anyformer reads the repo, probes the browser, and picks the back end that
fits the model's format, architecture and your GPU budget.**

Plain ES modules, no build step, no dependencies to install. Chromium with WebGPU (GGUF models also run on the CPU). Weights stay in the user's own
browser storage; no account, no server, no telemetry.

[![license](https://img.shields.io/badge/license-MIT-555?style=flat-square)](LICENSE)
[![dependencies](https://img.shields.io/badge/dependencies-none-555?style=flat-square)](package.json)

## Install

| How | Command |
|---|---|
| npm | `npm install anyformer` |
| CDN | `import { load } from 'https://cdn.jsdelivr.net/npm/anyformer@0.1.0/+esm';` |
| Copy | `index.js` and `src/`; nothing else is needed |

```js
import { load } from 'anyformer';

const chat = await load('onnx-community/Qwen2.5-0.5B-Instruct', { budget: 2.5e9, onProgress: console.log });
for await (const { text } of chat.generate([{ role: 'user', content: 'What does a CPU do?' }], { maxNewTokens: 64 }))
  show(text);
```

Every session has `run(input)`: chat messages for text generation (content a string, or blocks such as
`[{ type: 'image', image }, { type: 'text', text }]` for Gemma 4 and Qwen3.5), a string or strings for embeddings, ImageData for
vision, 16 kHz samples for speech. `load(repo, { task })` picks a task when a repo has more than one. The first load
downloads the weights into the browser; later loads read them from there. Each back end's runtime loads
from jsDelivr only when that back end is chosen.

## Try it

Serve the repo and open the workbench: pick a model, see which back end anyformer chose and why every other one
refused, then chat (with an image for the vision models), compare embeddings, or detect objects; remove downloads.

```bash
python3 -m http.server 8000      # then open http://localhost:8000/examples/chat/
```

## Why

You have a model id and a browser tab. Whether it runs depends on its format (ONNX, GGUF), its architecture, whether it
is autoregressive or diffusion, and how much GPU memory the tab can take. Each runtime covers part of that, and you
find out which part by failing to load.

anyformer asks every back end first. `plan(repo)` returns the model's description, the browser's capabilities and each
back end's answer: yes with an estimate (GPU bytes, disk bytes, tok/s when measured), or no with a code and a reason.
`load(repo)` runs the best yes.

## Back ends

| Back end | Runs | Models |
|---|---|---|
| `transformers` | transformers.js 4.3.1: text generation, image and audio chat (Gemma 4, Qwen3.5), embeddings, speech to text, detection, classification, depth, segmentation | ONNX repos in the onnx-community / Xenova layout |
| `kohra` | kohra, text-diffusion over onnxruntime-web | MDLM and BD3LM repos tagged `text-diffusion` |
| `localmind` | LocalMind's hand-written WebGPU kernels (shipped in diskformer), the whole model on the GPU | any LFM2 / LFM2.5 GGUF; Gemma 4 E2B / E4B QAT-mobile; Ternary Bonsai 2 |
| `wllama` | llama.cpp in WebAssembly: WebGPU when the budget holds the model, the CPU otherwise (no WebGPU needed) | any GGUF up to about 3.6 GB that no engine above covers; files over 2 GB are split into OPFS shards once |
| `diskformer` | diskformer engines: weights in OPFS, only what a step needs on the GPU | Gemma 4 26B-A4B (from 2.5 GB), Qwen3.6 35B-A3B (from 4 GB), Ling 3.0 Flash 124B (from 6.8 GB), MiniCPM-V 4.7 with images, Flint 40B, LFM2.5 24B (from 4 GB) |
| `diskformer-llama` | diskformer's llama.cpp build with WebGPU, mixture-of-experts layers paged in from OPFS | GGUFs of the 17 MoE architectures diskformer has checked (Granite 4.0 H, gpt-oss, GLM-4.7-Flash, LFM2 MoE…); a hand engine above wins where it covers the model |

Each runtime loads from an exact version or commit in [src/sources.js](src/sources.js); set a back end's `.source`
(for example `kohraBackend.source`) to load another copy. `BACKENDS` lists the back ends `load()` asks by default: each joins after its browser test passes.

## When no back end can run it

`load()` throws an `AnyformerError` with code `NO_BACKEND` and the plan, naming each back end, its code and its
reason. Codes are a closed set, each with one remedy in `REFUSALS`: `format`, `model`, `task`, `webgpu`, `f16`, `opfs`, `budget` (with `need`, the budget that would pass)
and `threw`. Every other failure is an `AnyformerError` too, with a code from `ERROR_CODES` (`LOAD_FAILED`,
`BUDGET_TOO_SMALL`, `ABORTED`, `EXECUTION_FAILED`, `INVALID_ARGUMENT`, `DISPOSED`); check it with
`isAnyformerError(e, code)`. A session runs one `generate()` at a time, in call order.

## Commands

```js
plan(repo, { budget, task })      // what would run it, and why the others would not; loads nothing
load(repo, { budget, task, file, maxCtx, poolBytes, localFile, fetch, ple, speculative, onProgress, signal })  // → { backend, task, plan, run(input), generate?, think?, prime?, reset?, speculative?, dispose() }
pipeline(task, repo, { budget })  // transformers.js's call shape: gen(messages) → [{ generated_text }]
probe()                           // WebGPU, shader-f16, subgroups, limits, OPFS quota, memory64
describe(repo)                    // Hub repo → formats, architecture, task, tags, files
resolve(model, caps, backends, { budget })  // rank back ends for a described model
stored() · remove(repo)           // what is downloaded in this browser; delete one
MODELS.chat.DEFAULT               // models that pass a browser test; load(MODELS.chat.GEMMA4_26B_A4B) works too
registerTools()                   // publish the agent face (below)
```

Agents get the same calls as tools (`anyformer_models`, `_probe`, `_plan`, `_load`, `_generate`, `_sessions`, `_unload`,
`_stored`, `_remove`) through
WebMCP's `navigator.modelContext` where the browser has it and `window.anyformer.tools` otherwise; `anyformer_load`
loads nothing until the caller accepts the plan's disk bytes, and `anyformer_remove` deletes nothing without
`confirm: true`.

## Verify it yourself

```bash
npm test                                   # Node: probe, describe, resolver, back-end verdicts, agent face
npm run test:browser -- probe              # headless Chrome: WebGPU, f16, OPFS
npm run test:browser -- transformers       # Qwen2.5-0.5B answers on topic
npm run test:browser -- kohra              # Qwen3-0.6B MDLM answers on topic
npm run test:browser -- embed             # MiniLM embeddings rank a paraphrase above an unrelated sentence
npm run test:browser -- detect            # DETR finds both cats in cats.jpg
npm run test:browser -- examples/chat/index.html --query "auto=1"   # the workbench loads and answers
npm run test:browser -- pipeline --port 8192 --profile ~/.cache/anyformer-chat-profile   # pipeline() on a GGUF
npm run test:browser -- store             # list a download, remove it
npm run test:browser -- chat --query "repo=LiquidAI/LFM2.5-1.2B-Instruct-GGUF"   # a LocalMind engine
npm run test:browser -- vision --query "repo=onnx-community/Qwen3.5-4B-ONNX-OPT"   # names the two cats
npm run test:browser -- diskformer-chat    # Gemma 4 26B-A4B at 2.5 GB: 9/9 replies identical to llama.cpp b9830
npm run test:browser -- llama-gate --query "repo=ibm-granite/granite-4.0-h-tiny-GGUF&refs=diskformer-granite-refs"   # 9/9 via diskformer-llama
```

The browser tests fail a reply that is off topic or repeats itself, and `diskformer-chat` fails any reply that differs
from llama.cpp's by one character. Only Apple silicon (M4 Pro, Chrome) has run them. Not tested yet: audio input blocks, abort mid-reply, speech / classification / depth / segmentation in a browser, any
machine other than the M4 Pro, and Qwen3.6 35B-A3B (it needs more than that machine's 24 GB).

## License

MIT. Spec and agent contract: [SPEC.md](SPEC.md) · depends on [diskformer.js](https://github.com/NakliTechie/diskformer.js)
· [kohra](https://github.com/NakliTechie/kohra) · [LocalMind](https://github.com/NakliTechie/LocalMind)
