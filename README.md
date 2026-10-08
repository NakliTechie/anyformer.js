<h1 align="center">anyformer.js</h1>

**One call to run a model in a browser tab: anyformer reads the repo, probes the browser, and picks the back end that
fits the model's format, architecture and your GPU budget.**

Plain ES modules, no build step, no dependencies to install. Chromium with WebGPU. Weights stay in the user's own
browser storage; no account, no server, no telemetry.

[![license](https://img.shields.io/badge/license-MIT-555?style=flat-square)](LICENSE)
[![dependencies](https://img.shields.io/badge/dependencies-none-555?style=flat-square)](package.json)

## Install

| How | Command |
|---|---|
| npm | not yet: `anyformer@0.0.1` is a name placeholder with no code |
| Copy | `index.js` and `src/`; nothing else is needed |

```js
import { load } from './anyformer/index.js';

const chat = await load('onnx-community/Qwen2.5-0.5B-Instruct', { budget: 2.5e9, onProgress: console.log });
for await (const { text } of chat.generate([{ role: 'user', content: 'What does a CPU do?' }], { maxNewTokens: 64 }))
  show(text);
```

Every session has `run(input)`: chat messages for text generation, a string or strings for embeddings, ImageData for
vision, 16 kHz samples for speech. `load(repo, { task })` picks a task when a repo has more than one. The first load
downloads the weights into the browser; later loads read them from there. Each back end's runtime loads
from jsDelivr only when that back end is chosen.

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
| `transformers` | transformers.js 4.3.1: text generation, embeddings, speech to text, detection, classification, depth, segmentation | ONNX repos in the onnx-community / Xenova layout |
| `kohra` | kohra, text-diffusion over onnxruntime-web | MDLM and BD3LM repos tagged `text-diffusion` |
| `diskformer` | diskformer engines: weights in OPFS, only what a step needs on the GPU | Gemma 4 26B-A4B (from 2.5 GB), Qwen3.6 35B-A3B (from 4 GB) |

kohra and diskformer load from their `main` branches; set `kohraBackend.source` or `diskformerBackend.source` to pin a
commit. `BACKENDS` lists the back ends `load()` asks by default: each joins after its browser test passes.

## When no back end can run it

`load()` throws an `AnyformerError` with code `NO_BACKEND` and the plan, naming each back end, its code and its
reason. Codes are a closed set, each with one
remedy in `REFUSALS`: `format`, `model`, `task`, `webgpu`, `f16`, `opfs`, `budget` (with `need`, the budget that would pass)
and `threw`. Every other failure is an `AnyformerError` too, with a code from `ERROR_CODES` (`LOAD_FAILED`,
`BUDGET_TOO_SMALL`, `ABORTED`, `EXECUTION_FAILED`, `INVALID_ARGUMENT`, `DISPOSED`); check it with
`isAnyformerError(e, code)`. A session runs one `generate()` at a time, in call order.

## Commands

```js
plan(repo, { budget, task })      // what would run it, and why the others would not; loads nothing
load(repo, { budget, task, maxCtx, onProgress, signal })  // → { backend, task, plan, run(input), generate?, dispose() }
probe()                           // WebGPU, shader-f16, subgroups, limits, OPFS quota, memory64
describe(repo)                    // Hub repo → formats, architecture, task, tags, files
resolve(model, caps, backends, { budget })  // rank back ends for a described model
registerTools()                   // publish the agent face (below)
```

Agents get the same calls as tools (`anyformer_probe`, `_plan`, `_load`, `_generate`, `_sessions`, `_unload`) through
WebMCP's `navigator.modelContext` where the browser has it and `window.anyformer.tools` otherwise; `anyformer_load`
loads nothing until the caller accepts the plan's disk bytes.

## Verify it yourself

```bash
npm test                                   # Node: probe, describe, resolver, back-end verdicts, agent face
npm run test:browser -- probe              # headless Chrome: WebGPU, f16, OPFS
npm run test:browser -- transformers       # Qwen2.5-0.5B answers on topic
npm run test:browser -- kohra              # Qwen3-0.6B MDLM answers on topic
npm run test:browser -- embed             # MiniLM embeddings rank a paraphrase above an unrelated sentence
npm run test:browser -- detect            # DETR finds both cats in cats.jpg
npm run test:browser -- diskformer-chat    # Gemma 4 26B-A4B at 2.5 GB: 9/9 replies identical to llama.cpp b9830
```

The browser tests fail a reply that is off topic or repeats itself, and `diskformer-chat` fails any reply that differs
from llama.cpp's by one character. Only Apple silicon (M4 Pro, Chrome) has run them.

## License

MIT. Spec and agent contract: [SPEC.md](SPEC.md) · depends on [diskformer.js](https://github.com/NakliTechie/diskformer.js)
· [kohra](https://github.com/NakliTechie/kohra)
