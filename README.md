# anyformer.js

One call to run any model in a browser tab: it picks the back end that fits the model's format, architecture and your
GPU budget.

## Install

Not usable yet. `anyformer@0.0.1` on npm is a name placeholder with no code.

## What it is

A superset runtime for in-browser inference. `load(modelId, { budget })` probes the browser (WebGPU features, GPU
budget, OPFS quota, memory64), resolves the model to a back end, and exposes one generate / pipeline API:

- **transformers.js** — ONNX, breadth across modalities.
- **diskformer** — models larger than GPU memory: weights in OPFS, only what each step needs on the GPU.
- **llama.cpp on WebGPU + diskformer** — GGUF across llama.cpp's architectures (after the diskformer spike).
- **LocalMind's hand engines** — the fastest path on the few models they cover.

One model store (diskformer's OPFS layer) serves every back end, so a model downloads once.

## Context

Scaffolded 2026-10-07 from the diskformer.js proposal
(`diskformer.js/plan/coverage-and-superset-proposal-2026-10.md`, §4). It depends on diskformer; diskformer never depends
on it. Tier: Product.

## License

MIT
