# Browser tests

Each page loads anyformer from the repo, runs one check in Chrome with WebGPU, and sets `window.result` (`ok` plus the
evidence) and the page title (PASS or FAIL). `run.mjs` serves the repo on 127.0.0.1, drives headless Chrome over CDP
and exits 0 on pass, 1 on fail, 2 on timeout.

| Page | Checks |
|---|---|
| `probe` | `probe()` reports WebGPU, shader-f16 and OPFS |
| `transformers` | onnx-community/Qwen2.5-0.5B-Instruct replies in English through the transformers.js back end |
| `kohra` | naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX replies in English through the kohra back end |
| `runntime-embed` | Xenova/all-MiniLM-L6-v2 through runntime: 384-dim vectors, the two cat sentences closer than the invoice |
| `runntime-detect` | onnx-community/yolo26n-ONNX through runntime finds both cats in transformers.js's `cats.jpg` |
| `diskformer-chat` | Gemma 4 26B-A4B at a 2.5 GB budget: 9/9 replies identical to llama.cpp b9830 (diskformer's chat gate) |

```bash
npm run test:browser -- probe
```

kohra and diskformer load at their current `main` commit. `diskformer-chat` uses port 8192 and the profile
`~/.cache/anyformer-chat-profile`, so the 14.4 GB download happens once; `--budget 4.3` changes the budget.
