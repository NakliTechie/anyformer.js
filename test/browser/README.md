# Browser tests

Each page loads anyformer from the repo, runs one check in Chrome with WebGPU, and sets `window.result` (`ok` plus the
evidence) and the page title (PASS or FAIL). `run.mjs` serves the repo on 127.0.0.1, drives headless Chrome over CDP
and exits 0 on pass, 1 on fail, 2 on timeout.

| Page | Checks |
|---|---|
| `probe` | `probe()` reports WebGPU, shader-f16 and OPFS |
| `transformers` | onnx-community/Qwen2.5-0.5B-Instruct replies in English through the transformers.js back end |
| `kohra` | naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX replies in English through the kohra back end |
| `embed` | Xenova/all-MiniLM-L6-v2: 384-dim vectors, the two cat sentences closer than the invoice |
| `detect` | Xenova/detr-resnet-50 finds both cats in transformers.js's `cats.jpg` |
| `store` | MiniLM appears in `stored()` after a load; the agent tool stages; `remove()` frees the same bytes |
| `stored` | lists the store, changes nothing (run with the chat profile to see Gemma) |
| `examples/chat/index.html --query "auto=1"` | the workbench loads the default chat model and answers; `model=GEMMA4_26B_A4B` with the chat port and profile runs Gemma |
| `chat` | one chat reply through anyformer answers on topic (`?repo=`, `?budget=`, `?backend=`) |
| `vision` | an image in a chat message: Qwen3.5 4B or Gemma 4 ONNX names the cats in `cats.jpg` |
| `diskformer-chat` | Gemma 4 26B-A4B at a 2.5 GB budget: 9/9 replies identical to llama.cpp b9830 (diskformer's chat gate) |

```bash
npm run test:browser -- probe
```

kohra, diskformer and LocalMind load at the commits `src/sources.js` pins; `--main` tests their current `main` instead.
To move the pins: `node scripts/pin-sources.mjs`, then this suite, then commit. `diskformer-chat` uses port 8192 and the profile
`~/.cache/anyformer-chat-profile`, so the 14.4 GB download happens once; `--budget 4.3` changes the budget. Other
pages get a throwaway profile, deleted when the run ends, unless `--profile` names one to keep.
