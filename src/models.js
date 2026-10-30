// Models that pass a browser test through anyformer, grouped by what they do. Pass an entry to load() in place of a
// repo id; it carries the task and, where the back end needs one, the budget. DEFAULT is the smallest download that
// passed. `test` names the page under test/browser/ that proves the entry; test/models.test.js checks it exists.
// `diffusion: true` marks a text-diffusion model (the reply denoises in blocks); `experimental: true` marks an entry its
// app shows as opt-in until that app's own checks pass; `audioOut: true` marks a model that can also answer in speech
// (generate({ speech: true })).

const M = (repo, backend, task, test, extra = {}) => ({ repo, backend, task, test, ...extra });

const QWEN2_5_0_5B = M('onnx-community/Qwen2.5-0.5B-Instruct', 'transformers', 'text-generation', 'transformers', { downloadBytes: 786e6 });
const TERNARY_BONSAI_4B = M('onnx-community/Ternary-Bonsai-4B-ONNX', 'transformers', 'text-generation', 'transformers', { downloadBytes: 1.16e9 });
const LFM2_5_230M = M('LiquidAI/LFM2.5-230M-GGUF', 'localmind', 'text-generation', 'chat', { downloadBytes: 149e6 });
const LFM2_5_1_2B = M('LiquidAI/LFM2.5-1.2B-Instruct-GGUF', 'localmind', 'text-generation', 'chat', { downloadBytes: 696e6 });
const GEMMA4_E2B = M('google/gemma-4-E2B-it-qat-mobile-transformers', 'localmind', 'text-generation', 'chat', { downloadBytes: 2.46e9 });
const GEMMA4_E4B = M('google/gemma-4-E4B-it-qat-mobile-transformers', 'localmind', 'text-generation', 'chat', { budget: 4e9, downloadBytes: 3.53e9 });
const TERNARY_BONSAI_2_27B = M('prism-ml/Ternary-Bonsai-2-27B-gguf', 'localmind', 'text-generation', 'chat', { budget: 7e9, downloadBytes: 5.94e9 });
const QWEN3_5_4B = M('onnx-community/Qwen3.5-4B-ONNX-OPT', 'transformers', 'text-generation', 'vision', { budget: 3.1e9, downloadBytes: 3.0e9 });
const GEMMA4_E2B_ONNX = M('onnx-community/gemma-4-E2B-it-ONNX', 'transformers', 'text-generation', 'vision', { budget: 3.5e9, downloadBytes: 3.38e9 });
const GEMMA4_E4B_ONNX = M('onnx-community/gemma-4-E4B-it-ONNX', 'transformers', 'text-generation', 'transformers', { budget: 5.3e9, downloadBytes: 5.18e9 });
const MINICPM5_2B = M('openbmb/MiniCPM5-2B-GGUF', 'wllama', 'text-generation', 'chat', { downloadBytes: 1.56e9 });
const QWEN3_0_6B_MDLM = M('naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX', 'kohra', 'text-generation', 'kohra', { downloadBytes: 1.54e9, diffusion: true });
const NEMOTRON_DIFFUSION_3B = M('naklitechie/Nemotron-Labs-Diffusion-3B-ONNX', 'kohra', 'text-generation', 'kohra', { downloadBytes: 2.77e9, diffusion: true, experimental: true });
const GEMMA4_26B_A4B = M('google/gemma-4-26B-A4B-it-qat-q4_0-gguf', 'diskformer', 'text-generation', 'diskformer-chat', { budget: 2.5e9, downloadBytes: 14.44e9 });
const LFM3_1_7B = M('LiquidAI/LFM3-1.7B-ONNX', 'transformers', 'text-generation', 'chat', { downloadBytes: 1.63e9 });
// transformers.js 4.3.2 has no lfm3_omni class, so diskformer's hand engine runs it; swap the entry to transformers.js
// when it adds the model.
const LFM3_OMNI_2B = M('LiquidAI/LFM3-Omni-2B-ONNX', 'diskformer-onnx', 'any-to-any', 'omni', { downloadBytes: 2.14e9, audioOut: true });
const NEMOTRON_3_5_BOLT_E4B = M('danyalmalik/NVIDIA-Nemotron-3.5-Bolt-E4B-ONNX-WebGPU', 'transformers', 'text-generation', 'transformers', { budget: 5e9, downloadBytes: 4.94e9 });
// A tech demo, not a gated model: 231 GB on disk (4 IQ1_M parts plus the 110.6 GB Engram side file), about 0.4 tok/s.
// diskformer checked one hand-written prompt through complete(); generate() renders the GGUF's chat template, which no
// browser run has checked.
const DEEPSEEK_V4_1_FLASH = M('ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF', 'diskformer-llama', 'text-generation', 'llama-gate', { budget: 8e9, downloadBytes: 231316411072, experimental: true });
const MINILM_L6 = M('Xenova/all-MiniLM-L6-v2', 'transformers', 'feature-extraction', 'embed', { downloadBytes: 90e6 });
const DETR_RESNET_50 = M('Xenova/detr-resnet-50', 'transformers', 'object-detection', 'detect', { downloadBytes: 167e6 });

export const MODELS = {
  chat: {
    DEFAULT: QWEN2_5_0_5B,
    QWEN2_5_0_5B,
    /** Ternary weights (q2 graph), 4B parameters in 1.16 GB. */
    TERNARY_BONSAI_4B,
    /** LocalMind's hand-written LFM2 kernels: 149 MB, the fastest chat here. */
    LFM2_5_230M,
    /** The same LFM2 engine on the 1.2B GGUF. */
    LFM2_5_1_2B,
    /** Gemma 4 E2B, Google's QAT-mobile export, on LocalMind's Gemma kernels. */
    GEMMA4_E2B,
    /** Gemma 4 E4B on the same kernels; the whole 3.5 GB stays on the GPU. */
    GEMMA4_E4B,
    /** 27B parameters at 1.6 bits (Prism ternary), 5.9 GB on the GPU. */
    TERNARY_BONSAI_2_27B,
    /** llama.cpp in WebAssembly (wllama): WebGPU when the budget holds it, the CPU otherwise. */
    MINICPM5_2B,
    /** Text diffusion: the whole reply denoises at once (kohra). */
    QWEN3_0_6B_MDLM,
    /** NVIDIA's Nemotron-Labs-Diffusion-3B, block diffusion with a KV cache, q4 (kohra). */
    NEMOTRON_DIFFUSION_3B,
    /** 26B parameters in 2.5 GB of GPU memory; 14.4 GB on disk (diskformer). */
    GEMMA4_26B_A4B,
    /** NVIDIA's Nemotron 3.5 Bolt E4B (dense, text only), a community q4f16 ONNX export in Gemma 4 E4B's graph. */
    NEMOTRON_3_5_BOLT_E4B,
    /** LiquidAI's LFM3 1.7B (dense), LiquidAI's own ONNX export; thinking off unless a request asks for it. */
    LFM3_1_7B,
    /** LiquidAI's LFM3-Omni-2B: text, image and speech in (16 kHz, at most 30 s a clip); text or text and speech out (diskformer's engine). */
    LFM3_OMNI_2B,
    /** DeepSeek V4.1 Flash (763B MoE), a tech demo on diskformer's llama.cpp path: 231 GB on disk, very slow. */
    DEEPSEEK_V4_1_FLASH,
  },
  /** Chat with images in the messages: content [{ type: 'image', image }, { type: 'text', text }]. Gemma 4 takes audio too. */
  vision: { DEFAULT: QWEN3_5_4B, QWEN3_5_4B, GEMMA4_E2B_ONNX, GEMMA4_E4B_ONNX },
  embedding: { DEFAULT: MINILM_L6, MINILM_L6 },
  objectDetection: { DEFAULT: DETR_RESNET_50, DETR_RESNET_50 },
};
