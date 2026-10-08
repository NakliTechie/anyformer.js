// Models that pass a browser test through anyformer, grouped by what they do. Pass an entry to load() in place of a
// repo id; it carries the task and, where the back end needs one, the budget. DEFAULT is the smallest download that
// passed. `test` names the page under test/browser/ that proves the entry; test/models.test.js checks it exists.

const M = (repo, backend, task, test, extra = {}) => ({ repo, backend, task, test, ...extra });

const QWEN2_5_0_5B = M('onnx-community/Qwen2.5-0.5B-Instruct', 'transformers', 'text-generation', 'transformers', { downloadBytes: 786e6 });
const TERNARY_BONSAI_4B = M('onnx-community/Ternary-Bonsai-4B-ONNX', 'transformers', 'text-generation', 'transformers', { downloadBytes: 1.16e9 });
const QWEN3_0_6B_MDLM = M('naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX', 'kohra', 'text-generation', 'kohra', { downloadBytes: 1.54e9 });
const GEMMA4_26B_A4B = M('google/gemma-4-26B-A4B-it-qat-q4_0-gguf', 'diskformer', 'text-generation', 'diskformer-chat', { budget: 2.5e9, downloadBytes: 14.44e9 });
const MINILM_L6 = M('Xenova/all-MiniLM-L6-v2', 'transformers', 'feature-extraction', 'embed', { downloadBytes: 90e6 });
const DETR_RESNET_50 = M('Xenova/detr-resnet-50', 'transformers', 'object-detection', 'detect', { downloadBytes: 167e6 });

export const MODELS = {
  chat: {
    DEFAULT: QWEN2_5_0_5B,
    QWEN2_5_0_5B,
    /** Ternary weights (q2 graph), 4B parameters in 1.16 GB. */
    TERNARY_BONSAI_4B,
    /** Text diffusion: the whole reply denoises at once (kohra). */
    QWEN3_0_6B_MDLM,
    /** 26B parameters in 2.5 GB of GPU memory; 14.4 GB on disk (diskformer). */
    GEMMA4_26B_A4B,
  },
  embedding: { DEFAULT: MINILM_L6, MINILM_L6 },
  objectDetection: { DEFAULT: DETR_RESNET_50, DETR_RESNET_50 },
};
