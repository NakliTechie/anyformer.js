// kohra: text-diffusion LMs (MDLM, BD3LM). kohra.js runs the denoising loop over onnxruntime-web; it loads from the
// CDN at load() time, at the commit src/sources.js pins; set kohraBackend.source to load another copy.

import { graphBytes } from './transformers.js';
import { SOURCES } from '../sources.js';

const FP16 = 'onnx/model_fp16_fused.onnx';
// The q4 graph decodes on WebGPU only with RTN packing and this ORT-web dev build (kohra README, "Known constraints").
const Q4 = 'onnx/model_q4f16_rtn_sym.onnx';
const Q4_ORT = '1.26.0-dev.20260416-b7804b056c';

// Models with their own graph and tokens, keyed by config.model_type (kohra index.html REGISTRY at the pinned commit).
// Nemotron-Labs-Diffusion-3B: a seeded block-KV graph (kohra.js finds its attention_mask input and denoises blocks
// from the prompt end), mask id 100, EOS 11, and Nemotron's template: a system turn, always, and an empty think block
// before each assistant reply.
const SPECIAL = {
  nemotron_labs_diffusion: {
    graph: 'onnx/model_kv_q4f16.onnx', opts: { kvCache: true, maskId: 100n, eosId: 11n }, blockCausal: false,
    // Fast-dLLM decoding: unmask every position at confidence 0.9 or more per step. Same answer in 34 forward passes
    // where fixed steps took 64 (kohra, primes prompt).
    threshold: 0.9,
    template: (messages) => {
      const [system, rest] = messages[0]?.role === 'system' ? [messages[0].content, messages.slice(1)] : ['', messages];
      return `<|im_start|>system\n${system}<|im_end|>\n` + rest.map((m) => (m.role === 'assistant'
        ? `<|im_start|>assistant\n<think></think>${m.content.trim()}<|im_end|>\n`
        : `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`)).join('') + '<|im_start|>assistant\n<think></think>';
    },
  },
};

const pickGraph = (model, caps) => {
  const have = new Set(model.files.onnx.map((f) => f.name));
  const special = SPECIAL[model.architecture];
  if (special) return have.has(special.graph) ? special.graph : null;
  return caps.f16 && have.has(FP16) ? FP16 : have.has(Q4) ? Q4 : null;
};

// kohra's own template covers one user turn; this is the same ChatML over the whole conversation.
const chatml = (messages) =>
  messages.map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`).join('') + '<|im_start|>assistant\n';

// Text blocks joined: kohra reads text only.
const text = (messages) => messages.map((m) => (typeof m.content === 'string' ? m : { ...m, content: m.content.map((b) => b.text ?? '').join('\n') }));

export const kohraBackend = {
  id: 'kohra',
  tasks: ['text-generation'],
  source: SOURCES.kohra,

  canRun(model, caps, { task } = {}) {
    if (!model.tags?.includes('text-diffusion')) return { ok: false, code: 'model', reason: 'not a text-diffusion model' };
    if (task && task !== 'text-generation') return { ok: false, code: 'task', reason: `task ${task}; kohra runs text-generation only` };
    if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
    const graph = pickGraph(model, caps);
    if (!graph) return { ok: false, code: 'format', reason: `no kohra graph (${SPECIAL[model.architecture]?.graph || `${FP16} or ${Q4}`}) in the repo` };
    const bytes = graphBytes(model, graph);
    return { ok: true, estimate: { gpuBytes: bytes, diskBytes: bytes, tokPerSec: null } };
  },

  async load(model, { caps, onProgress, signal }) {
    const graph = pickGraph(model, caps);
    const { DiffusionLM } = await import(kohraBackend.source);
    onProgress?.({ backend: 'kohra', status: 'loading', file: graph });
    const base = `https://huggingface.co/${model.repo}/resolve/${model.revision || 'main'}`;
    const special = SPECIAL[model.architecture];
    const lm = await DiffusionLM.from_pretrained({
      model: `${base}/${graph}`, tokenizer: model.repo, ...(graph === Q4 && { ortVersion: Q4_ORT }), ...special?.opts,
    });
    signal?.throwIfAborted();
    const blockCausal = special ? special.blockCausal : model.tags.includes('block-diffusion');
    const prompt = special?.template || chatml;
    return {
      task: 'text-generation',
      // Diffusion fills the whole block at once, so the reply arrives as one chunk.
      // threshold: unmask every position at that confidence per step (Fast-dLLM); null runs fixed steps.
      async *generate(messages, { maxNewTokens = 128, steps, threshold = special?.threshold ?? null, signal } = {}) {
        signal?.throwIfAborted();
        const out = await lm.generate(prompt(text(messages)), {
          maxNewTokens, steps: steps ?? maxNewTokens, threshold, blockCausal, chatTemplate: false, stripThink: true,
        });
        yield { text: out.text, tokens: out.tokens, tokPerSec: out.tokensPerSecond };
      },
      dispose: () => lm.session?.release?.(),
    };
  },
};
