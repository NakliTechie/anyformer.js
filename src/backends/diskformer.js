// diskformer: MoE models larger than GPU memory. Weights stay in OPFS; the GPU holds the dense layers, the KV cache
// and an expert cache sized by the budget. Engines load from the CDN at load() time, at the commit src/sources.js pins;
// set diskformerBackend.source to load another copy.

import { AnyformerError } from '../errors.js';
import { SOURCES } from '../sources.js';

// The GGUFs diskformer's engines are pinned to (GEMMA4_26B_A4B, QWEN36_35B_A3B in engines/*.js), with the smallest budget each
// has passed its gate at and the measured decode speeds (diskformer README, M4 Pro, 2026-10-05).
export const DISKFORMER_MODELS = [
  {
    repo: 'google/gemma-4-26B-A4B-it-qat-q4_0-gguf', file: 'gemma-4-26B_q4_0-it.gguf', size: 14439363584,
    engine: 'gemma4_moe_ssd.js', Engine: 'Gemma4MoeSsd', minBudget: 2.5e9,
    // LoRA at run time (v0.6.20+, gated against llama-server --lora on the same adapter file).
    adapters: true,
    speeds: [[2.5e9, 12.4], [4.3e9, 14.2], [7e9, 17.9]],
  },
  {
    repo: 'unsloth/Qwen3.6-35B-A3B-GGUF', file: 'Qwen3.6-35B-A3B-Q8_0.gguf', size: 36903140320,
    engine: 'qwen35_moe_ssd.js', Engine: 'Qwen35MoeSsd', minBudget: 4e9,
    // LoRA at run time (v0.6.20+, gated against llama-server --lora on the same adapter file).
    adapters: true,
    speeds: [[4e9, 9.2]],
  },
  // Qwen3.6's language model with a vision encoder (mmproj, ~1.1 GB, loaded on the first image by ensureVision).
  // From LocalMind's gate (plan/2026-10-14-minicpm-harness, 4 GiB pool, context 512): ~6.1 GB on the GPU (dense 1.59,
  // pool 4.29), 14.1 tok/s over short greedy continuations. The smallest working budget is unmeasured; the floor is
  // Qwen3.6's, the same language-model shape. Only this engine reads its images, and it ranks before diskformer-llama.
  {
    repo: 'openbmb/MiniCPM-V-4.7-35B-A3B-gguf', file: 'MiniCPM-V-4_7-35B-A3B-Q4_0.gguf', size: 19714550688,
    engine: 'qwen35_moe_ssd.js', Engine: 'MiniCpmV47Ssd', minBudget: 4e9,
    speeds: [[6.1e9, 14.1]],
  },
  // Clover-35B-A3B-Preview (Diffuspace, Apache-2.0): block diffusion on Qwen3.6-35B-A3B, text only. The engine
  // (engines/clover_moe_ssd.js CLOVER_35B_A3B, rev 32e22a0) fills a 32-token block over several denoising passes
  // (kohra's denoise module), so generate() yields once per committed block. Gated by diskformer (v0.6.15: oracle
  // and browser chat) at a 1 GB expert pool and maxCtx 1024; speed unmeasured. At maxCtx 1024 the engine fixes about
  // 2.0 GB (dense 1.46, KV and DeltaNet state 0.09, working buffers) and refuses under about 2.05 GB (diskformer's
  // reading of its budget formula); 4 GB, Qwen3.6's floor, leaves about 1.9 GB of experts.
  {
    repo: 'Diffuspace/Clover-35B-A3B-Preview-GGUF', file: 'Clover-35B-A3B-Preview-Q4_0.gguf', size: 19586909600,
    engine: 'clover_moe_ssd.js', Engine: 'CloverMoeSsd', minBudget: 4e9,
    speeds: [],
  },
  // From LocalMind's gates (M4 Pro, 4 GiB pool; plan/2026-10-22-flint-harness): Flint ~7.3 GB on the GPU at maxCtx
  // 8192 (dense 2.33, pool 4.29, KV 0.41, batch 0.20), 8.66 tok/s at lookahead 1. LFM2.5 24B ~16 tok/s at a 4 GiB
  // pool; its total GPU use was not recorded, so it carries no speed estimate. The smallest working budget of either
  // is unmeasured: 4 GB is a floor, and the engine refuses a smaller fit with the need. Both files are also on the
  // llama.cpp path's checked list; this back end ranks first (Flint on llama.cpp: 1.8–3.4 tok/s).
  {
    repo: 'RecursiveSI/flint-int4-gguf', file: 'flint-int4-q4_1.gguf', size: 25992966976,
    engine: 'flint_moe_ssd.js', Engine: 'FlintMoeSsd', minBudget: 4e9,
    speeds: [[7.3e9, 8.66]],
  },
  {
    repo: 'LiquidAI/LFM2.5-24B-A2B-GGUF', file: 'LFM2.5-24B-A2B-QAD-Q4_0.gguf', size: 13578190880,
    engine: 'lfm2_moe_ssd.js', Engine: 'Lfm2MoeSsd', minBudget: 4e9,
    // LoRA at run time (v0.6.19+: engine.loadAdapter / setAdapter; adapters on the routed experts are refused).
    adapters: true,
    speeds: [],
  },
  // Two parts, ingested as one file by the engine's splitFetch (engines/ling_moe_ssd.js LING_FLASH, rev 29edce5).
  // Budget and speed from LocalMind's gate (plan/2026-10-22-ling-harness/results.md): 2.51 GB of dense weights on
  // the GPU plus a 4.29 GB expert pool, 5.53 tok/s decode; the OPFS store is 80.7 GB.
  {
    repo: 'inclusionAI/Ling-3.0-flash-GGUF',
    parts: [
      { file: 'Q4_K_M/Ling-3.0-flash-Q4_K_M-00001-of-00002.gguf', size: 49995534208 },
      { file: 'Q4_K_M/Ling-3.0-flash-Q4_K_M-00002-of-00002.gguf', size: 27014610464 },
    ],
    size: 49995534208 + 27014610464, storeBytes: 80.7e9,
    engine: 'ling_moe_ssd.js', Engine: 'LingMoeSsd', minBudget: 6.8e9,
    speeds: [[6.8e9, 5.53]],
  },
];

// diskformer's engines refuse a budget with a plain Error ("GPU budget 2.50 GB is too small: … need 3.04 GB, and the
// expert cache at least 0.05 GB more"). Until they carry a code, read the two numbers to give BUDGET_TOO_SMALL a need.
function budgetError(e) {
  const m = /GPU budget [\d.]+ GB is too small\b.*?need ([\d.]+) GB.*?at least ([\d.]+) GB more/s.exec(e?.message || '');
  return m && new AnyformerError('BUDGET_TOO_SMALL', `diskformer: ${e.message}`, { cause: e, need: Math.ceil((+m[1] + +m[2]) * 1e9) });
}

const has = (model, file, size) => model.files.gguf.some((f) => f.name === file && f.size === size);
const match = (model) => DISKFORMER_MODELS.find((m) => m.repo === model.repo &&
  (m.parts ? m.parts.every((p) => has(model, p.file, p.size)) : has(model, m.file, m.size)));

// The speed measured at the largest budget not above this one.
const speedAt = (m, budget) => m.speeds.filter(([b]) => b <= budget).at(-1)?.[1] ?? null;

export { budgetError };

export const diskformerBackend = {
  id: 'diskformer',
  // Gemma 4's repo is tagged image-text-to-text; the engines run its text turns.
  tasks: ['text-generation', 'image-text-to-text'],
  source: SOURCES.diskformer,
  // Where the engine downloads its GGUF from; null is Hugging Face. A test points it at a local server.
  url: null,
  // Where an adapter GGUF downloads from.
  adapterUrl: (a) => `https://huggingface.co/${a.repo}/resolve/${a.revision || 'main'}/${a.file.split('/').map(encodeURIComponent).join('/')}`,
  // The engines that apply a LoRA at run time.
  adapters: (model) => !!match(model)?.adapters,

  canRun(model, caps, { budget, task }) {
    if (!model.formats.includes('gguf')) return { ok: false, code: 'format', reason: 'no GGUF in the repo' };
    if (task && !diskformerBackend.tasks.includes(task)) return { ok: false, code: 'task', reason: `task ${task}; diskformer engines run text generation only` };
    const m = match(model);
    if (!m) return { ok: false, code: 'model', reason: `no diskformer engine for this GGUF; engines cover ${DISKFORMER_MODELS.map((x) => x.repo).join(', ')}` };
    if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
    if (!caps.f16) return { ok: false, code: 'f16', reason: 'the GPU lacks shader-f16' };
    if (!caps.opfs) return { ok: false, code: 'opfs', reason: 'no OPFS (navigator.storage.getDirectory)' };
    if (budget < m.minBudget) return { ok: false, code: 'budget', need: m.minBudget, reason: `a ${(budget / 1e9).toFixed(1)} GB budget is under the ${m.minBudget / 1e9} GB this model has passed its gate at` };
    // Storage quota is not checked: Chrome raises it as the download writes, while the disk has room.
    return { ok: true, estimate: { gpuBytes: budget, diskBytes: m.storeBytes ?? m.size, tokPerSec: speedAt(m, budget) } };
  },

  // maxCtx sizes the KV cache, which stays on the GPU: the 2.5 GB minimum and the speeds above hold at 1024 tokens,
  // the context diskformer's chat gate runs at. A longer context needs a larger budget.
  // poolBytes: the expert cache size, passed to the engine; the budget then only admits the model (the engine sizes
  //   its pool from gpuBudgetBytes when given one, so it gets one or the other).
  // localFile: the GGUF as a File the user picked; the engine ingests it instead of downloading, and refuses a file
  //   whose size differs from the one it is pinned to.
  // adapter: { repo, file, revision, scale } loads onto the GPU beside the model and is on for every reply;
  //   session.setAdapter(false / true) switches it off and on again with no reload.
  async load(model, { budget, maxCtx = 1024, poolBytes, localFile, adapter = null, onProgress, signal }) {
    const m = match(model);
    const mod = await import(`${diskformerBackend.source}/${m.engine}`);
    let engine;
    try {
      engine = await mod[m.Engine].load(null, {
        // The engines' own default root, named so the store module (src/store.js) lists the same folder.
        ...(poolBytes ? { poolBytes } : { gpuBudgetBytes: budget }), maxCtx, signal, root: 'localmind-ssd',
        ...(localFile && { localFile }),
        ...(diskformerBackend.url && (m.parts ? { urls: m.parts.map((p) => diskformerBackend.url(model, { ...m, file: p.file })) } : { url: diskformerBackend.url(model, m) })),
        onProgress: onProgress && ((e) => onProgress({ backend: 'diskformer', ...e })),
      });
    } catch (e) { throw budgetError(e) || e; }
    let lora = null;
    if (adapter) {
      try {
        onProgress?.({ backend: 'diskformer', status: 'adapter', file: adapter.file });
        lora = await engine.loadAdapter(diskformerBackend.adapterUrl(adapter), { scale: adapter.scale ?? 1 });
        await engine.setAdapter(lora);
      } catch (e) { await engine.dispose?.(); throw e; }
    }
    // Where the model's reasoning starts and ends, as token ids, so a caller can split it out without parsing text.
    const open = engine.thinkOpenTokenId ?? null;
    return {
      task: 'text-generation',
      engine,
      think: open === null ? null : {
        openId: open, closeId: engine.thinkCloseTokenId ?? null, inPrompt: engine.thinkInPrompt ?? null,
        preamble: engine.thinkPreamble ?? null, always: engine.alwaysThinks ?? null,
      },
      // thinking: false (the default, as on every back end here) answers directly; true lets a model that reasons
      // (Gemma 4, Qwen3.6, Ling) think first. Image blocks ({ type: 'image', rgb, width, height }) pass through to
      // an engine that reads them.
      // A model that always reasons with the block opened by its prompt (LFM2.5-24B: engine.alwaysThinks and
      // thinkInPrompt) cannot answer directly, so with thinking off its reasoning moves to `reasoning` and `text` holds
      // only what follows </think> (empty until then).
      async *generate(messages, { maxNewTokens = 512, signal, thinking = false } = {}) {
        const split = thinking !== true && engine.alwaysThinks === true && engine.thinkInPrompt === true;
        for await (const r of engine.generate(messages, { maxNewTokens, signal, enableThinking: thinking === true })) {
          if (!split) { yield { text: r.text, token: r.token, tokens: r.tokens }; continue; }
          const at = r.text.indexOf('</think>');
          yield at < 0 ? { text: '', reasoning: r.text, token: r.token, tokens: r.tokens }
            : { text: r.text.slice(at + 8).replace(/^\s+/, ''), reasoning: r.text.slice(0, at), token: r.token, tokens: r.tokens };
        }
      },
      // Checkpoints a system prompt after load, so the first question does not wait for it (Qwen3.6, MiniCPM-V).
      ...(typeof engine.prime === 'function' && { prime: (system) => engine.prime(system) }),
      // Loads the vision tower, for an engine that reads images (MiniCPM-V).
      ...(typeof engine.ensureVision === 'function' && { ensureVision: (o) => engine.ensureVision(o) }),
      ...(lora && { setAdapter: (on) => engine.setAdapter(on ? lora : null) }),
      dispose: () => engine.dispose(),
    };
  },
};
