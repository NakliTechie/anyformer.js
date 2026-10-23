// LocalMind's hand-written WebGPU engines: whole model resident on the GPU, kernels written per architecture. Each
// engine reads its config from the model file, so it runs any checkpoint of its architecture in a format it reads,
// not only the one LocalMind ships. They live in diskformer (engines/, since v0.6.4) and load from the CDN at load()
// time, at the commit src/sources.js pins; set localmindBackend.source to load another copy. The id stays 'localmind',
// the name callers and stored() already use, though no code loads from LocalMind's repo any more.
//
//   lfm2     src/lfm2_5.js           GGUF, general.architecture 'lfm2' (dense LFM2 / LFM2.5), Q4_0 Q8_0 Q4_K Q5_K Q6_K
//   gemma4   src/gemma-4-e2b.js      Google's QAT-mobile safetensors export of Gemma 4 (E2B, E4B): config.json-driven
//   qwen35   src/ternary_bonsai_2_27b.js  GGUF, 'qwen35' with Prism's ternary types (PTQ1_0, PQ2_0) only: a standard
//            K-quant (unsloth/Qwen3.5-4B-GGUF Q4_K_M) fails with "This feature is not included in this bundle".
// Speeds: LocalMind's quiet-machine engine bench (plan/bench-engines-quiet-2026-09-24.md, M4 Pro, 256 tokens, greedy).

import { SOURCES } from '../sources.js';

const ENGINES = {
  lfm2: { file: 'lfm2_5.js', Engine: 'Lfm2Mobile' },
  gemma4: { file: 'gemma-4-e2b.js', Engine: 'Gemma4Mobile' },
  qwen35: { file: 'ternary_bonsai_2_27b.js', Engine: 'TernaryBonsai2' },
};

// Quant preference per engine, by the GGUF file-name suffix. Plain Q4_0 first: it is LocalMind's default.
const LFM2_QUANTS = ['Q4_0', 'Q4_K_M', 'Q4_K_S', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0'];
const QWEN35_QUANTS = ['PTQ1_0', 'PQ2_0'];
const MEASURED = {
  'LiquidAI/LFM2.5-230M-GGUF': 1062.5,
  'google/gemma-4-E2B-it-qat-mobile-transformers': 171.3,
  'prism-ml/Ternary-Bonsai-2-27B-gguf': 27.2,
};

// The GGUF to load: the first quant in preference order. Projector (mmproj) files are vision towers, never the LM.
function pickGguf(model, quants) {
  const lms = model.files.gguf.filter((f) => !/mmproj/i.test(f.name));
  for (const q of quants) {
    const f = lms.find((x) => new RegExp(`[-_.]${q}\\.gguf$`, 'i').test(x.name) && !/QAD/i.test(x.name));
    if (f) return f;
  }
  return null;
}

// Which engine reads this repo, and the file and bytes it would load; null when none does.
function match(model) {
  if (model.formats.includes('gguf') && model.architecture === 'lfm2') {
    const f = pickGguf(model, LFM2_QUANTS);
    return f && { engine: 'lfm2', file: f.name, bytes: f.size };
  }
  if (model.formats.includes('gguf') && model.architecture === 'qwen35') {
    const f = pickGguf(model, QWEN35_QUANTS);
    return f && { engine: 'qwen35', file: f.name, bytes: f.size };
  }
  // The QAT-mobile export's tensor names (weight_scale, embedding_quantized …) are what the engine reads; a plain
  // Gemma 4 safetensors checkpoint has model_type gemma4 too but will not load, so the repo name decides.
  if (model.architecture === 'gemma4' && /-qat-mobile-transformers$/.test(model.repo)) {
    const f = model.files.safetensors.find((x) => x.name === 'model.safetensors');
    return f && { engine: 'gemma4', file: f.name, bytes: f.size };
  }
  return null;
}

export const localmindBackend = {
  id: 'localmind',
  tasks: ['text-generation'],
  source: SOURCES.diskformer,

  canRun(model, caps, { budget, task }) {
    const m = match(model);
    if (!m) return { ok: false, code: 'model', reason: 'no LocalMind engine reads this repo (lfm2 GGUF, qwen35 Prism GGUF, Gemma 4 QAT-mobile)' };
    if (task && task !== 'text-generation') return { ok: false, code: 'task', reason: `task ${task}; LocalMind's engines run text generation` };
    if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
    if (m.engine === 'qwen35' && !(caps.f16 && caps.subgroups)) return { ok: false, code: 'f16', reason: 'the ternary kernels need shader-f16 and subgroups' };
    // The whole model stays on the GPU, so the budget must hold the file.
    if (budget < m.bytes) return { ok: false, code: 'budget', need: m.bytes, reason: `${m.file} stays on the GPU whole (${(m.bytes / 1e9).toFixed(2)} GB) and the budget is ${(budget / 1e9).toFixed(2)} GB` };
    return { ok: true, estimate: { gpuBytes: m.bytes, diskBytes: m.bytes, tokPerSec: MEASURED[model.repo] ?? null } };
  },

  // Options beyond the shared ones, as LocalMind's workers use the engines:
  //   fetch        the engines' fetch (a caller's retrying fetch for multi-GB range reads)
  //   ple          Gemma 4: stream the per-layer-embedding table from OPFS (ple-opfs.js) instead of holding it
  //   speculative  Ternary Bonsai 2: { dflash: { block } } or { mtp: { depth, adaptive } }, attached after the model is
  //                ready, so the first reply never waits for the drafter's download (src/backends/speculative.js)
  async load(model, { maxCtx, fetch, ple, speculative, onProgress, signal }) {
    const m = match(model);
    const e = ENGINES[m.engine];
    const engines = localmindBackend.source;
    const mod = await import(`${engines}/${e.file}`);
    const say = onProgress && ((p) => onProgress({ backend: 'localmind', engine: m.engine, ...p }));
    let pleTable = null;
    if (ple && m.engine === 'gemma4') {
      const { createGemmaPle } = await import(`${engines}/ple-opfs.js`);
      pleTable = await createGemmaPle({ source: { repo: model.repo }, onStatus: (s) => say?.({ status: 'ple', ...s }) });
    }
    const opts = {
      revision: model.revision || 'main', signal, onProgress: say,
      ...(fetch && { fetch }),
      ...(pleTable && { ple: pleTable }),
      ...(m.engine !== 'gemma4' && { file: m.file }),
      ...(maxCtx && m.engine !== 'gemma4' && { maxLength: maxCtx }),
    };
    const engine = await mod[e.Engine].load(model.repo, opts);
    signal?.throwIfAborted();
    // Compile the kernels now, as LocalMind's workers do, so the first reply is not paying for it.
    say?.({ status: 'warmup' });
    await engine.warmup?.();
    // Gemma's template has no system turn; LocalMind folds the system prompt into the first user turn the same way.
    const prepare = m.engine !== 'gemma4' ? (msgs) => msgs : (msgs) => {
      const sys = msgs.filter((x) => x.role === 'system').map((x) => x.content).join('\n\n');
      const rest = msgs.filter((x) => x.role !== 'system');
      const i = rest.findIndex((x) => x.role === 'user');
      return sys && i >= 0 ? rest.map((x, k) => (k === i ? { ...x, content: `${sys}\n\n${x.content}` } : x)) : rest;
    };
    // Speculative decoding, Bonsai only, in the background: a reply that starts before it lands decodes plainly.
    const spec = { status: speculative && m.engine === 'qwen35' ? 'loading' : 'off', runner: null, error: null, disposed: false };
    if (spec.status === 'loading') {
      import('./speculative.js')
        .then(({ attachSpeculative }) => attachSpeculative(engine, speculative, { engines, fetch, signal, onProgress: (p) => say?.({ ...p, speculative: true }) }))
        .then((runner) => {
          // Disposed while the drafter downloaded: detach and free it rather than attach to a dead engine.
          if (spec.disposed) { engine.model.dflashRunner = null; runner.dispose?.(); return; }
          spec.runner = runner; spec.status = 'ready';
        }, (err) => { spec.error = err.message; spec.status = 'error'; });
    }
    let tokens = 0;
    return {
      task: 'text-generation',
      engine,
      // Gemma 4 with the OPFS table: 'resident' or the streamed mode the engine reports.
      ...(m.engine === 'gemma4' && { get pleMode() { return engine.pleMode ?? null; } }),
      // { status: 'off' | 'loading' | 'ready' | 'error', error, stats }; stats are the runner's counters.
      get speculative() { return { status: spec.status, error: spec.error, stats: spec.runner?.stats ?? null }; },
      // thinking: per request, through the qwen35 engine's template arguments (reasoning_effort 'medium', LocalMind's
      // serving recipe); the other two engines have no thinking switch. assistantPrefix: raw text the reply continues
      // from (Bonsai's think-budget continuation). Chunks carry the engine's token id.
      async *generate(messages, { maxNewTokens = 512, signal, thinking = false, assistantPrefix } = {}) {
        if ('chatTemplateArgs' in engine) engine.chatTemplateArgs = { ...engine.chatTemplateArgs, enable_thinking: thinking === true, reasoning_effort: 'medium' };
        tokens = 0;
        for await (const c of engine.generate(prepare(messages), { maxNewTokens, signal, ...(assistantPrefix != null && { assistantPrefix }) })) {
          if (c.token != null) tokens++;
          yield { text: c.text, token: c.token, tokens };
        }
      },
      // Drops the KV and recurrent state, for a new conversation or a shorter history.
      ...(typeof engine.reset === 'function' && { reset: () => engine.reset() }),
      dispose: () => {
        spec.disposed = true;
        if (spec.runner) { engine.model.dflashRunner = null; spec.runner.dispose?.(); }
        return engine.dispose?.();
      },
    };
  },
};

export { match as localmindMatch };
