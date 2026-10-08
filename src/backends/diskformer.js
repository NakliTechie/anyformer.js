// diskformer: MoE models larger than GPU memory. Weights stay in OPFS; the GPU holds the dense layers, the KV cache
// and an expert cache sized by the budget. Engines load from the CDN at load() time from diskformer's main branch,
// which moves with diskformer's own work; set diskformerBackend.source to pin a release or point at a local copy.

// The GGUFs diskformer's engines are pinned to (GEMMA4_26B_A4B, QWEN36_35B_A3B in engines/*.js), with the smallest budget each
// has passed its gate at and the measured decode speeds (diskformer README, M4 Pro, 2026-10-05).
export const DISKFORMER_MODELS = [
  {
    repo: 'google/gemma-4-26B-A4B-it-qat-q4_0-gguf', file: 'gemma-4-26B_q4_0-it.gguf', size: 14439363584,
    engine: 'gemma4_moe_ssd.js', Engine: 'Gemma4MoeSsd', minBudget: 2.5e9,
    speeds: [[2.5e9, 12.4], [4.3e9, 14.2], [7e9, 17.9]],
  },
  {
    repo: 'unsloth/Qwen3.6-35B-A3B-GGUF', file: 'Qwen3.6-35B-A3B-Q8_0.gguf', size: 36903140320,
    engine: 'qwen35_moe_ssd.js', Engine: 'Qwen35MoeSsd', minBudget: 4e9,
    speeds: [[4e9, 9.2]],
  },
];

const match = (model) => DISKFORMER_MODELS.find((m) => m.repo === model.repo &&
  model.files.gguf.some((f) => f.name === m.file && f.size === m.size));

// The speed measured at the largest budget not above this one.
const speedAt = (m, budget) => m.speeds.filter(([b]) => b <= budget).at(-1)?.[1] ?? null;

export const diskformerBackend = {
  id: 'diskformer',
  source: 'https://cdn.jsdelivr.net/gh/NakliTechie/diskformer.js@main/engines',

  canRun(model, caps, { budget }) {
    if (!model.formats.includes('gguf')) return { ok: false, code: 'format', reason: 'no GGUF in the repo' };
    const m = match(model);
    if (!m) return { ok: false, code: 'model', reason: `no diskformer engine for this GGUF; engines cover ${DISKFORMER_MODELS.map((x) => x.file).join(', ')}` };
    if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
    if (!caps.f16) return { ok: false, code: 'f16', reason: 'the GPU lacks shader-f16' };
    if (!caps.opfs) return { ok: false, code: 'opfs', reason: 'no OPFS (navigator.storage.getDirectory)' };
    if (budget < m.minBudget) return { ok: false, code: 'budget', need: m.minBudget, reason: `a ${(budget / 1e9).toFixed(1)} GB budget is under the ${m.minBudget / 1e9} GB this model has passed its gate at` };
    // Storage quota is not checked: Chrome raises it as the download writes, while the disk has room.
    return { ok: true, estimate: { gpuBytes: budget, diskBytes: m.size, tokPerSec: speedAt(m, budget) } };
  },

  async load(model, { budget, onProgress, signal }) {
    const m = match(model);
    const mod = await import(`${diskformerBackend.source}/${m.engine}`);
    const engine = await mod[m.Engine].load(null, {
      gpuBudgetBytes: budget, signal,
      onProgress: onProgress && ((e) => onProgress({ backend: 'diskformer', ...e })),
    });
    return {
      engine,
      async *generate(messages, { maxNewTokens = 512, signal } = {}) {
        for await (const r of engine.generate(messages, { maxNewTokens, signal, enableThinking: false }))
          yield { text: r.text, tokens: r.tokens };
      },
      dispose: () => engine.dispose(),
    };
  },
};
