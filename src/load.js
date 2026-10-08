import { probe } from './probe.js';
import { describe } from './describe.js';
import { resolve } from './resolve.js';
import { transformersBackend } from './backends/transformers.js';
import { kohraBackend } from './backends/kohra.js';
import { diskformerBackend } from './backends/diskformer.js';

// Back ends land here as each one passes its browser test (test/browser/). The list order breaks ties.
export const BACKENDS = [transformersBackend, kohraBackend, diskformerBackend];

// 2.5 GB: the smallest budget diskformer's chat gate passes at. Browsers do not report GPU memory, so a caller that
// knows more passes { budget }.
export const DEFAULT_BUDGET = 2.5e9;

export class NoBackendError extends Error {
  constructor(plan) {
    const lines = plan.candidates.map((c) => `  ${c.backend} [${c.code}]: ${c.reason}`);
    super(`no back end can run ${plan.model.repo} at a ${(plan.budget / 1e9).toFixed(1)} GB budget` +
      (lines.length ? `:\n${lines.join('\n')}` : ': no back ends registered'));
    this.name = 'NoBackendError';
    this.plan = plan;
  }
}

// What load() would do, without loading: the descriptor, the caps, and every back end's answer.
export async function plan(repo, { budget = DEFAULT_BUDGET, backends = BACKENDS, caps, model, fetch } = {}) {
  caps ??= await probe();
  model ??= await describe(repo, fetch ? { fetch } : {});
  return { model, caps, budget, ...(await resolve(model, caps, backends, { budget })) };
}

export async function load(repo, opts = {}) {
  const { backends = BACKENDS, maxCtx, onProgress, signal } = opts;
  const p = await plan(repo, opts);
  if (!p.chosen) throw new NoBackendError(p);
  const backend = backends.find((b) => b.id === p.chosen);
  const session = await backend.load(p.model, { budget: p.budget, caps: p.caps, maxCtx, onProgress, signal });
  return {
    backend: backend.id,
    plan: p,
    generate: (messages, o) => session.generate(messages, o),
    dispose: () => session.dispose?.(),
  };
}
