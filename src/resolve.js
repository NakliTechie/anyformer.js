// Ask every back end whether it can run the model under the budget, then rank the ones that can.
//
// Back-end contract (each back end is a plain object):
//   id        string, stable, e.g. 'diskformer'
//   canRun(model, caps, { budget }) → { ok: true, estimate } | { ok: false, code, reason, need? }
//             estimate = { gpuBytes, diskBytes, tokPerSec | null }. No network, no GPU allocation: it reads the
//             descriptor and caps only. code is one of REFUSALS below; need is the budget that would pass, with
//             code 'budget'. A throw counts as code 'threw' with the message as the reason.
//   load(model, { budget, caps, maxCtx, onProgress, signal }) → session   (maxCtx: tokens; undefined = the back
//             end's default)
//             session = { generate(messages, opts) → async iterable of { text, tokens }, dispose() }
//             The back end applies the model's own chat template.

// Why a back end said no: a closed set, each with the move that changes the answer.
export const REFUSALS = {
  format: 'the repo has no weights in a format this back end reads; pick a repo in another format',
  model: 'this back end reads the format but not this model (architecture, task or file); pick another model',
  webgpu: 'no WebGPU adapter; use Chrome or Edge 121+ on a GPU with WebGPU',
  f16: 'the GPU lacks shader-f16; use a GPU that has it',
  opfs: 'no OPFS; use a browser with navigator.storage.getDirectory, outside private mode',
  budget: 'the GPU budget is too small; pass { budget } at least `need` bytes',
  threw: 'the back end crashed deciding; a bug in that back end, report it with the reason',
};

export async function resolve(model, caps, backends, { budget } = {}) {
  const candidates = [];
  for (const [order, b] of backends.entries()) {
    let r;
    try { r = await b.canRun(model, caps, { budget }); } catch (e) { r = { ok: false, code: 'threw', reason: `canRun threw: ${e.message}` }; }
    candidates.push(r.ok ? { backend: b.id, order, ok: true, estimate: r.estimate } : { backend: b.id, order, ok: false, code: r.code || 'model', reason: r.reason, ...(r.need && { need: r.need }) });
  }
  const ranked = candidates.filter((c) => c.ok).sort(rank);
  return { chosen: ranked[0]?.backend ?? null, ranked: ranked.map((c) => c.backend), candidates };
}

// A back end with a speed estimate ranks above one without; two estimates rank by tok/s; otherwise the order the
// caller listed them in, which is the default preference.
function rank(a, b) {
  const ta = a.estimate?.tokPerSec ?? null;
  const tb = b.estimate?.tokPerSec ?? null;
  if (ta !== null && tb !== null && ta !== tb) return tb - ta;
  if ((ta === null) !== (tb === null)) return ta === null ? 1 : -1;
  return a.order - b.order;
}
