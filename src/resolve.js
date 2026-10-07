// Ask every back end whether it can run the model under the budget, then rank the ones that can.
//
// Back-end contract (each back end is a plain object):
//   id        string, stable, e.g. 'diskformer'
//   canRun(model, caps, { budget }) → { ok: true, estimate } | { ok: false, reason }
//             estimate = { gpuBytes, diskBytes, tokPerSec | null }. No network, no GPU allocation: it reads the
//             descriptor and caps only. A throw counts as { ok: false } with the message as the reason.
//   load(model, { budget, caps, onProgress, signal }) → session
//             session = { generate(messages, opts) → async iterable of { text, tokens }, dispose() }
//             The back end applies the model's own chat template.

export async function resolve(model, caps, backends, { budget } = {}) {
  const candidates = [];
  for (const [order, b] of backends.entries()) {
    let r;
    try { r = await b.canRun(model, caps, { budget }); } catch (e) { r = { ok: false, reason: `canRun threw: ${e.message}` }; }
    candidates.push(r.ok ? { backend: b.id, order, ok: true, estimate: r.estimate } : { backend: b.id, order, ok: false, reason: r.reason });
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
