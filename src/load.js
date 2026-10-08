import { probe } from './probe.js';
import { describe } from './describe.js';
import { resolve } from './resolve.js';
import { transformersBackend } from './backends/transformers.js';
import { kohraBackend } from './backends/kohra.js';
import { diskformerBackend } from './backends/diskformer.js';
import { localmindBackend } from './backends/localmind.js';
import { AnyformerError, wrap } from './errors.js';

// Back ends land here as each one passes its browser test (test/browser/). The list order breaks ties.
export const BACKENDS = [transformersBackend, kohraBackend, diskformerBackend, localmindBackend];

// 2.5 GB: the smallest budget diskformer's chat gate passes at. Browsers do not report GPU memory, so a caller that
// knows more passes { budget }.
export const DEFAULT_BUDGET = 2.5e9;

function noBackend(plan) {
  const lines = plan.candidates.map((c) => `  ${c.backend} [${c.code}]: ${c.reason}`);
  return new AnyformerError('NO_BACKEND', `no back end can run ${plan.model.repo} at a ${(plan.budget / 1e9).toFixed(1)} GB budget` +
    (lines.length ? `:\n${lines.join('\n')}` : ': no back ends registered'), { plan });
}

// What load() would do, without loading: the descriptor, the caps, and every back end's answer.
// task: a Hugging Face pipeline task ('text-generation', 'feature-extraction', …); null runs the repo's own.
// repo: a Hub id, or an entry from MODELS (its task and budget apply unless opts set them).
const entry = (repo, opts) => (typeof repo === 'object' && repo?.repo
  ? [repo.repo, { ...(repo.task && { task: repo.task }), ...(repo.budget && { budget: repo.budget }), ...opts }]
  : [repo, opts]);

export async function plan(repoOrEntry, opts = {}) {
  let [repo, { budget = DEFAULT_BUDGET, task = null, backends = BACKENDS, caps, model, fetch }] = entry(repoOrEntry, opts);
  caps ??= await probe();
  model ??= await describe(repo, fetch ? { fetch } : {});
  return { model, caps, budget, task, ...(await resolve(model, caps, backends, { budget, task })) };
}

export async function load(repoOrEntry, options = {}) {
  const [repo, opts] = entry(repoOrEntry, options);
  const { backends = BACKENDS, maxCtx, onProgress, signal } = opts;
  let p;
  try { p = await plan(repo, opts); } catch (e) { throw wrap(e, 'LOAD_FAILED', signal, `${repo}: `); }
  if (!p.chosen) throw noBackend(p);
  const backend = backends.find((b) => b.id === p.chosen);
  let session;
  try {
    session = await backend.load(p.model, { budget: p.budget, task: p.task, caps: p.caps, maxCtx, onProgress, signal });
  } catch (e) { throw wrap(e, 'LOAD_FAILED', signal, `${backend.id}: `); }
  return { backend: backend.id, task: session.task || p.task || p.model.task || null, plan: p, ...serialize(session) };
}

const validMessages = (messages) => Array.isArray(messages) && messages.every((m) => typeof m?.role === 'string' && typeof m?.content === 'string');

// One call at a time per session: two overlapping generates would share one KV cache. A call waits for the previous
// one to finish (or for its caller to stop iterating). dispose() waits for the running call, then frees the GPU;
// calls after it throw DISPOSED.
function serialize(session) {
  let tail = Promise.resolve(), disposed = false;
  const gone = () => new AnyformerError('DISPOSED', 'this session was disposed; load the model again');
  const turn = async () => {
    let release;
    const mine = new Promise((r) => { release = r; });
    const prev = tail;
    tail = prev.then(() => mine);
    await prev;
    return release;
  };
  const generate = session.generate && async function* (messages, o = {}) {
    if (disposed) throw gone();
    if (!validMessages(messages)) throw new AnyformerError('INVALID_ARGUMENT', 'messages must be an array of { role, content } strings');
    const release = await turn();
    try {
      if (disposed) throw gone();
      for await (const chunk of session.generate(messages, o)) yield chunk;
    } catch (e) {
      throw wrap(e, 'EXECUTION_FAILED', o.signal);
    } finally {
      release();
    }
  };
  return {
    ...(generate && { generate }),
    // The one call every task has. For text generation it is the last chunk of generate().
    async run(input, o = {}) {
      if (generate) { let last; for await (const c of generate(input, o)) last = c; return last; }
      if (disposed) throw gone();
      const release = await turn();
      try {
        if (disposed) throw gone();
        return await session.run(input, o);
      } catch (e) {
        throw wrap(e, 'EXECUTION_FAILED', o.signal);
      } finally {
        release();
      }
    },
    async dispose() {
      if (disposed) return;
      disposed = true;
      await tail;
      await session.dispose?.();
    },
  };
}
