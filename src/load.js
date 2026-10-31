import { probe } from './probe.js';
import { describe } from './describe.js';
import { resolve } from './resolve.js';
import { transformersBackend } from './backends/transformers.js';
import { kohraBackend } from './backends/kohra.js';
import { diskformerBackend } from './backends/diskformer.js';
import { diskformerLlamaBackend } from './backends/diskformer-llama.js';
import { localmindBackend } from './backends/localmind.js';
import { wllamaBackend } from './backends/wllama.js';
import { diskformerOnnxBackend } from './backends/diskformer-onnx.js';
import { AnyformerError, wrap } from './errors.js';
import { toolFormat, withTools, parseToolCalls } from './tool-calls.js';

// Back ends land here as each one passes its browser test (test/browser/). The list order breaks ties: a
// diskformer hand engine carries a measured speed and outranks diskformer-llama for the same model; diskformer-llama
// comes before wllama because it pages experts instead of holding the whole model on the GPU. diskformer-onnx runs only
// the ONNX architectures transformers.js has no class for, so transformers.js wins once it adds one.
export const BACKENDS = [transformersBackend, diskformerOnnxBackend, kohraBackend, diskformerBackend, localmindBackend, diskformerLlamaBackend, wllamaBackend];

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
// file: a weights file in the repo to load (a GGUF name); null lets each back end pick.
// repo: a Hub id, or an entry from MODELS (its task, budget and adapter apply unless opts set them).
// adapter: a LoRA GGUF applied over the model, { repo, file, revision, size, scale }; only a back end whose
// `adapters` is true, or a function that says yes for the model, can apply one (diskformer's LFM2.5 engine at run time,
// diskformer-llama), so the others drop out of the plan.
const entry = (repo, opts) => (typeof repo === 'object' && repo?.repo
  ? [repo.repo, { ...(repo.task && { task: repo.task }), ...(repo.budget && { budget: repo.budget }), ...(repo.adapter && { adapter: repo.adapter }), ...opts }]
  : [repo, opts]);

export async function plan(repoOrEntry, opts = {}) {
  let [repo, { budget = DEFAULT_BUDGET, task = null, file = null, backends = BACKENDS, caps, model, fetch, adapter = null }] = entry(repoOrEntry, opts);
  caps ??= await probe();
  model ??= await describe(repo, fetch ? { fetch } : {});
  if (adapter) backends = backends.filter((b) => (typeof b.adapters === 'function' ? b.adapters(model) : b.adapters));
  return { model, caps, budget, task, file, ...(adapter && { adapter }), ...(await resolve(model, caps, backends, { budget, task, file })) };
}

export async function load(repoOrEntry, options = {}) {
  const [repo, opts] = entry(repoOrEntry, options);
  const { backends = BACKENDS, maxCtx, dtype, poolBytes, localFile, fetch: engineFetch, ple, speculative, speech, onProgress, signal } = opts;
  let p;
  try { p = await plan(repo, opts); } catch (e) { throw wrap(e, 'LOAD_FAILED', signal, `${repo}: `); }
  if (!p.chosen) throw noBackend(p);
  const backend = backends.find((b) => b.id === p.chosen);
  let session;
  try {
    session = await backend.load(p.model, { budget: p.budget, task: p.task, file: p.file, adapter: p.adapter, caps: p.caps, maxCtx, dtype, poolBytes, localFile, fetch: engineFetch, ple, speculative, speech, onProgress, signal });
  } catch (e) { throw wrap(e, 'LOAD_FAILED', signal, `${backend.id}: `); }
  // Added onto serialize()'s object, not spread: a spread would read its getters once and freeze them.
  return Object.assign(serialize(session, p.model.architecture), { backend: backend.id, task: session.task || p.task || p.model.task || null, ...(session.device && { device: session.device }), plan: p });
}

// content: a string, or an array of { type, … } blocks (text, image, audio) for multimodal models.
const validContent = (c) => typeof c === 'string' || (Array.isArray(c) && c.every((b) => typeof b?.type === 'string'));
// An assistant turn that only calls tools may have no content.
const validMessages = (messages) => Array.isArray(messages) && messages.every((m) => typeof m?.role === 'string' &&
  (validContent(m?.content) || (m.role === 'assistant' && Array.isArray(m.tool_calls) && (m.content == null || m.content === ''))));

// One call at a time per session: two overlapping generates would share one KV cache. A call waits for the previous
// one to finish (or for its caller to stop iterating). dispose() waits for the running call, then frees the GPU;
// calls after it throw DISPOSED.
// tools: OpenAI-style tool definitions. A back end whose chat template takes them (templateTools) gets them as they are;
// for the others the conversation carries them as text in the model family's format (src/tool-calls.js). Either way a
// chunk whose text holds complete tool calls carries them as `toolCalls` (OpenAI tool_calls) and its text without them.
function serialize(session, architecture) {
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
    if (!validMessages(messages)) throw new AnyformerError('INVALID_ARGUMENT', 'messages must be an array of { role, content }, content a string or an array of { type } blocks');
    const release = await turn();
    try {
      if (disposed) throw gone();
      const tools = o.tools?.length ? o.tools : null;
      const traffic = messages.some((m) => m.role === 'tool' || m.tool_calls?.length);
      const asText = (tools || traffic) && !session.templateTools;
      const msgs = asText ? withTools(messages, tools, toolFormat(architecture)) : messages;
      const { tools: _, ...rest } = o;
      for await (const chunk of session.generate(msgs, asText ? rest : o)) {
        const found = tools && typeof chunk.text === 'string' ? parseToolCalls(chunk.text, toolFormat(architecture)) : null;
        yield found?.toolCalls.length ? { ...chunk, text: found.content, toolCalls: found.toolCalls } : chunk;
      }
    } catch (e) {
      throw wrap(e, 'EXECUTION_FAILED', o.signal);
    } finally {
      release();
    }
  };
  // A call that waits its turn like generate(): prime and ensureVision share the engine with it.
  const queued = (fn) => async (...args) => {
    if (disposed) throw gone();
    const release = await turn();
    try {
      if (disposed) throw gone();
      return await fn(...args);
    } catch (e) {
      throw wrap(e, 'EXECUTION_FAILED');
    } finally {
      release();
    }
  };
  const out = {
    ...(generate && { generate }),
    // Where reasoning starts and ends as token ids ({ openId, closeId, inPrompt, preamble, always }), or null.
    ...('think' in session && { think: session.think }),
    ...(session.prime && { prime: queued(session.prime) }),
    ...(session.ensureVision && { ensureVision: queued(session.ensureVision) }),
    ...(session.reset && { reset: queued(session.reset) }),
    // Switches a loaded adapter on (true) or off (false) for the next reply, with no reload (diskformer's LFM2.5 engine).
    ...(session.setAdapter && { setAdapter: queued(session.setAdapter) }),
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
  // Read live from the back end's session, never copied: the engine instance it built (diskformer and localmind back
  // ends; read-only, for token-level work a caller already does with the engine; calls on it bypass the queue),
  // Gemma 4's PLE mode, Bonsai's speculative decoding state, which changes after load, and diskformer-onnx's baseUrl
  // (its Cache Storage keys).
  for (const k of ['engine', 'pleMode', 'speculative', 'baseUrl']) if (k in session) Object.defineProperty(out, k, { get: () => session[k], enumerable: true });
  return out;
}
