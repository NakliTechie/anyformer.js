// transformers.js's pipeline() call shape over anyformer's resolver, so `import { pipeline } from 'anyformer'` replaces
// `from '@huggingface/transformers'` for the calls below. Any repo anyformer can plan works: GGUF on diskformer,
// text diffusion on kohra, ONNX on transformers.js.
//
//   const gen = await pipeline('text-generation', 'google/gemma-4-26B-A4B-it-qat-q4_0-gguf', { budget: 2.5e9 });
//   const [{ generated_text }] = await gen(messages, { max_new_tokens: 128 });   // messages in → messages + reply out
//   const [{ generated_text: text }] = await gen('Once upon a time');            // a string in → the reply string out
//
// Other tasks return what the session's run() returns (feature-extraction: plain arrays, not a Tensor).
// Options: budget, maxCtx, backends, progress_callback (transformers.js's name) or onProgress, signal.

import { load } from './load.js';
import { AnyformerError } from './errors.js';

export async function pipeline(task, repo, options = {}) {
  if (!repo) throw new AnyformerError('INVALID_ARGUMENT', 'pipeline(task, repo): anyformer needs a repo id or a MODELS entry');
  const { progress_callback, ...opts } = options;
  const session = await load(repo, { ...opts, task, onProgress: opts.onProgress || progress_callback });
  const call = task === 'text-generation'
    ? async (input, o = {}) => {
      const messages = typeof input === 'string' ? [{ role: 'user', content: input }] : input;
      const reply = await session.run(messages, { maxNewTokens: o.max_new_tokens ?? o.maxNewTokens ?? 256, signal: o.signal });
      return [{ generated_text: typeof input === 'string' ? reply.text : [...messages, { role: 'assistant', content: reply.text }] }];
    }
    : (input, o) => session.run(input, o);
  return Object.assign(call, { task, backend: session.backend, plan: session.plan, session, dispose: () => session.dispose() });
}
