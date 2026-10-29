// A stand-in for transformers.js: records which class loads and with what options; generate streams one piece.
export const calls = [];
const net = (cls) => ({
  async from_pretrained(repo, opts) {
    calls.push({ type: 'load', cls, repo, opts });
    return {
      async generate(o) { calls.push({ type: 'generate', cls, inputs: o.input_ids }); o.streamer?.callback_function?.('391'); },
      dispose() {},
    };
  },
});
export const Gemma4ForCausalLM = net('Gemma4ForCausalLM');
export const Gemma4ForConditionalGeneration = net('Gemma4ForConditionalGeneration');
export const AutoTokenizer = {
  async from_pretrained(repo) {
    calls.push({ type: 'tokenizer', repo });
    return { apply_chat_template: (messages, o) => { calls.push({ type: 'template', messages, o }); return { input_ids: [1, 2, 3] }; } };
  },
};
export class TextStreamer { constructor(tokenizer, o) { this.callback_function = o.callback_function; } }
export class InterruptableStoppingCriteria { interrupt() {} }
// pipeline('text-generation'): records its options and each call's kwargs; the model's session asks for logits_to_keep
// as LiquidAI's LFM3 export does, and records the feeds it runs with.
export class Tensor { constructor(type, data, dims) { this.ort_tensor = { type, data, dims }; } }
export async function pipeline(task, repo, opts) {
  const session = { inputNames: ['input_ids', 'logits_to_keep'], async run(feeds) { calls.push({ type: 'run', feeds }); return {}; } };
  calls.push({ type: 'pipeline', task, repo, opts, session });
  const gen = async (messages, o) => { calls.push({ type: 'call', messages, o }); o.streamer?.callback_function?.('391'); };
  return Object.assign(gen, { tokenizer: {}, model: { sessions: { model: session } }, dispose() {} });
}
