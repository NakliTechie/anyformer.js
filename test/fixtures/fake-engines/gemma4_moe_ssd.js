// A stand-in for diskformer's Gemma engine: records what load() and generate() receive (test/backends.test.js).
export const calls = [];
export const loads = [];
export const primed = [];
export class Gemma4MoeSsd {
  static async load(id, opts) { loads.push(opts); return new Gemma4MoeSsd(); }
  async *generate(messages, opts) { calls.push(opts); yield { text: 'ok', token: 42, tokens: 1 }; }
  async prime(system) { primed.push(system); }
  get thinkOpenTokenId() { return 100; }
  get thinkCloseTokenId() { return 101; }
  get thinkInPrompt() { return false; }
  get thinkPreamble() { return 'thought\n'; }
  dispose() {}
}
