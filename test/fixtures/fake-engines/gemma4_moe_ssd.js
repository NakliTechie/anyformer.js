// A stand-in for diskformer's Gemma engine: records the options generate() receives (test/backends.test.js).
export const calls = [];
export class Gemma4MoeSsd {
  static async load() { return new Gemma4MoeSsd(); }
  async *generate(messages, opts) { calls.push(opts); yield { text: 'ok', tokens: 1 }; }
  dispose() {}
}
