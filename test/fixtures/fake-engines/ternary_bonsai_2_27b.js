// A stand-in for diskformer's Ternary Bonsai 2 engine: records template args and generate options (test/localmind.test.js).
export const log = { loads: [], gens: [] };
export class TernaryBonsai2 {
  static async load(repo, opts) { log.loads.push(opts); return new TernaryBonsai2(); }
  constructor() { this.chatTemplateArgs = {}; this.model = { dflashRunner: null }; this.resets = 0; }
  async warmup() {}
  async *generate(messages, opts) { log.gens.push({ opts, args: { ...this.chatTemplateArgs } }); yield { token: 7, text: 'a' }; yield { token: 8, text: 'ab' }; }
  reset() { this.resets++; }
  dispose() {}
}
