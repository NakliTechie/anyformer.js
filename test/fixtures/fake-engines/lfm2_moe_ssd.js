// A stand-in for diskformer's engines/lfm2_moe_ssd.js: records loads and adapter calls.
export const calls = [];
export class Lfm2MoeSsd {
  static async load(_, opts) { calls.push({ type: 'load', opts }); return new Lfm2MoeSsd(); }
  async loadAdapter(url, opts) { calls.push({ type: 'loadAdapter', url, opts }); return { id: 'pi' }; }
  async setAdapter(a) { calls.push({ type: 'setAdapter', adapter: a }); }
  get alwaysThinks() { return true; }
  get thinkInPrompt() { return true; }
  async *generate(_, o) { calls.push({ type: 'generate', o }); yield { text: '\nPlan.', token: 1, tokens: 1 }; yield { text: '\nPlan.\n</think>\n\nAnswer.', token: 2, tokens: 2 }; }
  dispose() { calls.push({ type: 'dispose' }); }
}
