export const loads = [];
export class Gemma4Mobile {
  static async load(repo, opts) { loads.push(opts); const g = new Gemma4Mobile(); g.pleMode = opts.ple ? 'opfs' : 'resident'; return g; }
  async warmup() {}
  async *generate(messages) { yield { token: 1, text: messages[0].content }; }
  dispose() {}
}
