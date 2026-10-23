// A stand-in MTP module: two ranges of a 20-byte file; the runner reads one byte from each range.
export const MTP_HEAD = { url: 'https://example.test/head.gguf', fileSize: 20, ranges: [[2, 3], [10, 4]] };
export const created = [];
export class MTPRunner {
  static async create(engine, opts) {
    created.push({ depth: opts.depth, adaptive: opts.adaptive, a: [...await opts.head(3, 1)], b: [...await opts.head(12, 2)] });
    return new MTPRunner();
  }
  constructor() { this.stats = { cycles: 1, tokens: 2, accepted: 1, drafted: 2, plainTokens: 0 }; }
  dispose() {}
}
