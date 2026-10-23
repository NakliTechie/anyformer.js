// A stand-in DFlash runner: records how it was created.
export const created = [];
export class DFlashRunner {
  static async create(engine, opts) { const head = await opts.drafter(0, 4); created.push({ block: opts.block, head: [...head] }); return new DFlashRunner(); }
  constructor() { this.stats = { cycles: 3, tokens: 9, accepted: 6 }; }
  dispose() {}
}
