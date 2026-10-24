// A stand-in for kohra.js: records from_pretrained options and the prompt generate gets.
export const calls = [];
export class DiffusionLM {
  static async from_pretrained(opts) {
    calls.push({ type: 'load', opts });
    return { async generate(prompt, o) { calls.push({ type: 'generate', prompt, o }); return { text: '391', tokens: 2, tokensPerSecond: 10 }; }, session: null };
  }
}
