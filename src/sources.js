// Where each back end's runtime loads from: exact versions and commits, so a given anyformer behaves the same tomorrow.
// kohra, diskformer and LocalMind move fast on their main branches; `node scripts/pin-sources.mjs` moves these pins
// to each repo's current main, and the browser tests then prove the new pins (test/browser/README.md).
// transformers.js and wllama move only by hand, in step with LocalMind's pins (plan/history.md, 2026-10-08).

export const PINS = {
  transformers: '4.3.1',
  wllama: '3.4.1',      // LocalMind's pin; its stream-drain workaround (src/backends/wllama.js) is for this version
  kohra: 'f678fac7342c79b58b640540b8b3a7d078b7ab1a',
  diskformer: '5bccf328ba4c753c65b6f497039f2d179885a8ac',   // v0.6.1: engines/ and ggml/ (the llama.cpp path)
  localmind: '203288a0f0e23919b43c76bff8dc5d49f48ed300',
};

const gh = (repo, sha, path) => `https://cdn.jsdelivr.net/gh/NakliTechie/${repo}@${sha}/${path}`;

export const SOURCES = {
  transformers: `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${PINS.transformers}/+esm`,
  wllama: `https://cdn.jsdelivr.net/npm/@wllama/wllama@${PINS.wllama}/esm`,
  kohra: gh('kohra', PINS.kohra, 'kohra.js'),
  diskformer: gh('diskformer.js', PINS.diskformer, 'engines'),
  diskformerLlama: gh('diskformer.js', PINS.diskformer, 'ggml/llama.js'),
  localmind: gh('LocalMind', PINS.localmind, 'src'),
};
