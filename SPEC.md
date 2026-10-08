# anyformer.js — spec

Tier: Product (several back ends, a resolver, one store). anyformer depends on diskformer and kohra; neither depends on
anyformer.

## §0 Agent contract (DRIVER v0.1 pass, 2026-10-08)

The driver: an agent in a browser tab (or a script) that must run a model it names by repo id, with no idea which
runtime fits, how much GPU memory the tab may take, or whether a download is 0.5 GB or 36 GB. It is context-poor, may
be killed mid-turn, and must not burn a user's disk or bandwidth by guessing.

1. **One perception act: `anyformer_plan` (`plan()`).** One call returns the model's formats, architecture and task,
   the browser's caps, the budget, the chosen back end, and every back end's verdict. The agent never probes or
   fetches separately before deciding.
2. **Closed vocabularies.** Errors from load() and a session are `AnyformerError` with `code` in `ERROR_CODES`
   (NO_BACKEND · LOAD_FAILED · BUDGET_TOO_SMALL+need · ABORTED · EXECUTION_FAILED · INVALID_ARGUMENT · DISPOSED),
   checked by name and code (`isAnyformerError`), so two copies of the library agree. A refusal is `{code, reason, need?}` with `code` in `REFUSALS`: format · model · task · webgpu ·
   f16 · opfs · budget · threw. Each code has one remedy (`REFUSALS[code]`); `budget` carries `need`, the bytes that
   would pass. Tool consequences are `read` · `stages` · `mutates`. Tool errors are `{error}` data, never throws.
3. **One verdict per next action.** `chosen` set → load. `chosen` null → read the codes: format/model → another repo;
   webgpu/f16/opfs → another browser or GPU (nothing in this tab fixes it); budget → re-plan with `need`; threw → a bug.
4. **Bounded output.** `anyformer_plan` grows with the number of back ends, never with the repo's file count (the file
   list stays out). The journal holds the last 50 calls. `generate` returns one reply.
5. **Every failure names its remedy.** NO_BACKEND lists each back end with its code and reason; `REFUSALS[code]`
   is the next move.
6. **Crash-safe and idempotent.** A session runs one generate at a time; dispose() waits for it and is idempotent. `plan` and `probe` have no side effects. Weights land in OPFS through diskformer's
   resumable ingest (an interrupted download resumes); transformers.js and kohra use the browser HTTP cache. Loading
   twice loads twice; it never corrupts a store.
7. **The tool holds the memory.** `anyformer_sessions` renders the open sessions and the last 50 tool calls with the
   door each came through (`direct`, `window`, `modelContext`).
8. **Accretion.** A back end joins `BACKENDS` only after its browser test passes (test/registry.test.js pins the list);
   every escaped wrong answer becomes a browser-test check (the transformers.js q4f16 word salad became `onTopic`).
9. **The tower.** One line per layer; each reads only the one below.
   - `probe` (browser caps) · `describe` (Hub repo → descriptor)
   - back ends: `canRun` (pure, descriptor + caps) · `load` → session
   - `resolve` (rank the verdicts) → `plan` (probe + describe + resolve) → `load` (plan + back-end load)
   - agent face: `TOOLS` manifest → `call` → WebMCP `registerTool` / `window.anyformer.tools`
10. **The evaluator stays outside.** The browser tests (test/browser/) judge back ends against fixed references:
    llama.cpp b9830 replies for diskformer, an on-topic check for transformers.js and kohra. A page that times out
    exits 2 (indeterminate), never 0. Staging is enforced in the tool, not left to the agent: `anyformer_load` loads
    nothing unless `acceptDiskBytes` covers the plan's disk bytes.

Self-check: perception act ✓ · closed vocabularies ✓ · output bound ✓ · crash-safety ✓ · trajectory ✓ · accretion ✓ ·
tower ✓ · evaluator boundary ✓.

## What shipped (Batch A)

- `load(repo, {budget, maxCtx, backends, onProgress, signal})` → `{backend, plan, generate(messages, {maxNewTokens, signal}),
  dispose()}`; `generate` yields `{text, tokens}`, `text` cumulative.
- Tasks: `plan/load(repo, { task })`, default the repo's pipeline tag; every session has `run(input)`, text generation
  also `generate()`. Back ends list their `tasks`.
- Back ends: transformers.js 4.3.1 (ONNX text-generation, q4 first), kohra @main (text-diffusion ONNX), diskformer
  @main (pinned Gemma 4 26B-A4B and Qwen3.6 35B-A3B GGUFs). transformers.js also runs feature-extraction, speech recognition, object
  detection, image classification, depth, segmentation, token and text classification on fp32 graphs. kohra and diskformer load from their main branches;
  `<backend>.source` pins another copy.
- transformers.js also runs Gemma 4 and Qwen3.5 ONNX as chat with image (and, for Gemma 4, audio) blocks, through
  their conditional-generation classes at q4f16, as LocalMind does.
- localmind @main: LocalMind's hand-written engines, matched by architecture and format (lfm2 GGUF, Gemma 4
  QAT-mobile, qwen35 Prism ternary); resident on the GPU, so budget ≥ file size.
- Agent face: 9 tools (models, probe, plan, load, generate, sessions, unload, stored, remove). Person-only acts: none.
- `MODELS`: tested entries by family; `load(entry)`. `stored()` / `remove(repo)` over the transformers.js cache and
  diskformer's OPFS stores.
- Gaps: no wasm fallback (every back end needs WebGPU); one store per back end, not the shared OPFS store yet (kohra's graph is in the HTTP cache, unlisted);
  llama.cpp-webgpu not started; speed estimates exist for diskformer only.
