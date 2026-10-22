// A model descriptor from a Hugging Face repo id: which weight formats the repo holds, their files and sizes, and the
// architecture. Back ends decide from this alone; describe() never decides who runs a model.

const HF = 'https://huggingface.co';

// The last answer the Hub gave for each URL, kept in Cache Storage so a model already downloaded still loads offline.
export const DESCRIBE_CACHE = 'anyformer-describe-v1';

export async function describe(repo, { fetch = globalThis.fetch, revision = 'main', endpoint = HF, caches = globalThis.caches } = {}) {
  const url = `${endpoint}/api/models/${repo}${revision === 'main' ? '' : `/revision/${revision}`}?blobs=true`;
  const store = caches ? await caches.open(DESCRIBE_CACHE).catch(() => null) : null;
  let res, failed;
  try { res = await fetch(url); } catch (e) { failed = e; }
  if (res?.ok) {
    const info = await res.json();
    await store?.put(url, new Response(JSON.stringify(info), { headers: { 'Content-Type': 'application/json' } })).catch(() => {});
    return fromHubInfo(info, { endpoint });
  }
  // No network, or the Hub failed: the cached answer, when this browser has one.
  const hit = await store?.match(url).catch(() => null);
  if (hit) return fromHubInfo(await hit.json(), { endpoint });
  throw failed ? new Error(`describe ${repo}: ${url} failed: ${failed.message}`, { cause: failed })
    : new Error(`describe ${repo}: ${url} returned ${res.status}`);
}

// Pure: Hub API JSON → descriptor. Exported for tests and for callers that already hold the JSON.
export function fromHubInfo(info, { endpoint = HF } = {}) {
  const files = (info.siblings || []).map((s) => ({ name: s.rfilename, size: s.size ?? null }));
  const gguf = files.filter((f) => f.name.endsWith('.gguf'));
  const onnx = files.filter((f) => f.name.endsWith('.onnx'));
  const safetensors = files.filter((f) => f.name.endsWith('.safetensors'));
  const formats = [gguf.length && 'gguf', onnx.length && 'onnx', safetensors.length && 'safetensors'].filter(Boolean);
  return {
    repo: info.id,
    revision: info.sha || null,
    endpoint,
    formats,
    // GGUF repos report the architecture in the gguf block; transformers repos in config.model_type.
    architecture: info.gguf?.architecture || info.config?.model_type || null,
    task: info.pipeline_tag || null,
    // Hub tags carry what format and architecture miss, e.g. 'text-diffusion' marks a kohra model.
    tags: info.tags || [],
    // all: every file, for the side files a format splits off (ONNX external data, tokenizers).
    files: { gguf, onnx, safetensors, all: files },
    ggufTotalBytes: info.gguf?.total ?? null,
  };
}
