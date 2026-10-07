// A model descriptor from a Hugging Face repo id: which weight formats the repo holds, their files and sizes, and the
// architecture. Back ends decide from this alone; describe() never decides who runs a model.

const HF = 'https://huggingface.co';

export async function describe(repo, { fetch = globalThis.fetch, revision = 'main', endpoint = HF } = {}) {
  const url = `${endpoint}/api/models/${repo}${revision === 'main' ? '' : `/revision/${revision}`}?blobs=true`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`describe ${repo}: ${url} returned ${res.status}`);
  return fromHubInfo(await res.json(), { endpoint });
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
    files: { gguf, onnx, safetensors },
    ggufTotalBytes: info.gguf?.total ?? null,
  };
}
