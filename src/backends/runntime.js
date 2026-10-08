// runntime (Software Mansion): hand-written TypeGPU kernels for speech, vision, embedding and privacy-filter models.
// It covers the Hugging Face ids its own transformers.js plugin maps (runntime src/zoo/transformers/*/loader.ts) and
// loads their weights from the software-mansion runntime-* repos. runntime and TypeGPU load from the CDN at load()
// time; TypeGPU comes from the exact URL runntime's jsDelivr bundle imports, so both share one module and one device.

const RUNNTIME = 'https://cdn.jsdelivr.net/npm/runntime@0.1.0/zoo/+esm';
const TYPEGPU = 'https://cdn.jsdelivr.net/npm/typegpu@0.12.5/+esm';

// Each entry: the task, any other Hub tag for the same task, the zoo factory, the path into runntime's `models`, the
// method run() calls, whether the task needs shader-f16, and the weight download in bytes (runntime models.ts).
const T = (task, create, path, method, f16, bytes, aliases = []) => ({ task, aliases, create, path, method, f16, bytes });
const EMBED = T('feature-extraction', 'createTextEmbedder', ['textEmbedding', 'ALL_MINILM_L6_V2', 'DEFAULT'], 'embed', false, 45e6, ['sentence-similarity']);
export const RUNNTIME_MODELS = {
  'Xenova/all-MiniLM-L6-v2': EMBED,
  'sentence-transformers/all-MiniLM-L6-v2': EMBED,
  'openai/privacy-filter': T('token-classification', 'createPrivacyFilter', ['privacyFilter', 'PRIVACY_FILTER', 'DEFAULT'], 'detect', false, 1.55e9),
  'onnx-community/moonshine-tiny-ONNX': T('automatic-speech-recognition', 'createSpeechToText', ['speechToText', 'MOONSHINE', 'TINY', 'DEFAULT'], 'transcribe', false, 55e6),
  'onnx-community/moonshine-base-ONNX': T('automatic-speech-recognition', 'createSpeechToText', ['speechToText', 'MOONSHINE', 'BASE', 'DEFAULT'], 'transcribe', false, 125e6),
  'onnx-community/yolo26n-ONNX': T('object-detection', 'createObjectDetector', ['objectDetection', 'YOLO26', 'N'], 'detectObjects', true, 5e6),
  'onnx-community/yolo26s-ONNX': T('object-detection', 'createObjectDetector', ['objectDetection', 'YOLO26', 'S'], 'detectObjects', true, 18e6),
  'onnx-community/yolo26m-ONNX': T('object-detection', 'createObjectDetector', ['objectDetection', 'YOLO26', 'M'], 'detectObjects', true, 39e6),
  'onnx-community/depth-anything-v2-small': T('depth-estimation', 'createDepthEstimator', ['depthEstimation', 'DEPTHART', 'S'], 'estimateDepth', true, 13e6),
  'onnx-community/depth-anything-v2-base': T('depth-estimation', 'createDepthEstimator', ['depthEstimation', 'DEPTHART', 'B'], 'estimateDepth', true, 24e6),
  'onnx-community/mobilenetv4_conv_small.e2400_r224_in1k': T('image-classification', 'createImageClassifier', ['imageClassification', 'MOBILENETV4', 'SMALL'], 'classify', true, 10e6),
};

// One TypeGPU root per page, made on first load: runntime runs every model on the device initRunntime() was given.
let ready;
async function zoo() {
  ready ??= (async () => {
    const [rt, { default: tgpu }] = await Promise.all([import(RUNNTIME), import(TYPEGPU)]);
    rt.initRunntime(await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } }));
    return rt;
  })();
  return ready;
}

// ImageData (from a canvas) is converted; anything else goes to runntime as it is (an ImageBuffer).
const imageInput = (rt, x) => (typeof ImageData !== 'undefined' && x instanceof ImageData ? rt.imageBufferFromImageData(x) : x);

export const runntimeBackend = {
  id: 'runntime',
  tasks: [...new Set(Object.values(RUNNTIME_MODELS).map((m) => m.task))],
  source: RUNNTIME,

  canRun(model, caps, { task } = {}) {
    const m = RUNNTIME_MODELS[model.repo];
    if (!m) return { ok: false, code: 'model', reason: `runntime has no port of ${model.repo}; it covers ${Object.keys(RUNNTIME_MODELS).length} Hub ids (RUNNTIME_MODELS)` };
    const want = task || model.task;
    if (want && want !== m.task && !m.aliases.includes(want)) return { ok: false, code: 'task', reason: `runntime runs ${model.repo} as ${m.task}, not ${want}` };
    if (!caps.webgpu) return { ok: false, code: 'webgpu', reason: 'no WebGPU adapter' };
    if (m.f16 && !caps.f16) return { ok: false, code: 'f16', reason: `runntime's ${m.task} runs in half precision and needs shader-f16` };
    return { ok: true, estimate: { gpuBytes: m.bytes, diskBytes: m.bytes, tokPerSec: null } };
  },

  async load(model, { onProgress, signal }) {
    const m = RUNNTIME_MODELS[model.repo];
    const rt = await zoo();
    const entry = m.path.reduce((o, k) => o[k], rt.models);
    const cache = await rt.createOpfsCache('anyformer-runntime');
    const runner = await rt[m.create](entry, {
      cache, signal,
      onProgress: onProgress && ((name, done, total) => onProgress({ backend: 'runntime', status: 'weights', file: name, loaded: done, total })),
    });
    const vision = m.task.startsWith('image') || m.task === 'object-detection' || m.task === 'depth-estimation';
    return {
      task: m.task,
      // Text embedding takes a string or an array of strings (embedBatch); vision takes ImageData or an ImageBuffer;
      // speech takes 16 kHz mono Float32Array samples (runntime's decodeAudio makes them from a file).
      async run(input, opts) {
        if (m.method === 'embed' && Array.isArray(input)) return runner.embedBatch(input);
        if (m.method === 'transcribe') return { text: await runner.transcribe(input) };
        return runner[m.method](vision ? imageInput(rt, input) : input, opts);
      },
      dispose: () => runner.dispose(),
    };
  },
};
