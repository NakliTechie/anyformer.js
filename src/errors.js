// Every error load() and a session throw is an AnyformerError with a code from ERROR_CODES. Callers branch on the
// code; the message is for people and can change. (Back-end refusals before loading are REFUSALS in resolve.js.)

export const ERROR_CODES = {
  NO_BACKEND: 'no back end can run the model under this budget; err.plan.candidates holds each code and reason',
  LOAD_FAILED: 'the repo lookup, a download or the back end failed while loading; err.cause holds the original error',
  BUDGET_TOO_SMALL: 'the back end measured the model at load time and the budget does not cover it; retry with err.need',
  ABORTED: 'the AbortSignal fired; err.cause holds its reason',
  EXECUTION_FAILED: 'the back end failed while generating; err.cause holds the original error',
  INVALID_ARGUMENT: 'a value passed by the caller is wrong',
  DISPOSED: 'the session was disposed; load the model again',
};

export class AnyformerError extends Error {
  constructor(code, message, { cause, ...extra } = {}) {
    super(message, cause === undefined ? undefined : { cause });
    this.code = code;
    Object.assign(this, extra);
  }
}
// On the prototype, so stack traces say AnyformerError.
AnyformerError.prototype.name = 'AnyformerError';

// By name and code, not instanceof: a page can hold two copies of anyformer (CDN imports), and the check still holds.
export function isAnyformerError(e, code) {
  return e instanceof Error && e.name === 'AnyformerError' && e.code in ERROR_CODES && (code === undefined || e.code === code);
}

// Any error into an AnyformerError. An aborted signal wins: fetch rejects with the abort reason, which is not always
// an AbortError. An AnyformerError keeps its code; anything else gets `code` (the step that failed).
export function wrap(e, code, signal, where = '') {
  if (signal?.aborted) return isAnyformerError(e, 'ABORTED') ? e : new AnyformerError('ABORTED', `${where}aborted`, { cause: signal.reason });
  if (isAnyformerError(e)) return e;
  if (e?.name === 'AbortError') return new AnyformerError('ABORTED', `${where}aborted`, { cause: e });
  return new AnyformerError(code, `${where}${e?.message ?? e}`, { cause: e });
}
