// All browser requests remain same-origin; Next.js proxies to local FastAPI server-side.
export class ApiError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/backend${path}`, {
    cache: "no-store",
    ...init,
    headers: {
      ...(init.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init.headers ?? {}),
    },
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    let reason = `Request failed (${response.status})`;
    if (payload && typeof payload === "object" && "detail" in payload) {
      const detail = (payload as { detail?: unknown }).detail;
      if (typeof detail === "string") reason = detail;
      else if (detail && typeof detail === "object" && "reason" in detail &&
               typeof (detail as { reason: unknown }).reason === "string") {
        reason = (detail as { reason: string }).reason;
      }
    }
    throw new ApiError(reason, response.status);
  }
  return payload as T;
}

export function postJSON<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export async function deleteAPI(path: string): Promise<void> {
  const result = await fetch(`/api/backend${path}`, { method: "DELETE", cache: "no-store" });
  if (!result.ok) throw new ApiError(`Delete failed (${result.status})`, result.status);
}

export function patchJSON<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, { method: "PATCH", body: JSON.stringify(body) });
}
