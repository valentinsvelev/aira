// Narrow server-side proxy. No API key is sent to the browser.
// This prototype has no authentication; run both servers on 127.0.0.1 only.
import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const chatId = "[a-fA-F0-9-]{36}";
const agentId = "(?:default|[a-fA-F0-9-]{36})";
const ragId = "(?:default|[a-fA-F0-9-]{36})";
const rules: Record<string, RegExp[]> = {
  GET: [
    /^scholar\/model-status$/, /^chat\/progress\/[a-fA-F0-9-]{36}$/, /^health$/, /^traces$/, /^sessions$/, /^tracing\/status$/, /^agents$/, new RegExp(`^agents/${agentId}$`), new RegExp(`^agents/${agentId}/workflow$`), new RegExp(`^agents/${agentId}/workflow/runs$`), new RegExp(`^agents/${agentId}/workflow/runs/[0-9]+/phoenix$`), /^rag-systems$/, /^documents$/, /^retrieval\/status$/, /^literature\/reviews$/, /^literature\/reviews\/[a-fA-F0-9-]{36}$/,
    /^benchmarks\/jobs\/[a-fA-F0-9-]{36}$/, /^retrieval-evals\/datasets$/, /^retrieval-evals\/runs$/, /^retrieval-evals\/runs\/[a-fA-F0-9-]{36}$/, /^retrieval-evals\/jobs\/[a-fA-F0-9-]{36}$/,
    /^conversations$/,
    new RegExp(`^conversations/${chatId}$`),
    new RegExp(`^conversations/${chatId}/messages$`),
    new RegExp(`^conversations/${chatId}/memory/search$`),
    new RegExp(`^conversations/${chatId}/memory/turns$`),
    /^evidence\/[a-f0-9]{64}$/,
    /^context\/[a-zA-Z0-9_-]{1,100}$/,
    /^context\/[a-zA-Z0-9_-]{1,100}\/events$/,
  ],
  POST: [
    /^tracing\/test$/, /^retrieval-evals\/jobs$/, /^benchmarks\/run$/, /^benchmarks\/jobs$/, /^scholar\/answer$/, /^agents$/, /^rag-systems$/, /^answer$/, /^chat$/, /^ingest\/sync$/, /^documents\/upload$/,
    /^retrieval\/search$/, /^retrieval\/reindex$/,
    /^web\/search$/, /^academic\/(search|scholar-url|import)$/,
    /^context\/compare$/, /^context\/[a-zA-Z0-9_-]{1,100}\/notes$/,
    /^conversations$/, /^literature\/reviews$/,
    /^literature\/reviews\/[a-fA-F0-9-]{36}\/(discover|survey)$/,
    /^literature\/reviews\/[a-fA-F0-9-]{36}\/papers\/[a-fA-F0-9-]{36}\/import$/,
  ],
  PUT: [new RegExp(`^agents/${agentId}/workflow$`)],
  PATCH: [new RegExp(`^agents/${agentId}$`), new RegExp(`^rag-systems/${ragId}$`), new RegExp(`^conversations/${chatId}$`), /^literature\/reviews\/[a-fA-F0-9-]{36}\/papers\/[a-fA-F0-9-]{36}$/],
  DELETE: [
    /^benchmarks\/jobs\/[a-fA-F0-9-]{36}$/, /^retrieval-evals\/jobs\/[a-fA-F0-9-]{36}$/,
    new RegExp(`^agents/${agentId}$`), new RegExp(`^rag-systems/${ragId}$`),
    /^context\/[a-zA-Z0-9_-]{1,100}\/notes\/[a-fA-F0-9-]{36}$/,
    new RegExp(`^conversations/${chatId}$`), /^literature\/reviews\/[a-fA-F0-9-]{36}$/,
  ],
};

async function forward(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const route = path.join("/");
  if (!rules[request.method]?.some((rule) => rule.test(route))) {
    return Response.json({ detail: "Unsupported API route" }, { status: 404 });
  }

  // Only a known internal base URL, configured by the developer, is contacted.
  const backend = process.env.BACKEND_URL || "http://127.0.0.1:8000";
  let url: URL;
  try {
    const base = new URL(backend);
    if (!["http:", "https:"].includes(base.protocol)) throw new Error("Invalid protocol");
    url = new URL(`/${route}${request.nextUrl.search}`, base);
  } catch {
    return Response.json({ detail: "Invalid BACKEND_URL configuration" }, { status: 500 });
  }
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  try {
    const body = request.method === "GET" ? undefined : await request.arrayBuffer();
    if (body && body.byteLength > 21 * 1024 * 1024) {
      return Response.json({ detail: "Request exceeds upload limit" }, { status: 413 });
    }
    const response = await fetch(url, {
      method: request.method,
      headers,
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(route === "benchmarks/run" ? 900_000 : 240_000),
    });
    return new Response(response.body, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") || "application/json",
        "cache-control": "no-store",
      },
    });
  } catch {
    return Response.json({ detail: "FastAPI is unavailable or timed out. Check port 8000." }, { status: 502 });
  }
}

export const GET = forward;
export const POST = forward;
export const PUT = forward;
export const PATCH = forward;
export const DELETE = forward;
