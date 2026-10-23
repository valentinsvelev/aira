/** Browser-local Paper Explorer sessions. Research Chat history remains in PostgreSQL. */
export const PAPER_HISTORY_KEY = "aira.paper-explorer.sessions.v1";

export type PaperRecord = {
  number: number; evidence_id: string; title: string; authors: string[]; year: number | null;
  url: string; abstract: string; provider: string; cited: boolean;
};
export type PaperAnswerResult = {
  question: string; answer: string; papers: PaperRecord[]; scope: "abstracts_only";
  search_query: string; provider_errors: Record<string, string>;
};
export type PaperTurn = { question: string; result: PaperAnswerResult };
export type PaperSession = {
  id: string; title: string; created_at: string; updated_at: string; turns: PaperTurn[];
};

export function parsePaperHistory(raw: string | null): PaperSession[] {
  if (!raw) return [];
  try {
    const value: unknown = JSON.parse(raw);
    if (!Array.isArray(value)) return [];
    const sessions: PaperSession[] = [];
    for (const entry of value) {
      if (!entry || typeof entry !== "object") continue;
      const item = entry as Partial<PaperSession>;
      if (typeof item.id !== "string" || !item.id || typeof item.title !== "string" ||
          typeof item.created_at !== "string" || typeof item.updated_at !== "string" || !Array.isArray(item.turns)) continue;
      const turns = item.turns.filter((turn): turn is PaperTurn =>
        Boolean(turn && typeof turn.question === "string" && turn.result &&
          typeof turn.result.answer === "string" && Array.isArray(turn.result.papers) &&
          turn.result.papers.every(paper => paper && typeof paper.number === "number" && typeof paper.title === "string" &&
            typeof paper.url === "string" && typeof paper.abstract === "string" && typeof paper.provider === "string" &&
            typeof paper.cited === "boolean" && Array.isArray(paper.authors)) &&
          turn.result.provider_errors && typeof turn.result.provider_errors === "object"));
      if (turns.length) sessions.push({...item, id:item.id, title:item.title, created_at:item.created_at,
        updated_at:item.updated_at, turns});
    }
    return sessions.sort((a,b) => b.updated_at.localeCompare(a.updated_at));
  } catch { return []; }
}

export function readPaperHistory(): PaperSession[] {
  try { return parsePaperHistory(window.localStorage.getItem(PAPER_HISTORY_KEY)); }
  catch { return []; }
}

/** Throws on quota/storage errors so the caller can warn that changes aren't durable. */
export function writePaperHistory(sessions: PaperSession[]): void {
  window.localStorage.setItem(PAPER_HISTORY_KEY, JSON.stringify(sessions));
}

export function upsertPaperHistory(sessions: PaperSession[], next: PaperSession): PaperSession[] {
  return [next, ...sessions.filter(item => item.id !== next.id)]
    .sort((a,b) => b.updated_at.localeCompare(a.updated_at));
}

export function paperSessionTitle(question: string): string {
  const title = question.trim().replace(/\s+/g, " ");
  return title.length > 74 ? `${title.slice(0, 73).trimEnd()}…` : title || "Paper research";
}
