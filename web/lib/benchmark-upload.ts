/** Parse a small *browser-local* benchmark file. Send only a selected sample. */
export type BenchmarkRow = { question: string; answer: string; context: string };

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let cells: string[] = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { cell += '"'; i++; }
      else quoted = !quoted;
    } else if (char === "," && !quoted) { cells.push(cell); cell = ""; }
    else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i++;
      cells.push(cell); if (cells.some(value => value.trim())) rows.push(cells);
      cells = []; cell = "";
    } else cell += char;
  }
  if (quoted) throw new Error("CSV has an unclosed quoted field.");
  cells.push(cell); if (cells.some(value => value.trim())) rows.push(cells);
  return rows;
}

export function parseBenchmarkFile(filename: string, text: string): BenchmarkRow[] {
  const ext = filename.split(".").at(-1)?.toLowerCase();
  if (text.length > 2_000_000) throw new Error("Dataset must be smaller than 2 MB for a browser preview.");
  let raw: unknown;
  if (ext === "json") raw = JSON.parse(text);
  else if (ext === "jsonl") raw = text.split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line));
  else if (ext === "csv") {
    const [columns, ...records] = parseCsv(text);
    if (!columns) throw new Error("CSV must include a header row.");
    const names = columns.map(value => value.trim().toLowerCase().replace(/^\uFEFF/, ""));
    raw = records.map(row => Object.fromEntries(names.map((name, i) => [name, row[i] ?? ""])));
  } else throw new Error("Use JSON, JSONL or CSV (Parquet is not supported yet).");
  if (raw && typeof raw === "object" && !Array.isArray(raw) && "examples" in raw) raw = (raw as {examples:unknown}).examples;
  if (!Array.isArray(raw)) throw new Error("Dataset must contain a list of question/answer objects.");
  if (raw.length > 10000) throw new Error("For safety, limit uploaded datasets to 10,000 rows.");
  const parsed = raw.map((value: unknown) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return null;
    const record = value as Record<string, unknown>;
    const question = record.question ?? record.query;
    const answer = record.answer ?? record.reference ?? record.gold_answer;
    const context = record.context ?? record.evidence ?? "";
    if (typeof question !== "string" || !question.trim() || typeof answer !== "string" || !answer.trim() || typeof context !== "string") return null;
    return { question: question.trim().slice(0, 5000), answer: answer.trim().slice(0, 2000), context: context.slice(0, 15000) };
  }).filter((item): item is BenchmarkRow => item !== null);
  if (!parsed.length) throw new Error("No valid rows. Required columns: question, answer; optional: context.");
  return parsed;
}
