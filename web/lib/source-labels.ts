export type PaperSource = { id: string; path?: string; title?: string; authors?: string[]; year?: number | null; page?: number | null; source_kind?: string; url?: string };

export function sourceLabel(item?: PaperSource, fallback = "Source"): string {
  if (!item) return fallback;
  const filename = (item.path || item.title || fallback).split("/").at(-1) || fallback;
  const match = /^(.*?)\s*\((\d{4})\)/.exec(filename);
  const authors = item.authors?.length ? (item.authors.length > 1 ? `${item.authors[0].split(" ").at(-1)} et al.` : item.authors[0].split(" ").at(-1)) : undefined;
  const lead = match ? `${match[1].trim()} (${match[2]})` : authors && item.year ? `${authors} (${item.year})` : filename.replace(/\.pdf$/i, "");
  return item.page != null ? `${lead}, p. ${item.page}` : lead;
}

/** Citations show the canonical web URL rather than a slug or page title. */
export function citedSourceLabel(item?: PaperSource, fallback = "Source"): string {
  if (!item) return fallback;
  const url = item.url || (item.path && /^https?:\/\//i.test(item.path) ? item.path : undefined);
  if (url && /^https?:\/\//i.test(url)) return url;
  return sourceLabel(item, fallback);
}
