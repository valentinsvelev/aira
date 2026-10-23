/**
 * A citation marker is valid only if it resolves to an evidence ID returned by
 * the server. Display numbers are local to the *answer*, not to retrieval order.
 *
 * We keep the original source IDs intact so reopening saved conversations with
 * legacy [11] citations still resolves the same immutable evidence records.
 */
export function normalizeCitations(content: string, rawSourceIds: string[]) {
  const sourceIds = Array.from(new Set(rawSourceIds));
  let unverifiedCount = 0;
  let modelBibliographyRemoved = false;
  const citedSourceIds: string[] = [];
  const citedIndexById = new Map<string, number>();

  // Never present a model-written bibliography as verified provenance.
  const bibliographyStart = /^\s*(?:#{1,4}\s*)?(?:References|Bibliography)\s*:?\s*$/im;
  const match = bibliographyStart.exec(content);
  if (match && match.index > 0) {
    content = content.slice(0, match.index).trimEnd();
    modelBibliographyRemoved = true;
  }

  // A generated "Sources: [1], [2]" footer is not trusted provenance.
  // Render one verified, clickable Sources line from the actual answer citations instead.
  content = content.replace(/(?:^|\n)\s*(?:\*\*)?Sources(?:\*\*)?\s*:\s*(?:\[(?:\d+|evidence:[^\]]+)\]\s*[,;·]?\s*)+$/i, "").trimEnd();

  const markdown = content.replace(/\[(?:evidence:([^\]]+)|(\d+))\](?!\()/g,
    (_original, evidenceId: string | undefined, numeric: string | undefined) => {
      const originalIndex = evidenceId !== undefined
        ? sourceIds.indexOf(evidenceId)
        : Number(numeric) - 1;
      if (originalIndex < 0 || originalIndex >= sourceIds.length) {
        unverifiedCount += 1;
        return "**[unverified citation]**";
      }
      const id = sourceIds[originalIndex];
      let displayNumber = citedIndexById.get(id);
      if (displayNumber === undefined) {
        citedSourceIds.push(id);
        displayNumber = citedSourceIds.length;
        citedIndexById.set(id, displayNumber);
      }
      return `[${displayNumber}](#research-source-${displayNumber})`;
    });

  return {
    markdown, sourceIds, citedSourceIds,
    citedSourceNumbers: citedSourceIds.map((_, index) => index + 1),
    unverifiedCount, modelBibliographyRemoved,
  };
}
