"use client";

import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { normalizeCitations } from "@/lib/citations";

type Props = {
  content: string;
  sourceIds: string[];
  onOpenSource: (id: string) => void;
};

export function CitationMarkdown({ content, sourceIds, onOpenSource }: Props) {
  const result = useMemo(() => normalizeCitations(content, sourceIds), [content, sourceIds]);

  return <div className="answer-markdown">
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
      a: ({ href, children }) => {
        const match = /^#research-source-(\d+)$/.exec(href || "");
        if (match) {
          const number = Number(match[1]);
          const id = result.citedSourceIds[number - 1];
          if (!id) return <span className="unverified-citation">[unverified citation]</span>;
          return <button type="button" className="inline-citation" aria-label={`Inspect source ${number}`} title={`Inspect evidence ${number}`} onClick={() => onOpenSource(id)}>[{number}]</button>;
        }
        // Other Markdown links are not treated as verified research citations.
        return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
      },
    }}>{result.markdown}</ReactMarkdown>
    {(result.unverifiedCount > 0 || result.modelBibliographyRemoved) && <p className="citation-warning" role="note">
      {result.unverifiedCount > 0 && <span>{result.unverifiedCount} unsupported citation marker(s) were not linked. </span>}
      {result.modelBibliographyRemoved && <span>The model-written reference list was hidden; only sources explicitly cited in this answer appear below. </span>}
      A retrieved passage is not, by itself, proof that the claim is correct.
    </p>}
  </div>;
}
