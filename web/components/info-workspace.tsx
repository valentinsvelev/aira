"use client";

import { ArrowUpRight, BookOpen, Bot, Database, FileSearch, History, ShieldCheck, Workflow } from "lucide-react";

const papers = [
  { title: "Ai2 Scholar QA: Organized Literature Synthesis with Attribution", authors: "Singh et al. (2025)",
    description: "An open scientific QA system that organizes retrieved literature and attributes claims to sources.",
    url: "https://aclanthology.org/2025.acl-demo.49/" },
  { title: "Context Language Models", authors: "Shao et al. (2026)",
    description: "Editable context and working memory for language-model systems.",
    url: "https://arxiv.org/abs/2609.37725" },
  { title: "Synthesizing scientific literature with retrieval-augmented language models", authors: "Asai et al. (2026)",
    description: "OpenScholar: evidence-grounded scientific literature synthesis and evaluation.",
    url: "https://www.nature.com/articles/s41586-025-10072-4" },
  { title: "Controllable Context Sensitivity and the Knob Behind It", authors: "Minder et al. (2025)",
    description: "How strongly model predictions respond to information provided in context.",
    url: "https://openreview.net/forum?id=Igm9bbkzHC" },
  { title: "Tracing and Reversing Edits in LLMs", authors: "Youssef et al. (2026)",
    description: "Identifying and reversing knowledge edits in language models (model weights, not document context).",
    url: "https://openreview.net/forum?id=AiT8F6pbfi" },
  { title: "How Persuasive Is Your Context?", authors: "Nguyen et al. (2025)",
    description: "Measures how much new contextual information shifts model predictions.",
    url: "https://aclanthology.org/2025.emnlp-main.1633/" },
  { title: "Context versus Prior Knowledge in Language Models", authors: "Du et al. (2024)",
    description: "Studies how models balance prior knowledge with evidence supplied in context.",
    url: "https://aclanthology.org/2024.acl-long.714/" },
  { title: "DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research", authors: "Shao et al. (2026)",
    description: "Training an open deep-research model with evolving evaluation rubrics.",
    url: "https://openreview.net/forum?id=97NEP1pyS3" },
  { title: "Out of Style: RAG’s Fragility to Linguistic Variation", authors: "Cao et al. (2026)",
    description: "Examines how linguistic variation in user queries affects retrieval and answer quality.",
    url: "https://aclanthology.org/2026.eacl-long.13/" },
  { title: "ScholarCatalyst: A Benchmark for Retrieving Papers That Inspire New Research", authors: "Kim et al. (2026)",
    description: "Evaluates retrieving earlier scientific ideas from the literature using author-annotated relevance judgments.",
    url: "https://arxiv.org/abs/2610.02202" },
  { title: "Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction", authors: "Li et al. (2026)",
    description: "Explores agentic retrieval through direct interactions with corpora rather than fixed top-k similarity search.",
    url: "https://arxiv.org/abs/2605.05242" },
].sort((a, b) => {
  // The authors field begins with the first author's surname.
  const surname = (paper: { authors: string }) => paper.authors.split(/\s+/)[0].toLocaleLowerCase("en");
  return surname(a).localeCompare(surname(b), "en") || a.title.localeCompare(b.title, "en");
});

const retrievalSettings = [
  { name: "Retrieval strategy", detail: "Sparse (BM25) ranks keyword matches; Dense uses embedding similarity; Hybrid merges their ranked results with reciprocal-rank fusion (RRF)." },
  { name: "Embedding model", detail: "The model that turns passages and queries into vectors for Dense and Hybrid. Each model needs its own indexed embeddings." },
  { name: "Top-k", detail: "The maximum number of passages finally sent to the language model. Larger values increase context but can introduce noise." },
  { name: "Sparse / Dense candidates", detail: "How many matches each retriever considers before final selection or fusion. More candidates may improve recall at added cost." },
  { name: "RRF constant (k)", detail: "Controls how rank positions contribute to Hybrid fusion. Higher values flatten differences between early and later ranks; it is not a token limit." },
  { name: "Minimum cosine similarity", detail: "Optional lower bound for dense matches. A threshold that is too high can exclude useful evidence; it does not apply to sparse scores." },
];

export function InfoWorkspace() {
  return <div className="feature-page info-page">
    <header className="feature-head"><div><div className="eyebrow">ABOUT THE PROJECT</div><h1>What is AIRA?</h1><p>AIRA is a local retrieval-augmented research assistant for discovering, reading, and critically examining scientific research with AI assistance applying the newest discoveries in industry and academia.</p></div></header>
    <section className="info-intro"><h2>Research assistance, not autonomous research</h2>
      <p>You direct the investigation. The assistant searches indexed PDFs and optionally public research sources, reads relevant passages and helps synthesize answers with inspectable citations. Research notes are editable and checked against stored source evidence.</p>
    </section>
    <div className="info-steps">
      <article><FileSearch size={21}/><h3>1. Discover</h3><p>Build a literature-review shortlist from arXiv and OpenAlex.</p></article>
      <article><Database size={21}/><h3>2. Collect</h3><p>Import approved PDFs and index passages for sparse or vector retrieval.</p></article>
      <article><Bot size={21}/><h3>3. Investigate</h3><p>Ask questions using predictable RAG or a tool-calling research agent.</p></article>
      <article><ShieldCheck size={21}/><h3>4. Verify</h3><p>Inspect source citations, edit working notes and check faithfulness.</p></article>
      <article><Workflow size={21}/><h3>5. Inspect</h3><p>Review model and tool-call traces in the local observability workspace.</p></article>
    </div>
    <section className="info-section"><h2>Two research modes</h2>
      <div className="info-mode-grid"><div><h3><BookOpen size={17}/> RAG</h3><p>Retrieves relevant evidence before generating a response. Predictable and generally easier to evaluate.</p></div>
        <div><h3><Bot size={17}/> Agentic RAG</h3><p>Lets the language model select tools and perform follow-up retrieval within configured permissions and limits.</p></div></div>
    </section>
    <section className="info-section info-retrieval" aria-labelledby="retrieval-heading">
      <h2 id="retrieval-heading">How retrieval works</h2>
      <p>AIRA indexes text passages from documents. When you ask a question, a retriever ranks candidate passages and selects the ones the model can cite. This step is separate from answer generation: a fluent answer is not proof that its sources were relevant.</p>
      <div className="info-retrieval-modes">
        <div><strong>Sparse <span>Default</span></strong><p>Finds passages using overlapping terms and statistics about how rare those terms are. A strong, interpretable lexical baseline.</p></div>
        <div><strong>Dense</strong><p>Embeds questions and passages with the same model and retrieves similar vectors, even when exact wording differs.</p></div>
        <div><strong>Hybrid</strong><p>Combines lexical and dense rankings with RRF. Useful when you want coverage of both exact terms and semantic matches.</p></div>
      </div>
      <h3>Retrieval settings explained</h3>
      <dl className="info-parameter-list">{retrievalSettings.map(setting => <div key={setting.name}><dt>{setting.name}</dt><dd>{setting.detail}</dd></div>)}</dl>
      <div className="info-research-link"><div><strong>Want to learn which retriever is right for you? Check out my arXiv paper:</strong><p>In Velev & Spitz (2026), my coauthor and I compare 33 retrieval systems across seven datasets and analyzing accuracy, latency, and failure modes.</p></div><a href="https://arxiv.org/abs/2609.29455" target="_blank" rel="noopener noreferrer">Read paper <ArrowUpRight size={16}/></a></div>
    </section>
    <section className="info-section"><h2>Research inspiration</h2><p>Selected scientific papers, which inspired me to develop AIRA:</p>
      <div className="info-papers">{papers.map(p => <a key={p.url} href={p.url} target="_blank" rel="noopener noreferrer" className="info-paper">
        <span><small>{p.authors}</small><strong>{p.title}</strong><span>{p.description}</span></span><ArrowUpRight size={19}/>
      </a>)}</div>
    </section>
  </div>;
}
