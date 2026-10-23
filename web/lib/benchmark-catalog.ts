/**
 * Curated official research benchmarks, checked October 2026.
 * QASPER and DRUID support an on-demand, small-sample diagnostic runner.
 * Advanced benchmarks remain protocol references, not turnkey evaluators.
 */
export const BENCHMARK_TRACKS = [
  "Scientific QA", "Research agents",
] as const;
export type BenchmarkTrack = typeof BENCHMARK_TRACKS[number];

export type Benchmark = {
  id: string;
  name: string;
  year: number;
  track: BenchmarkTrack;
  summary: string;
  protocol: string;
  metrics: string;
  officialUrl: string;
  datasetUrl: string;
  caveat: string;
  priority?: "recommended" | "advanced";
};

export const BENCHMARK_CATALOG: readonly Benchmark[] = [
  {
    id: "qasper", name: "QASPER", year: 2021, track: "Scientific QA",
    summary: "Questions about scientific papers with human-annotated answers and supporting evidence.",
    protocol: "Index each paper's full text, answer held-out questions, predict supporting spans, and score with the official evaluator.",
    metrics: "Answer F1, Evidence F1",
    officialUrl: "https://github.com/allenai/qasper-led-baseline",
    datasetUrl: "https://huggingface.co/datasets/allenai/qasper",
    caveat: "Evidence scoring requires mapping your PDF chunks to the benchmark's annotated evidence spans.",
    priority: "recommended",
  },
  {
    id: "druid", name: "DRUID", year: 2024, track: "Scientific QA",
    summary: "Real-world claims with evidence annotated for relevance and stance to test how RAG models use imperfect context.",
    protocol: "For each supplied claim–evidence pair, predict evidence relevance and stance or evaluate claim verification; compare against the human relevance and stance labels.",
    metrics: "Relevance/stance F1, claim-verification accuracy",
    officialUrl: "https://arxiv.org/abs/2412.17031",
    datasetUrl: "https://huggingface.co/datasets/copenlu/druid",
    caveat: "This is a context-use/fact-verification dataset, not a ready-made scientific paper retrieval corpus. Official scoring adapters are not implemented.",
  },

  {
    id: "browsecomp-plus", name: "BrowseComp-Plus", year: 2025, track: "Research agents",
    summary: "Deep-research QA with a fixed collection of approximately 100,000 web documents and 830 difficult questions.",
    protocol: "Run agents against the released fixed corpus, not the live web. Compare generated answers with gold answers; evaluate retrieved gold/evidence documents with graded relevance judgments.",
    metrics: "Answer accuracy, evidence/gold nDCG@10, Recall@k",
    officialUrl: "https://github.com/texttron/BrowseComp-Plus",
    datasetUrl: "https://huggingface.co/datasets/Tevatron/browsecomp-plus",
    caveat: "Large corpus and obfuscated QA records require the official preparation/evaluation scripts; no adapter or download is included.",
    priority: "advanced",
  },
  {
    id: "deepresearch-bench", name: "DeepResearch Bench", year: 2025, track: "Research agents",
    summary: "Long-form research reports across 100 expert-written tasks in 22 domains.",
    protocol: "Generate a cited report for each benchmark prompt; score report quality using RACE and citation trustworthiness using FACT.",
    metrics: "RACE report-quality dimensions, FACT citation metrics",
    officialUrl: "https://deepresearch-bench.github.io/",
    datasetUrl: "https://github.com/Ayanami0730/deep_research_bench",
    caveat: "Official evaluator currently uses proprietary judge models and web collection; reproducing leaderboard scores is not $0.",
    priority: "advanced",
  },
];
