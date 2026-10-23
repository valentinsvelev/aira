export type Health = {
  status: string;
  llm_enabled: boolean;
  llm_provider: string;
  local_model: string;
  model_name?: string;
  llm_reachable: string;
  faithfulness_mode: string;
  web_enabled: boolean;
  web_provider?: string;
  retrieval: string;
  vector_index?: VectorStatus;
};

export type DocumentInfo = { path: string; active: boolean; version: number | null; chunks: number };
export type DocumentList = { documents: DocumentInfo[] };
export type IngestResult = { path: string; status: string; chunks: number; error?: string; vector_index?: VectorStatus; vector_index_error?: string };
export type IngestResponse = { results: IngestResult[]; summary: Record<string, number>; vector_index?: VectorStatus; vector_index_error?: string };

export type SearchHit = {
  id: string;
  path: string;
  page: number | null;
  version: number | null;
  score: number;
  text: string;
  source_kind: string;
  retrieval_method?: string;
};
export type WebHit = {
  title?: string;
  url?: string;
  excerpt?: string;
  evidence_id?: string;
  source_kind?: string;
  error?: string;
  retrieved_at?: string;
  provider?: string;
};
export type EvidenceDetail = {
  id: string;
  source_kind: string;
  source_url: string | null;
  document_path?: string;
  document_version?: number;
  document_sha256?: string;
  page: number | null;
  text: string;
  text_sha256: string;
  source_active: boolean;
  note?: string;
};
export type MemoryRoute = "auto" | "conversation" | "research" | "both" | "neither";
export type ConversationMemorySource = {
  turn_id: number; conversation_id: string; role: string; created_at: string; score: number;
};
export type ConversationMemoryHit = ConversationMemorySource & {
  text: string; source_kind: "conversation_turn";
};
export type AnswerResponse = {
  memory_route?: MemoryRoute;
  trace_id?: string | null;
  conversation_sources?: ConversationMemorySource[];
  answer: string;
  model?: string;
  mode: string;
  session_id: string;
  context_revision?: number;
  sources?: { id: string; path: string; page: number | null; version: number | null; score: number | null; source_kind?: string; url?: string; title?: string }[];
  preloaded_source_ids?: string[];
  web_used?: boolean;
  web_pages_fetched?: number;
  web_fetch_errors?: { url: string; reason: string }[];
  academic_used?: boolean;
  academic_provider_errors?: Record<string, string>;
  message_count?: number;
};
export type ContextNote = {
  id: string;
  claim: string;
  quote: string;
  evidence_id: string;
  source_active: boolean;
};
export type WorkingContext = { session_id: string; revision: number; notes: ContextNote[] };
export type ContextEvent = {
  id: string;
  operation: string;
  status: string;
  old_claim: string | null;
  new_claim: string | null;
  note_id: string | null;
  evidence_id: string | null;
  reason: string;
  previous_revision: number;
  new_revision: number;
};
export type NoteResult = {
  accepted: boolean;
  reason: string;
  note_id: string | null;
  revision: number;
  event_id: string;
  similarity: number | null;
  verification_method: string;
};
export type CompareResult = {
  verification_method: string;
  cosine_similarity?: number;
  threshold?: number;
  above_threshold?: boolean;
  supported_probability?: number | null;
  accepted?: boolean;
  reason?: string;
  warning: string;
};
export type WorkflowStep = {
  node: string;
  status: "completed" | "skipped" | "failed";
  latency_ms: number;
  sources: number;
  reason?: string;
  stage?: number;
  start_ms?: number;
  span_id?: string;
  branch?: "enough" | "insufficient" | "repeat" | "done";
  iteration?: number;
};
export type SavedWorkflowGraph = {
  schema_version?: number;
  nodes: { id: string; type: string; x: number; y: number }[];
  edges: { source: string; target: string; condition?: string | null }[];
};
export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: "answer" | "chat";
  model?: string;
  sourceIds?: string[];
  sourceDetails?: { id: string; path: string; page: number | null; version: number | null; score: number | null; source_kind?: string; url?: string; title?: string }[];
  webUsed?: boolean;
  workflowSteps?: WorkflowStep[];
  workflowGraph?: SavedWorkflowGraph | null;
  workflowAgentId?: string;
  memoryRoute?: MemoryRoute;
  traceId?: string | null;
  conversationSources?: ConversationMemorySource[];
  webPagesFetched?: number;
  webFetchErrors?: { url: string; reason: string }[];
  academicUsed?: boolean;
  time: string;
};

export type RetrievalOptions = {
  mode: "sparse" | "dense" | "hybrid";
  top_k: number;
  bm25_candidates: number;
  dense_candidates: number;
  rrf_k: number;
  min_similarity: number;
  reranker: "none" | "jinaai/jina-reranker-v3.5" | "Alibaba-NLP/gte-reranker-modernbert-base";
  rerank_candidates: number;
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2" | "Alibaba-NLP/gte-large-en-v1.5";
};
export type GenerationOptions = {
  model: "llama3.2:3b" | "qwen3.5:4b" | "granite4.1:3b" | "openbmb/minicpm-v4.7:1b" | null;
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  num_predict: number; // -1 means model-decided (no explicit application limit)
  num_ctx: number;
  repeat_penalty: number;
  seed: number | null;
};
export type VectorStatus = {
  enabled: boolean;
  indexed: number;
  missing: number | null;
  model?: string;
  dimensions?: number;
};

export type AcademicPaper = {
  provider: "arxiv" | "openalex";
  provider_id: string;
  found_in: string[];
  title: string;
  authors: string[];
  year: number | null;
  abstract: string;
  url: string;
  pdf_url: string | null;
  doi: string | null;
  arxiv_id: string | null;
  evidence_id: string | null;
  retrieved_at: string;
  source_kind: "academic_metadata_unverified";
};
export type AcademicResult = {
  results: AcademicPaper[];
  errors: Record<string,string>;
  sources: string[];
  query: string;
};

// Persisted in SQL; history pages return the newest 40 messages, with an older cursor.
export type ConversationRecord = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};
export type ConversationList = { conversations: ConversationRecord[]; has_more: boolean };
export type StoredTurn = {
  id: number;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  mode: "answer" | "chat" | null;
  model: string | null;
  sources: NonNullable<AnswerResponse["sources"]>;
  metadata: {
    web_used?: boolean;
    workflow_steps?: WorkflowStep[];
    workflow_graph?: SavedWorkflowGraph | null;
    memory_route?: MemoryRoute;
    trace_id?: string | null;
    conversation_sources?: ConversationMemorySource[];
    web_pages_fetched?: number;
    web_fetch_errors?: { url: string; reason: string }[];
    academic_used?: boolean;
    agent_profile_id?: string;
    agent_profile_name?: string;
  };
  created_at: string;
};
export type ConversationMessages = { messages: StoredTurn[]; has_more: boolean; next_cursor: number | null };

export type AgentTool = "local_search" | "conversation_search" | "web_search" | "academic_search" | "context_read" | "context_edit";
export type AgentProfile = {
  id: string;
  name: string;
  description: string;
  system_instructions: string;
  enabled_tools: AgentTool[];
  generation?: GenerationOptions | null;
  builtin?: boolean;
};
export type AgentProfiles = { agents: AgentProfile[] };
export type RAGSystem = {
  id: string; name: string; description: string; system_instructions: string;
  retrieval: RetrievalOptions; generation: GenerationOptions; builtin?: boolean;
};
export type RAGSystems = { systems: RAGSystem[] };

export type TraceSpan = { id: string; parent_id: string | null; name: string; kind: "tool" | "model" | "retrieval" | "agent" | "operation"; start_us: number; duration_us: number; error: boolean; session_id?: string | null; model?: string | null; strategy?: string | null; tool?: string | null; evidence_count?: number | null; input?: string | null; output?: string | null };
export type TraceRun = { id: string; session_id?: string | null; duration_us: number; spans: TraceSpan[] };
export type TraceList = { available: boolean; traces: TraceRun[]; message?: string };
