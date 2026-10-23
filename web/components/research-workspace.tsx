"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties, PointerEvent as ReactPointerEvent } from "react";
import {
  Activity, ArrowLeft, ArrowRight, BookOpen, BrainCircuit, Check, CheckCheck,
  ChevronDown, CircleAlert, CircleCheck, Clock3, Database, FilePlus2, FileSearch,
  FileText, Globe2, History, Info, Layers3, Link2, LoaderCircle, Menu, MessageSquareText,
  PanelLeftClose, PanelLeftOpen, Plus, RefreshCw, Search, Send, Settings2, ShieldCheck, Sparkles,
  Trash2, Upload, X, Pencil, Bot, Workflow, HelpCircle,
} from "lucide-react";
import { api, postJSON, deleteAPI, patchJSON } from "@/lib/api";
import { CitationMarkdown } from "@/components/citation-markdown";
import { LiveWorkflowCanvas, progressFromSavedRun, type LiveWorkflowProgress } from "@/components/live-workflow-canvas";
import { normalizeCitations } from "@/lib/citations";
import { LiteratureWorkspace } from "@/components/literature-workspace";
import { RecentChatsDialog } from "@/components/recent-chats-dialog";
import { readPaperHistory, writePaperHistory, upsertPaperHistory } from "@/lib/paper-history";
import type { PaperSession } from "@/lib/paper-history";
import { RAGWorkspace } from "@/components/rag-workspace";
import { AgentWorkspace, DEFAULT_GENERATION } from "@/components/agent-workspace";
import { ResearchLogo } from "@/components/research-logo";
import { LibraryWorkspace } from "@/components/library-workspace";
import { TracesWorkspace } from "@/components/traces-workspace";
import { ExperimentsWorkspace } from "@/components/experiments-workspace";
import { RetrievalEvaluationWorkspace } from "@/components/retrieval-evaluation-workspace";
import { InfoWorkspace } from "@/components/info-workspace";
import { ConfigurationWorkspace } from "@/components/configuration-workspace";
import type { ConfigurationDestination } from "@/components/configuration-workspace";
import { sourceLabel, citedSourceLabel } from "@/lib/source-labels";
import type {
  AnswerResponse, ChatMessage, CompareResult, ContextEvent, DocumentList, EvidenceDetail,
  Health, IngestResponse, IngestResult, NoteResult, SearchHit, WebHit, WorkingContext, RetrievalOptions, GenerationOptions, VectorStatus, AcademicPaper, AcademicResult, ConversationRecord, ConversationList, ConversationMessages, StoredTurn, AgentProfile, AgentProfiles, RAGSystem, RAGSystems, MemoryRoute, ConversationMemoryHit,
} from "@/lib/types";

type WorkspaceView = "research" | "search" | "literature" | "agents" | "rag" | "library" | "traces" | "experiments" | "retrieval-eval" | "info" | "configuration";
type InspectorView = "evidence" | "memory" | "verify";
type ModelMode = "answer" | "chat";
type NoteDraft = { claim: string; quote: string; evidenceId: string; replaceId?: string };

const EXAMPLE_QUESTIONS = [
  { title: "Compare LoRA and QLoRA", detail: "Techniques, memory costs and limitations", prompt: "Compare LoRA and QLoRA using my local research papers. Highlight their memory requirements and trade-offs." },
  { title: "Explain NF4 and double quantization", detail: "How QLoRA achieves memory savings", prompt: "How do NF4 and double quantization reduce the memory requirements of QLoRA?" },
  { title: "Analyze the evidence", detail: "Identify assumptions and open questions", prompt: "What are the main caveats and limitations of QLoRA based on the papers in my corpus?" },
];

const iconSize = 16;
const now = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const shortName = (path: string) => path.split("/").at(-1) || path;
const shortId = (id: string) => `${id.slice(0, 9)}…`;
const messageId = () => Math.random().toString(36).slice(2);
const displayTime = (value: string) => new Date(value).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
// A result can retrieve many passages. The bibliography displays only evidence
// explicitly cited in the answer, renumbered from [1] per response.
const citedRefs = (message: ChatMessage): { id: string; number: number }[] => {
  const normalized = normalizeCitations(message.content, message.sourceIds || []);
  return normalized.citedSourceNumbers.map((number) => ({ id: normalized.citedSourceIds[number - 1], number }));
};
const restoredMessage = (turn: StoredTurn): ChatMessage => ({
  id: String(turn.id), role: turn.role, content: turn.content,
  mode: turn.mode ?? undefined, model: turn.model ?? undefined,
  sourceIds: turn.sources.map((source) => source.id), sourceDetails: turn.sources,
  webUsed: turn.metadata.web_used, workflowSteps: turn.metadata.workflow_steps,
  workflowGraph: turn.metadata.workflow_graph, workflowAgentId: turn.metadata.agent_profile_id,
  memoryRoute: turn.metadata.memory_route, traceId: turn.metadata.trace_id,
  conversationSources: turn.metadata.conversation_sources,
  webPagesFetched: turn.metadata.web_pages_fetched,
  webFetchErrors: turn.metadata.web_fetch_errors, academicUsed: turn.metadata.academic_used,
  time: displayTime(turn.created_at),
});

function EmptyHint({ title, detail }: { title: string; detail: string }) {
  return <div className="empty-hint"><div className="empty-icon"><FileSearch size={22} /></div><strong>{title}</strong><p>{detail}</p></div>;
}

/** Mount React Flow only when expanded; hidden <details> cannot be measured by fitView. */
function SavedWorkflowDetails({ message }: { message: ChatMessage }) {
  const [open, setOpen] = useState(false);
  return <details className="research-workflow-result" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary><Workflow size={14}/> Executed visual workflow · {message.workflowSteps?.length} steps</summary>
    {open && (message.workflowGraph?.nodes?.length && message.workflowGraph.edges ?
      <LiveWorkflowCanvas key={message.id}
        progress={progressFromSavedRun(message.workflowGraph, message.workflowSteps || [], message.traceId)}
        savedRun={message.workflowAgentId && /^\d+$/.test(message.id)
          ? { profileId: message.workflowAgentId, turnId: Number(message.id) } : undefined}/>
      : <ol>{message.workflowSteps?.map((step,i)=><li key={`${i}-${step.node}`}>
          <strong>{step.node.replaceAll("_"," ")}</strong> · {step.status}
          {step.status === "completed" && step.latency_ms > 0 ? ` · ${step.latency_ms} ms` : ""}
          {step.sources > 0 ? ` · ${step.sources} sources` : ""}{step.reason ? ` · ${step.reason}` : ""}
        </li>)}</ol>)}
  </details>;
}

export function ResearchWorkspace() {
  const [view, setView] = useState<WorkspaceView>("research");
  const [inspector, setInspector] = useState<InspectorView>("evidence");
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(254);
  useEffect(() => {
    try {
      const saved = Number(window.localStorage.getItem("evidence-lab.sidebar-width"));
      if (saved >= 254 && saved <= 520) setSidebarWidth(saved);
    } catch { /* Storage disabled; resizing still works for this tab. */ }
  }, []);

  const dragSidebar = useRef<{x:number; width:number} | null>(null);
  const generationChangedForChat = useRef(false);
  const [chatNavExpanded, setChatNavExpanded] = useState(true);
  const [paperNavExpanded, setPaperNavExpanded] = useState(true);
  const [recentsDialog, setRecentsDialog] = useState<"research" | "paper" | null>(null);
  const [paperHistory, setPaperHistory] = useState<PaperSession[]>([]);
  const [paperHistoryLoaded, setPaperHistoryLoaded] = useState(false);
  const [paperSessionId, setPaperSessionId] = useState("");
  const [paperNewCounter, setPaperNewCounter] = useState(0);
  const paperViewNonceRef = useRef(0);
  useEffect(() => { setPaperHistory(readPaperHistory()); setPaperHistoryLoaded(true); }, []);
  useEffect(() => {
    if (!paperHistoryLoaded) return;
    try { writePaperHistory(paperHistory); }
    catch { setError("Paper Explorer history could not be saved in this browser (storage may be full or disabled)."); }
  }, [paperHistory, paperHistoryLoaded]);

  const [settingsOpen, setSettingsOpen] = useState(false);
  useEffect(() => {
    if (!settingsOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => { if (event.key === "Escape") setSettingsOpen(false); };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [settingsOpen]);
  const [mobileMenu, setMobileMenu] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [backendError, setBackendError] = useState("");
  const [documents, setDocuments] = useState<DocumentList>({ documents: [] });
  const [sessionId, setSessionId] = useState("");
  const [conversations, setConversations] = useState<ConversationRecord[]>([]);
  const [historyHasMore, setHistoryHasMore] = useState(false);
  const [historyCursor, setHistoryCursor] = useState<number | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [listHasMore, setListHasMore] = useState(false);
  const [listLoading, setListLoading] = useState(false);
  const [context, setContext] = useState<WorkingContext>({ session_id: "", revision: 0, notes: [] });
  const [events, setEvents] = useState<ContextEvent[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<ModelMode>("answer");
  const [darkMode, setDarkMode] = useState(false);
  useEffect(() => {
    try { const mode = window.localStorage.getItem("evidence-lab.theme"); setDarkMode(mode === "dark"); document.documentElement.dataset.theme = mode === "dark" ? "dark" : "light"; }
    catch { document.documentElement.dataset.theme = "light"; }
  }, []);
  function toggleTheme() {
    setDarkMode(old => { const next = !old; document.documentElement.dataset.theme = next ? "dark" : "light"; try {window.localStorage.setItem("evidence-lab.theme", next ? "dark" : "light");} catch {} return next; });
  }
  const [agentProfiles, setAgentProfiles] = useState<AgentProfile[]>([{id:"default", name:"Research assistant", description:"Built-in profile", system_instructions:"", enabled_tools:["local_search", "conversation_search", "web_search", "academic_search", "context_read", "context_edit"]}]);
  const [agentProfileId, setAgentProfileId] = useState("default");
  const [ragSystems, setRagSystems] = useState<RAGSystem[]>([]);
  const [ragSystemId, setRagSystemId] = useState("default");
  const refreshRagSystems = useCallback(async () => {
    const result = await api<RAGSystems>("/rag-systems");
    setRagSystems(result.systems);
    setRagSystemId(current=>result.systems.some(system=>system.id===current)?current:"default");
  }, []);
  useEffect(()=>{void refreshRagSystems().catch(()=>{});},[refreshRagSystems]);
  // RAG uses the built-in profile; custom profiles apply only to Agentic RAG.
  const currentChatAgentId = mode === "chat" ? agentProfileId : "default";
  const [useWeb, setUseWeb] = useState(false);
  const [webPages, setWebPages] = useState(2);
  const [useAcademic, setUseAcademic] = useState(false);
  const [verificationMode, setVerificationMode] = useState<"semantic" | "tev1" | "extractive">("semantic");
  const [memoryRoute, setMemoryRoute] = useState<MemoryRoute>("auto");
  const [openedMemoryMessage, setOpenedMemoryMessage] = useState<string | null>(null);
  const [memoryDetails, setMemoryDetails] = useState<Record<string, ConversationMemoryHit[]>>({});
  const [retrieval, setRetrieval] = useState<RetrievalOptions>({
    mode: "sparse", embedding_model: "sentence-transformers/all-MiniLM-L6-v2", top_k: 5, bm25_candidates: 20, dense_candidates: 20,
    rrf_k: 60, min_similarity: 0, reranker: "none", rerank_candidates: 20,
  });
  const [generation, setGeneration] = useState<GenerationOptions>({...DEFAULT_GENERATION});
  const [vectorStatus, setVectorStatus] = useState<VectorStatus | null>(null);
  const [reindexing, setReindexing] = useState(false);
  const [question, setQuestion] = useState("");
  const [working, setWorking] = useState(false);
  const [liveStages, setLiveStages] = useState<string[]>([]);
  const [liveWorkflow, setLiveWorkflow] = useState<LiveWorkflowProgress | null>(null);
  const [showLiveWorkflow, setShowLiveWorkflow] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [toast, setToast] = useState("");
  const [error, setError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchKind, setSearchKind] = useState<"local" | "web" | "academic">("local");
  const [searchHits, setSearchHits] = useState<SearchHit[]>([]);
  const [webHits, setWebHits] = useState<WebHit[]>([]);
  const [academicHits, setAcademicHits] = useState<AcademicPaper[]>([]);
  const [academicErrors, setAcademicErrors] = useState<Record<string,string>>({});
  const [yearFrom, setYearFrom] = useState("");
  const [importUrl, setImportUrl] = useState("");
  const [importingPaper, setImportingPaper] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<EvidenceDetail | null>(null);
  const [inspectorSourceRefs, setInspectorSourceRefs] = useState<{ id: string; number: number }[] | null>(null);
  const [evidenceBusy, setEvidenceBusy] = useState(false);
  const [noteDraft, setNoteDraft] = useState<NoteDraft | null>(null);
  const [savingNote, setSavingNote] = useState(false);
  const [verificationResult, setVerificationResult] = useState<NoteResult | null>(null);
  const [comparison, setComparison] = useState({ original: "", edited: "" });
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [comparing, setComparing] = useState(false);
  const [selectedQuote, setSelectedQuote] = useState("");
  const conversationEnd = useRef<HTMLDivElement>(null);
  const conversationScroll = useRef<HTMLDivElement>(null);
  const loadedSession = useRef("");
  const selectionCounter = useRef(0);
  const olderMessagesLoading = useRef(false);

  const refreshDocuments = useCallback(async () => {
    const result = await api<DocumentList>("/documents");
    setDocuments(result);
  }, []);

  const refreshContext = useCallback(async (id: string) => {
    const [current, audit] = await Promise.all([
      api<WorkingContext>(`/context/${encodeURIComponent(id)}`),
      api<{ events: ContextEvent[] }>(`/context/${encodeURIComponent(id)}/events`),
    ]);
    setContext(current);
    setEvents(audit.events);
  }, []);

  const refreshConversations = useCallback(async () => {
    const result = await api<ConversationList>("/conversations?limit=50");
    setConversations(result.conversations);
    setListHasMore(result.has_more);
  }, []);

  async function openConversation(id: string) {
    if (working) return;
    const nonce = ++selectionCounter.current;
    setSessionId(id);
    loadedSession.current = id;
    setView("research"); setMobileMenu(false); setSelectedEvidence(null);
    setInspectorSourceRefs(null); setNoteDraft(null);
    setMessages([]); setHistoryLoading(true); setError("");
    setContext({ session_id: id, revision: 0, notes: [] }); setEvents([]);
    if (new URL(window.location.href).searchParams.get("chat") !== id) {
      const url = new URL(window.location.href);
      url.searchParams.set("chat", id);
      window.history.pushState({chatId:id}, "", url.pathname + url.search);
    }
    try {
      const page = await api<ConversationMessages>(`/conversations/${encodeURIComponent(id)}/messages?limit=40`);
      if (selectionCounter.current !== nonce) return;
      setMessages(page.messages.map(restoredMessage));
      const lastAssistant = [...page.messages].reverse().find(message => message.role === "assistant");
      setAgentProfileId(lastAssistant?.metadata.agent_profile_id || "default");
      setHistoryCursor(page.next_cursor); setHistoryHasMore(page.has_more);
    } catch (cause) {
      if (selectionCounter.current === nonce) setError(cause instanceof Error ? cause.message : "Could not load conversation");
    } finally { if (selectionCounter.current === nonce) setHistoryLoading(false); }
  }

  function showFreshChat() {
    if (working) return;
    ++selectionCounter.current;
    setSessionId(""); loadedSession.current = "";
    setMessages([]); setHistoryCursor(null); setHistoryHasMore(false); setHistoryLoading(false);
    setContext({session_id: "", revision: 0, notes: []}); setEvents([]);
    setAgentProfileId("default"); setQuestion(""); setError("");
    generationChangedForChat.current = false;
    const savedDefault = agentProfiles.find(agent => agent.id === "default")?.generation;
    setGeneration({...DEFAULT_GENERATION, ...(savedDefault || {})});
    setSelectedEvidence(null); setInspectorSourceRefs(null); setNoteDraft(null);
    setInspectorOpen(false); setSettingsOpen(false);
    setView("research"); setChatNavExpanded(true); setMobileMenu(false); setRecentsDialog(null);
    const url = new URL(window.location.href);
    if (url.searchParams.has("chat")) {
      url.searchParams.delete("chat");
      window.history.pushState({chatId:null}, "", url.pathname + url.search);
    }
  }

  // Kept as an async method because other workspace actions await chat setup.
  // Blank chats are intentionally not persisted until the first user message.
  async function createNewChat() { showFreshChat(); }

  async function renameChat(chat: ConversationRecord) {
    if (working) return;
    const title = window.prompt("Rename conversation", chat.title)?.trim();
    if (!title || title === chat.title) return;
    try {
      const updated = await patchJSON<ConversationRecord>(`/conversations/${chat.id}`, { title: title.slice(0, 120) });
      setConversations((items) => items.map((item) => item.id === chat.id ? updated : item));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not rename chat"); }
  }

  async function removeChat(chat: ConversationRecord) {
    if (working || !window.confirm(`Delete chat “${chat.title}”? Its messages will be deleted. Evidence and context audit records remain.`)) return;
    try {
      await deleteAPI(`/conversations/${chat.id}`);
      const remaining = conversations.filter((item) => item.id !== chat.id);
      setConversations(remaining);
      if (chat.id === sessionId) {
        if (remaining.length) await openConversation(remaining[0].id);
        else {
          ++selectionCounter.current;
          setSessionId(""); loadedSession.current = ""; setMessages([]);
          setHistoryHasMore(false); setHistoryCursor(null);
          setContext({ session_id: "", revision: 0, notes: [] }); setEvents([]);
          const url = new URL(window.location.href);
          url.searchParams.delete("chat");
          window.history.replaceState({chatId:null}, "", url.pathname + url.search);
        }
      }
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not delete chat"); }
  }


  function newPaperChat() {
    paperViewNonceRef.current += 1; setPaperSessionId(""); setPaperNewCounter(value => value + 1);
    setRecentsDialog(null); setView("literature"); setMobileMenu(false);
  }
  function openPaperChat(id: string) {
    if (!paperHistory.some(session => session.id === id)) return;
    paperViewNonceRef.current += 1; setPaperSessionId(id); setPaperNewCounter(value => value + 1); setRecentsDialog(null); setView("literature"); setMobileMenu(false);
  }
  function savePaperChat(session: PaperSession, originNonce: number) {
    setPaperHistory(previous => upsertPaperHistory(previous, session));
    // Don't bring an older in-flight response to the foreground if the user
    // has since opened another conversation or started a fresh draft.
    if (paperViewNonceRef.current === originNonce) setPaperSessionId(session.id);
  }
  function renamePaperChat(id: string) {
    const paper = paperHistory.find(item => item.id === id);
    if (!paper) return;
    const title = window.prompt("Rename paper exploration", paper.title)?.trim();
    if (title && title !== paper.title) setPaperHistory(previous => previous.map(item =>
      item.id === id ? {...item, title:title.slice(0,120)} : item));
  }
  function deletePaperChat(id: string) {
    const paper = paperHistory.find(item => item.id === id);
    if (!paper || !window.confirm(`Delete paper exploration “${paper.title}”?`)) return;
    setPaperHistory(previous => previous.filter(item => item.id !== id));
    if (paperSessionId === id) { paperViewNonceRef.current += 1; setPaperSessionId(""); setPaperNewCounter(value => value + 1); }
  }

  async function loadOlder() {
    if (!sessionId || !historyCursor || olderMessagesLoading.current) return;
    olderMessagesLoading.current = true;
    const viewport = conversationScroll.current;
    const oldHeight = viewport?.scrollHeight ?? 0;
    const oldTop = viewport?.scrollTop ?? 0;
    try {
      const page = await api<ConversationMessages>(`/conversations/${sessionId}/messages?limit=40&before=${historyCursor}`);
      if (loadedSession.current !== sessionId) { olderMessagesLoading.current = false; return; }
      setMessages((current) => [...page.messages.map(restoredMessage), ...current]);
      setHistoryCursor(page.next_cursor); setHistoryHasMore(page.has_more);
      requestAnimationFrame(() => {
        if (viewport) viewport.scrollTop = oldTop + viewport.scrollHeight - oldHeight;
        olderMessagesLoading.current = false;
      });
    } catch (cause) {
      olderMessagesLoading.current = false;
      setError(cause instanceof Error ? cause.message : "Could not load older messages");
    }
  }

  async function loadMoreChats() {
    if (listLoading || !listHasMore) return;
    setListLoading(true);
    try {
      const page = await api<ConversationList>(`/conversations?limit=50&offset=${conversations.length}`);
      setConversations((prev) => [...prev, ...page.conversations.filter((next) => !prev.some((old) => old.id === next.id))]);
      setListHasMore(page.has_more);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not load conversations"); }
    finally { setListLoading(false); }
  }

  const refreshAgentProfiles = useCallback(async () => {
    const result = await api<AgentProfiles>("/agents");
    setAgentProfiles(result.agents);
    setAgentProfileId(current => result.agents.some(a => a.id === current) ? current : "default");
    if (!generationChangedForChat.current) {
      setGeneration({...DEFAULT_GENERATION, ...(result.agents.find(agent => agent.id === "default")?.generation || {})});
    }
  }, []);

  useEffect(() => {
    void refreshAgentProfiles().catch(() => {});
  }, [refreshAgentProfiles]);

  // Profile edits can revoke tool access while a conversation is selected.
  // Keep composer toggles consistent with server-enforced permissions.
  useEffect(() => {
    const selected = agentProfiles.find(profile => profile.id === currentChatAgentId);
    if (!selected) return;
    if (!selected.enabled_tools.includes("web_search")) setUseWeb(false);
    if (!selected.enabled_tools.includes("academic_search")) setUseAcademic(false);
  }, [agentProfiles, currentChatAgentId]);

  useEffect(() => {
    let mounted = true;
    const setup = async () => {
      try {
        const [status, corpus] = await Promise.all([api<Health>("/health"), api<DocumentList>("/documents")]);
        if (mounted) { setHealth(status); setVectorStatus(status.vector_index ?? null); setDocuments(corpus); setBackendError(""); }
      } catch {
        if (mounted) setBackendError("FastAPI is not reachable. Start the Python backend on port 8000.");
      }
    };
    void setup();
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    let active = true;
    api<ConversationList>("/conversations?limit=50").then((result) => {
      if (!active) return;
      setConversations(result.conversations); setListHasMore(result.has_more);
      // New visits to / start with a fresh draft. Only an explicit ?chat=id
      // deep link is restored across refreshes (no localStorage auto-resume).
      const requested = new URL(window.location.href).searchParams.get("chat");
      // Open by stable ID even if the chat falls outside the first 50 list rows.
      if (requested) void openConversation(requested);
    }).catch((cause: unknown) => {
      if (active) setError(cause instanceof Error ? cause.message : "Could not load chat history");
    });
    return () => { active = false; };
  // Initial hydration only. Conversation changes are managed explicitly.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const onPopState = () => {
      const id = new URL(window.location.href).searchParams.get("chat");
      if (id && id !== loadedSession.current) void openConversation(id);
      else if (!id) showFreshChat();
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  // Registered once; handler reads the current URL when fired.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    let mounted = true;
    Promise.all([
      api<WorkingContext>(`/context/${encodeURIComponent(sessionId)}`),
      api<{ events: ContextEvent[] }>(`/context/${encodeURIComponent(sessionId)}/events`),
    ]).then(([current, audit]) => {
      if (mounted) { setContext(current); setEvents(audit.events); }
    }).catch((cause: unknown) => {
      if (mounted) setError(cause instanceof Error ? cause.message : "Couldn't load context");
    });
    return () => { mounted = false; };
  }, [sessionId]);

  useEffect(() => {
    if (!olderMessagesLoading.current) conversationEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, working]);

  useEffect(() => {
    if (!inspectorOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setInspectorOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [inspectorOpen]);

  useEffect(() => {
    let active = true;
    api<VectorStatus>(`/retrieval/status?embedding_model=${encodeURIComponent(retrieval.embedding_model)}`)
      .then((status) => { if (active) setVectorStatus(status); })
      .catch(() => { if (active) setVectorStatus(null); });
    return () => { active = false; };
  }, [retrieval.embedding_model]);

  const activeAssistantMessage = messages.filter((m) => m.role === "assistant").at(-1);
  const activeSourceRefs = activeAssistantMessage ? citedRefs(activeAssistantMessage) : [];
  const currentSourceRefs = inspectorSourceRefs ?? activeSourceRefs;
  const activeSourceDetails = messages.filter((m) => m.role === "assistant").at(-1)?.sourceDetails || [];
  const archivedSourceDetails = messages.flatMap((m) => m.sourceDetails || []);
  const activeDocuments = documents.documents.filter((d) => d.active);

  function notify(message: string) {
    setToast(message);
    // Avoid timers: notification clears on the next user action or request.
  }

  async function inspectEvidence(id: string) {
    setInspector("evidence");
    setInspectorOpen(true);
    setEvidenceBusy(true);
    setSelectedQuote("");
    setError("");
    try {
      setSelectedEvidence(await api<EvidenceDetail>(`/evidence/${encodeURIComponent(id)}`));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load evidence");
    } finally {
      setEvidenceBusy(false);
    }
  }

  function setChatGeneration(next: GenerationOptions) {
    generationChangedForChat.current = true;
    setGeneration(next);
  }

  function chooseAgentProfile(id: string) {
    setAgentProfileId(id);
    const allowed = agentProfiles.find(agent => agent.id === id)?.enabled_tools || [];
    if (!allowed.includes("web_search")) setUseWeb(false);
    if (!allowed.includes("academic_search")) setUseAcademic(false);
  }

  async function sendQuestion(input?: string) {
    const query = (input ?? question).trim();
    if (!query || working) return;
    setView("research");
    setError(""); setToast(""); setQuestion(""); setWorking(true);
    setLiveStages(["Preparing research request"]);
    setLiveWorkflow(null); setShowLiveWorkflow(true);
    const progressId = crypto.randomUUID();
    let progressTimer: ReturnType<typeof setInterval> | undefined;
    let activeId = sessionId;
    const isFirstMessage = !activeId;
    const optimisticId = messageId();
    try {
      if (!activeId) {
        const created = await postJSON<ConversationRecord>("/conversations", { title: "New chat" });
        activeId = created.id;
        setSessionId(created.id); loadedSession.current = created.id;
        setConversations((items) => [created, ...items]);
        const url = new URL(window.location.href);
        url.searchParams.set("chat", created.id);
        window.history.pushState({chatId:created.id}, "", url.pathname + url.search);
      }
      const userMessage: ChatMessage = { id: optimisticId, role: "user", content: query, time: now() };
      setMessages((list) => [...list, userMessage]);
      // Progress polling reports server-observed stages; it is not a fake
      // reasoning transcript and cannot expose hidden model thoughts.
      progressTimer = setInterval(() => {
        void api<{history:string[];workflow?:LiveWorkflowProgress}>(`/chat/progress/${progressId}`).then(progress => {
          if (progress.history.length) setLiveStages(progress.history);
          if (progress.workflow) setLiveWorkflow(progress.workflow);
        }).catch(() => {});
      }, 850);
      await postJSON<AnswerResponse>(mode === "answer" ? "/answer" : "/chat", {
        progress_id: progressId, question: query, session_id: activeId, agent_profile_id: currentChatAgentId, rag_system_id: mode === "answer" ? ragSystemId : "default", retrieval, generation, memory_route: memoryRoute,
        use_web: useWeb, web_fetch_pages: webPages, use_academic: useAcademic,
      });
      // Reload the durable exchange from SQL (including citations and metadata).
      const page = await api<ConversationMessages>(`/conversations/${activeId}/messages?limit=40`);
      if (loadedSession.current === activeId) {
        setMessages(page.messages.map(restoredMessage));
        setHistoryCursor(page.next_cursor); setHistoryHasMore(page.has_more);
        setSelectedEvidence(null); setInspectorSourceRefs(null); setInspector("evidence");
        await refreshContext(activeId);
      }
      await refreshConversations();
      // Small local title model runs after the response is sent. Refresh sidebar
      // metadata once it has had a chance to finish; no chat content is reloaded.
      if (isFirstMessage) {
        window.setTimeout(() => { void refreshConversations(); }, 1800);
        window.setTimeout(() => { void refreshConversations(); }, 6500);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The model request failed");
      // Failed requests are not persisted: leave the entered question editable for retry.
      setQuestion(query);
      setMessages((list) => list.filter((message) => message.id !== optimisticId));
    } finally { if (progressTimer) clearInterval(progressTimer); setWorking(false); setLiveStages([]); setLiveWorkflow(null); }
  }

  async function toggleConversationEvidence(message: ChatMessage) {
    if (openedMemoryMessage === message.id) { setOpenedMemoryMessage(null); return; }
    setOpenedMemoryMessage(message.id);
    const ids = (message.conversationSources || []).map(source => source.turn_id);
    if (!sessionId || !ids.length || memoryDetails[message.id]) return;
    try {
      const result = await api<{ results: ConversationMemoryHit[] }>(
        `/conversations/${sessionId}/memory/turns?ids=${ids.join(",")}`);
      setMemoryDetails(current => ({ ...current, [message.id]: result.results }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load archived turns");
    }
  }

  async function runSearch(kind: "local" | "web" | "academic") {
    if (!searchQuery.trim() || searching) return;
    setError(""); setToast(""); setSearching(true); setSearchKind(kind);
    try {
      if (kind === "local") {
        const result = await postJSON<{ results: SearchHit[] }>("/retrieval/search", { query: searchQuery.trim(), retrieval });
        setSearchHits(result.results); setWebHits([]); setAcademicHits([]);
      } else if (kind === "web") {
        const result = await postJSON<{ results: WebHit[] }>("/web/search", { query: searchQuery.trim(), top_k: 4 });
        setWebHits(result.results); setSearchHits([]); setAcademicHits([]);
      } else {
        const result = await postJSON<AcademicResult>("/academic/search", {
          query: searchQuery.trim(), top_k: 8,
          year_from: yearFrom ? Number(yearFrom) : null,
          sources: ["arxiv", "openalex"],
        });
        setAcademicHits(result.results); setAcademicErrors(result.errors);
        setWebHits([]); setSearchHits([]);
      }
      setSelectedEvidence(null); setInspector("evidence");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Search failed");
    } finally { setSearching(false); }
  }

  async function importAcademicPdf(url: string) {
    if (!url.trim() || importingPaper) return;
    // Explicit user approval is required for each remote download.
    if (!window.confirm(`Download and index this PDF?\n${url}`)) return;
    setImportingPaper(true); setError(""); setToast("");
    try {
      const result = await postJSON<IngestResult>("/academic/import", { url: url.trim() });
      await refreshDocuments();
      notify(`Paper ${result.status}: ${shortName(result.path)} · ${result.chunks} chunks`);
      setImportUrl("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "PDF import failed");
    } finally { setImportingPaper(false); }
  }

  async function reindexCorpus() {
    setReindexing(true); setError("");
    try {
      const status = await postJSON<VectorStatus & { embedded: number }>("/retrieval/reindex", { embedding_model: retrieval.embedding_model });
      setVectorStatus(status); notify(`Indexed ${status.embedded} new passages.`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Indexing failed"); }
    finally { setReindexing(false); }
  }

  async function ingestCorpus() {
    setIngesting(true); setError(""); setToast("");
    try {
      const result = await postJSON<IngestResponse>("/ingest/sync", {});
      await refreshDocuments();
      if (result.vector_index) setVectorStatus(result.vector_index);
      if (result.vector_index_error) setError(result.vector_index_error);
      notify(`Ingestion finished: ${result.summary.ingested ?? 0} added, ${result.summary.skipped ?? 0} unchanged, ${result.summary.failed ?? 0} failed.`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Ingestion failed"); }
    finally { setIngesting(false); }
  }

  async function uploadFile(file: File) {
    setUploading(true); setError(""); setToast("");
    try {
      const data = new FormData();
      data.append("file", file);
      const result = await api<{ path: string; chunks: number; status: string; vector_index?: VectorStatus; vector_index_error?: string }>("/documents/upload", { method: "POST", body: data });
      await refreshDocuments();
      if (result.vector_index) setVectorStatus(result.vector_index);
      if (result.vector_index_error) setError(result.vector_index_error);
      notify(`${shortName(result.path)} ${result.status} · ${result.chunks} chunks`);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Upload failed"); }
    finally { setUploading(false); }
  }

  function draftFromEvidence() {
    if (!selectedEvidence) return;
    setNoteDraft({ evidenceId: selectedEvidence.id, quote: selectedQuote, claim: selectedQuote });
    setVerificationResult(null);
    setInspector("memory");
    setInspectorOpen(true);
  }

  async function saveNote() {
    if (!noteDraft || savingNote || !sessionId) return;
    setSavingNote(true); setError(""); setVerificationResult(null); setToast("");
    try {
      const result = await postJSON<NoteResult>(`/context/${encodeURIComponent(sessionId)}/notes`, {
        claim: noteDraft.claim, quote: noteDraft.quote, evidence_id: noteDraft.evidenceId,
        replace_note_id: noteDraft.replaceId || null, expected_revision: context.revision,
        verification_mode: verificationMode,
      });
      setVerificationResult(result);
      setNoteDraft(null);
      await refreshContext(sessionId);
      notify(`Note accepted at revision ${result.revision}${result.similarity != null ? ` · similarity ${result.similarity.toFixed(3)}` : ""}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Context edit rejected");
      await refreshContext(sessionId).catch(() => {});
    } finally { setSavingNote(false); }
  }

  async function deleteNote(noteId: string) {
    if (!sessionId || !window.confirm("Remove this working-context note? The event will remain in the audit log.")) return;
    setError(""); setToast("");
    try {
      await api(`/context/${encodeURIComponent(sessionId)}/notes/${noteId}`, {
        method: "DELETE", body: JSON.stringify({ expected_revision: context.revision }),
      });
      await refreshContext(sessionId); notify("Note removed. Audit event retained.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not remove note"); }
  }

  async function compareTexts() {
    if (!comparison.original.trim() || !comparison.edited.trim()) return;
    setComparing(true); setCompareResult(null); setError(""); setToast("");
    try {
      setCompareResult(await postJSON<CompareResult>("/context/compare", { ...comparison, verification_mode: verificationMode }));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Comparison failed"); }
    finally { setComparing(false); }
  }

  function useQuoteSelection() {
    const selected = window.getSelection()?.toString() || "";
    // Verify exact substring before offering it to the source-grounded checker.
    if (selectedEvidence && selected && selectedEvidence.text.includes(selected)) setSelectedQuote(selected);
  }

  const renderSourceList = (refs: { id: string; number: number }[]) => (
    <div className="source-list">
      {refs.map(({ id, number }) => <button key={id} className="source-row" type="button" onClick={() => void inspectEvidence(id)}>
        <span className="source-index">{number}</span>
        <span className="source-row-text"><strong>{(() => { const hit = archivedSourceDetails.find(h=>h.id===id) || searchHits.find(h=>h.id===id); return sourceLabel(hit ? {...hit, id} : undefined, webHits.find(h=>h.evidence_id===id)?.title || academicHits.find(h=>h.evidence_id===id)?.title || `Evidence ${shortId(id)}`); })()}</strong><small>{(() => { const hit = searchHits.find((item) => item.id === id) || archivedSourceDetails.find((item) => item.id === id); return hit?.source_kind === "academic_metadata_unverified" ? "Academic abstract (metadata only)" : hit?.source_kind === "web_snippet_unverified" ? "Unverified web snippet" : hit ? `Page ${hit.page ?? "?"} · v${hit.version}` : "Inspect retrieved excerpt"; })()}</small></span>
        <ArrowRight size={14} />
      </button>)}
    </div>
  );

  const sidebarResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (sidebarCollapsed || window.innerWidth <= 700) return;
    dragSidebar.current = {x: event.clientX, width: sidebarWidth};
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const moveSidebar = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragSidebar.current) return;
    const {x,width} = dragSidebar.current;
    const next = Math.max(254, Math.min(520, width + event.clientX - x));
    setSidebarWidth(next);
    try { window.localStorage.setItem("evidence-lab.sidebar-width", String(next)); } catch { /* optional */ }
  };
  const endSidebarResize = () => {dragSidebar.current = null;};

  return <div className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`} style={{"--expanded-sidebar-width": `${sidebarWidth}px`} as CSSProperties}>
    {mobileMenu && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setMobileMenu(false)} />}
    <aside className={`sidebar ${mobileMenu ? "sidebar-show" : ""}`}>
      <div className="brand"><div className="brand-mark"><ResearchLogo size={25} /></div><div className="brand-copy"><strong>AIRA</strong><span>AI Research Assistant</span></div><button type="button" className="sidebar-collapse-control" aria-label={sidebarCollapsed ? "Expand navigation" : "Collapse navigation"} title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"} onClick={() => setSidebarCollapsed((value) => !value)}>{sidebarCollapsed ? <PanelLeftOpen size={19}/> : <PanelLeftClose size={19}/>}</button></div>
      <nav className="nav-list" aria-label="Workspace navigation">
        <div className="nav-parent">
          <button type="button" title="Research chat" className={`nav-entry ${view === "research" ? "active" : ""}`} onClick={showFreshChat}><MessageSquareText size={iconSize} /> <span className="nav-label">Research chat</span></button>
          <button type="button" className="nav-section-toggle" aria-label={chatNavExpanded ? "Collapse conversations" : "Expand conversations"} title={chatNavExpanded ? "Collapse conversations" : "Expand conversations"} aria-expanded={chatNavExpanded} aria-controls="sidebar-conversations" onClick={() => setChatNavExpanded(expanded => !expanded)}><ChevronDown size={15} className={chatNavExpanded ? "" : "nav-chevron-closed"}/></button>
        </div>
        {chatNavExpanded && <div className="nav-subsection" id="sidebar-conversations">
          <div className="sidebar-chat-actions">
            <button type="button" className="new-chat-button" onClick={() => void createNewChat()} disabled={working} title="Create new research conversation"><Plus size={15}/><span>New chat</span></button>
            <button type="button" className="sidebar-recents-button" onClick={() => {setRecentsDialog("research");setMobileMenu(false);void refreshConversations().catch(()=>{});}} aria-haspopup="dialog" title="Browse previous research chats"><History size={15}/><span>Recents</span></button>
          </div>
        </div>}
        <div className="nav-parent">
          <button type="button" title="Paper explorer" className={`nav-entry ${view === "literature" ? "active" : ""}`} onClick={() => { setView("literature"); setMobileMenu(false); }}><BookOpen size={iconSize} /> <span className="nav-label">Paper explorer</span></button>
          <button type="button" className="nav-section-toggle" aria-label={paperNavExpanded ? "Collapse paper conversations" : "Expand paper conversations"} aria-expanded={paperNavExpanded} aria-controls="sidebar-paper-conversations" onClick={() => setPaperNavExpanded(expanded => !expanded)}><ChevronDown size={15} className={paperNavExpanded ? "" : "nav-chevron-closed"}/></button>
        </div>
        {paperNavExpanded && <div className="nav-subsection" id="sidebar-paper-conversations">
          <div className="sidebar-chat-actions">
            <button type="button" className="new-chat-button" onClick={newPaperChat} title="Start a new paper exploration"><Plus size={15}/><span>New chat</span></button>
            <button type="button" className="sidebar-recents-button" onClick={() => {setRecentsDialog("paper");setMobileMenu(false);}} aria-haspopup="dialog" title="Browse previous paper explorations"><History size={15}/><span>Recents</span></button>
          </div>
        </div>}
      </nav>
      <nav className="sidebar-bottom-nav" aria-label="Configuration and project information">
        <button type="button" title="Configuration" className={`nav-entry ${["configuration", "agents", "rag", "library", "traces", "experiments", "retrieval-eval"].includes(view) ? "active" : ""}`} onClick={() => { setView("configuration"); setMobileMenu(false); }}><Settings2 size={iconSize}/><span className="nav-label">Configuration</span></button>
        <button type="button" title="About AIRA" className={`nav-entry ${view === "info" ? "active" : ""}`} onClick={() => { setView("info"); setMobileMenu(false); }}><HelpCircle size={iconSize}/><span className="nav-label">About the project</span></button>
      </nav>
      <div className="sidebar-footer"><span className={`live-dot ${backendError ? "offline" : ""}`} /><span className="sidebar-footer-label">{backendError ? "Backend disconnected" : health ? "Backend connected" : "Checking backend"}</span><span className="sidebar-version">v0.1.0</span></div>
    </aside>
    <div className="sidebar-resize-handle" role="separator" aria-label="Resize sidebar" aria-orientation="vertical" tabIndex={0}
      aria-valuenow={sidebarWidth} aria-valuemin={254} aria-valuemax={520}
      onPointerDown={sidebarResize} onPointerMove={moveSidebar} onPointerUp={endSidebarResize} onPointerCancel={endSidebarResize}
      onKeyDown={(event) => {if (event.key === "ArrowRight" || event.key === "ArrowLeft") {event.preventDefault(); setSidebarWidth(w => {const next = Math.max(254, Math.min(520, w+(event.key === "ArrowRight" ? 20 : -20))); try {window.localStorage.setItem("evidence-lab.sidebar-width", String(next));} catch {} return next;});}}}
    />

    <main className="main-area">
      <header className="topbar"><div className="topbar-heading"><button className="icon-button mobile-menu-btn" aria-label="Open navigation" onClick={() => setMobileMenu(true)}><Menu size={20} /></button>
        {view === "research" ? <><button type="button" className="topbar-parent-link" onClick={showFreshChat}>Research chat</button><span className="topbar-divider">/</span><strong>{conversations.find((item) => item.id === sessionId)?.title || "New chat"}</strong></>
          : ["agents", "rag", "library", "traces", "experiments", "retrieval-eval"].includes(view) ? <><button type="button" className="topbar-parent-link" onClick={() => setView("configuration")}>Configuration</button><span className="topbar-divider">/</span><strong>{view === "agents" ? "Research agents" : view === "rag" ? "RAG systems" : view === "library" ? "Your library" : view === "traces" ? "Behind the scenes" : view === "retrieval-eval" ? "Retrieval evaluation" : "Experiments on Benchmark Datasets"}</strong></>
          : view === "literature" ? <><button type="button" className="topbar-parent-link" onClick={newPaperChat}>Paper explorer</button><span className="topbar-divider">/</span><strong>{paperHistory.find(item => item.id === paperSessionId)?.title || "New chat"}</strong></>
          : <strong>{view === "configuration" ? "Configuration" : view === "info" ? "About the project" : "New chat"}</strong>}
      </div><button type="button" className="theme-toggle" aria-label={darkMode ? "Use light mode" : "Use dark mode"} aria-pressed={darkMode} onClick={toggleTheme}>{darkMode ? "☀ Light mode" : "☾ Dark mode"}</button></header>
      {(error || backendError) && <div className="alert-banner" role="alert"><CircleAlert size={17} /><span>{error || backendError}</span><button onClick={() => { setError(""); setBackendError(""); }} aria-label="Dismiss error"><X size={16} /></button></div>}
      {toast && <div className="success-banner" role="status"><CircleCheck size={17} /><span>{toast}</span><button onClick={() => setToast("")} aria-label="Dismiss"><X size={15} /></button></div>}
      {["agents", "rag", "library", "traces", "experiments", "retrieval-eval"].includes(view) && <div className="config-back-container"><button type="button" className="config-back-control" onClick={() => setView("configuration")}><ArrowLeft size={16}/> Back to Configuration</button></div>}
      {view === "configuration" ? <ConfigurationWorkspace documentCount={activeDocuments.length} chunkCount={activeDocuments.reduce((sum,doc)=>sum+(doc.chunks||0),0)} agentCount={agentProfiles.length} ragCount={ragSystems.length} onNavigate={(destination: ConfigurationDestination) => { setView(destination); setMobileMenu(false); }}/> : view === "library" ? <LibraryWorkspace documents={documents.documents} uploading={uploading} syncing={ingesting} upload={uploadFile} sync={ingestCorpus}/> : view === "traces" ? <TracesWorkspace conversationTitles={Object.fromEntries(conversations.map(item => [item.id, item.title]))}/> : view === "retrieval-eval" ? <RetrievalEvaluationWorkspace/> : view === "experiments" ? <ExperimentsWorkspace/> : view === "info" ? <InfoWorkspace/> : view === "rag" ? <RAGWorkspace systems={ragSystems} selectedId={ragSystemId} onSelect={(id)=>{setRagSystemId(id);setView("research");}} onChanged={refreshRagSystems}/> : view === "agents" ? <AgentWorkspace profiles={agentProfiles} selectedId={agentProfileId} onSelect={(id) => { chooseAgentProfile(id); setMode("chat"); setView("research"); }} onChanged={async () => {generationChangedForChat.current = false; await refreshAgentProfiles();}} /> : view === "literature" ? <LiteratureWorkspace key={paperNewCounter} generation={generation}
        session={paperHistory.find(item => item.id === paperSessionId) ?? null} onSave={session => savePaperChat(session, paperNewCounter)} onNewTopic={newPaperChat}/> : <>
        <div className="conversation" ref={conversationScroll} role="log" aria-label="Research conversation">
          {historyLoading ? <div className="loading-panel"><LoaderCircle className="spin" size={18}/> Loading conversation…</div> : messages.length === 0 ? <div className="welcome">
            <h1>Hey fellow nerd, ready to make sense of <em>complex research?</em></h1>
            <div className="prompts-grid">{EXAMPLE_QUESTIONS.map((item) => <button key={item.title} className="prompt-card" onClick={() => void sendQuestion(item.prompt)}><span className="prompt-icon"><ArrowRight size={17} /></span><strong>{item.title}</strong><span>{item.detail}</span></button>)}</div>
          </div> : <div className="messages">
            {historyHasMore && <button type="button" className="load-older" onClick={() => void loadOlder()} disabled={olderMessagesLoading.current}>Load earlier messages</button>}
            {messages.map((message) => <div key={message.id} className={`message message-${message.role}`}>
              {message.role === "assistant" ? <div className="message-avatar"><ResearchLogo size={20}/></div> : null}
              <div className="message-content"><div className="message-header">{message.role === "assistant" ? "Research assistant" : "You"}<span>{message.time}</span></div>
                {message.role === "assistant" ? <CitationMarkdown content={message.content} sourceIds={message.sourceIds || []} onOpenSource={(id) => { setInspectorSourceRefs(citedRefs(message)); void inspectEvidence(id); }} /> : <p className="user-question">{message.content}</p>}
                {message.role === "assistant" && citedRefs(message).length > 0 &&
                  <div className="answer-sources" aria-label="Cited sources"><strong>Sources:</strong>{citedRefs(message).map(({id, number}) => {
                    const item = message.sourceDetails?.find(source => source.id === id);
                    return <button key={id} type="button" title="Inspect source and original evidence"
                      onClick={() => {setInspectorSourceRefs(citedRefs(message)); void inspectEvidence(id);}}>
                      <span className="answer-source-number">[{number}]</span> {citedSourceLabel(item ? {...item,id} : undefined, `Evidence ${shortId(id)}`)}
                    </button>;
                  })}</div>}
                {message.role === "assistant" && (message.workflowSteps?.length || 0) > 0 &&
                  <SavedWorkflowDetails message={message}/>}
                {message.role === "assistant" && <div className="message-meta">{message.model && <span><Layers3 size={13} /> {message.model}</span>}{message.mode && <span>{message.mode === "answer" ? "RAG" : "Agentic RAG"}</span>}{message.memoryRoute && <span title="Selected active context archives"><Database size={13}/> {message.memoryRoute === "both" ? "Conversation + papers" : message.memoryRoute === "conversation" ? "Conversation memory" : message.memoryRoute === "research" ? "Research evidence" : "No archive search"}{message.conversationSources?.length ? ` · ${message.conversationSources.length} prior turns` : ""}</span>}{message.webUsed && <span><Globe2 size={13}/> Web search · {message.webPagesFetched ?? 0} pages read</span>}{message.webFetchErrors && message.webFetchErrors.length > 0 && <span title={message.webFetchErrors.map(e => `${e.url}: ${e.reason}`).join("\n")}><Info size={13}/> {message.webFetchErrors.length} page fetch failures</span>}{message.academicUsed && <span><BookOpen size={13}/> Academic APIs enabled</span>}{message.traceId && <button type="button" title="Copy this research run’s Phoenix trace ID" onClick={() => { void navigator.clipboard.writeText(message.traceId!).then(() => notify("Phoenix trace ID copied. Search for it in Phoenix."), () => notify("Could not copy the trace ID.")); }}><Activity size={13}/> Copy trace ID</button>}{message.conversationSources && message.conversationSources.length > 0 && <button type="button" onClick={() => void toggleConversationEvidence(message)} aria-expanded={openedMemoryMessage === message.id} title="Inspect archived conversation turns used as context"><History size={13}/> View {message.conversationSources.length} earlier turns</button>}{citedRefs(message).length > 0 && <button onClick={() => { setInspectorSourceRefs(citedRefs(message)); setInspectorOpen(true); setInspector("evidence"); setSelectedEvidence(null); }}><Link2 size={13} /> {citedRefs(message).length} cited sources <ArrowRight size={12} /></button>}</div>}{message.role === "assistant" && openedMemoryMessage === message.id && <section className="conversation-memory-inspector" aria-label="Archived conversation evidence"><strong>Earlier conversation (not scientific evidence)</strong>{(memoryDetails[message.id] || []).length ? (memoryDetails[message.id] || []).map(turn => <div key={turn.turn_id}><small>Turn {turn.turn_id} · {turn.role}</small><p>{turn.text}</p></div>) : <p>Loading archived turns…</p>}</section>}
              </div>
            </div>)}
            {working && <div className="message message-assistant"><div className="message-avatar"><ResearchLogo size={20}/></div><div className="thinking research-live-activity" aria-live="polite"><div><LoaderCircle size={17} className="spin"/><strong>{liveStages.at(-1) || "Preparing research request"}…</strong></div><ol>{liveStages.slice(-4).map((stage,index)=><li key={`${index}-${stage}`}>{stage}{index===liveStages.slice(-4).length-1 ? "…" : " ✓"}</li>)}</ol><small>Live operation updates · Not the model’s private reasoning or a token stream</small>{liveWorkflow && <><button type="button" className="aira-live-toggle" aria-expanded={showLiveWorkflow} onClick={()=>setShowLiveWorkflow(open=>!open)}>{showLiveWorkflow ? "Hide" : "Show"} live workflow</button>{showLiveWorkflow && <LiveWorkflowCanvas progress={liveWorkflow}/>}</>}</div></div>}
            <div ref={conversationEnd} />
          </div>}
        </div>
        <div className="composer-area shared-chat-composer"><form className={`composer ${messages.length === 0 ? "composer-new-chat" : ""}`} onSubmit={(event) => { event.preventDefault(); void sendQuestion(); }}>
          <textarea placeholder="Ask a question about your research…" value={question} onChange={(e) => setQuestion(e.target.value)} rows={1} aria-label="Research question" onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void sendQuestion(); } }} />
          <div className="composer-footer composer-footer-simplified">
            <button type="button" className={`chat-config-trigger ${settingsOpen ? "active" : ""}`} aria-label="Configure research chat" title="Configure research chat" aria-expanded={settingsOpen} aria-haspopup="dialog" onClick={() => setSettingsOpen(true)}><Settings2 size={21}/></button>
            <button className="send-button" disabled={working || !question.trim()} title="Send question" aria-label="Send question"><Send size={17}/></button>
          </div>
        </form><p className="composer-note"><ShieldCheck size={12} /> Responses may contain mistakes. Verify key claims against the linked evidence.</p></div>
        {settingsOpen && <div className="chat-config-overlay" onMouseDown={event => { if (event.target === event.currentTarget) setSettingsOpen(false); }}>
          <section className="chat-config-dialog" role="dialog" aria-modal="true" aria-labelledby="chat-config-title">
            <header className="chat-config-header"><div><div className="eyebrow">RESEARCH CHAT</div><h2 id="chat-config-title">Chat configuration</h2><p>Choose how AIRA should answer your questions.</p></div><button type="button" aria-label="Close chat configuration" className="chat-config-close" onClick={() => setSettingsOpen(false)}><X size={19}/></button></header>
            <div className="chat-config-body">
              <div className="chat-config-section"><h3>Research mode</h3><div className="chat-config-modes" role="group" aria-label="Research mode">
                <button type="button" className={`chat-config-mode ${mode === "answer" ? "selected" : ""}`} aria-pressed={mode === "answer"} onClick={() => setMode("answer")}><BookOpen size={19}/><strong>RAG</strong><span>Answer using indexed papers and your selected evidence sources. Predictable retrieval before generation.</span></button>
                <button type="button" className={`chat-config-mode ${mode === "chat" ? "selected" : ""}`} aria-pressed={mode === "chat"} onClick={() => setMode("chat")}><BrainCircuit size={19}/><strong>Agentic RAG</strong><span>Let the agent choose tools and run follow-up searches as needed.</span></button>
              </div></div>
              {mode === "answer" && <div className="chat-config-section"><label className="chat-config-field" htmlFor="chat-config-rag"><strong>RAG system</strong><span>Saved retrieval, generation and instructions</span></label><select id="chat-config-rag" value={ragSystemId} onChange={e=>setRagSystemId(e.target.value)}>{ragSystems.map(system=><option key={system.id} value={system.id}>{system.name}</option>)}</select></div>}
              {mode === "chat" && <div className="chat-config-section"><label className="chat-config-field" htmlFor="chat-config-agent"><strong>Research agent</strong><span>Instructions, model and tool permissions</span></label><select id="chat-config-agent" value={agentProfileId} onChange={event => chooseAgentProfile(event.target.value)}>{agentProfiles.map(profile => <option key={profile.id} value={profile.id}>{profile.name}</option>)}</select></div>}
              {(mode === "chat" || ragSystemId === "default") && <div className="chat-config-section">
                <label className="chat-config-field" htmlFor="chat-config-reranker"><strong>Reranking</strong><span>Optional second-stage ranking of locally retrieved paper passages</span></label>
                <select id="chat-config-reranker" value={retrieval.reranker ?? "none"} onChange={e=>setRetrieval({...retrieval,reranker:e.target.value as RetrievalOptions["reranker"]})}>
                  <option value="none">None (first-stage ranking)</option>
                  <option value="Alibaba-NLP/gte-reranker-modernbert-base">GTE ModernBERT · 149M</option>
                  <option value="jinaai/jina-reranker-v3.5">Jina v3.5 · 0.6B (non-commercial)</option>
                </select>
                {(retrieval.reranker ?? "none") !== "none" && <label className="chat-config-field">Candidate pool to rerank
                  <input type="number" min={retrieval.top_k} max={100} value={retrieval.rerank_candidates ?? 20} onChange={e=>setRetrieval({...retrieval,rerank_candidates:Number(e.target.value)})}/>
                </label>}
              </div>}
              {mode === "answer" && ragSystemId !== "default" && <p className="agent-note">The selected saved RAG system controls reranking. Edit it under Configuration → RAG systems.</p>}
              <div className="chat-config-section"><h3>Optional research sources</h3><div className="chat-config-options">
                <label><span><Globe2 size={19}/><strong>Web search</strong><small>Search and read public webpages</small></span><input type="checkbox" checked={useWeb} disabled={health?.web_enabled === false || !agentProfiles.find(a => a.id === currentChatAgentId)?.enabled_tools.includes("web_search")} onChange={event => setUseWeb(event.target.checked)}/></label>
                <label><span><BookOpen size={19}/><strong>Academic search</strong><small>Find arXiv and OpenAlex paper metadata</small></span><input type="checkbox" checked={useAcademic} disabled={!agentProfiles.find(a => a.id === currentChatAgentId)?.enabled_tools.includes("academic_search")} onChange={event => setUseAcademic(event.target.checked)}/></label>
              </div>{useWeb && <label className="chat-config-pages">Webpages to read <select value={webPages} aria-label="Web pages to read" onChange={event => setWebPages(Number(event.target.value))}>{[0,1,2,3].map(n => <option key={n} value={n}>{n}</option>)}</select></label>}</div>
              <div className="chat-config-agent-link"><span>Configure saved {mode === "answer" ? "retrieval systems" : "agent profiles"} in Configuration.</span><button type="button" onClick={() => { setSettingsOpen(false); setView(mode === "answer" ? "rag" : "agents"); setMobileMenu(false); }}><Settings2 size={16}/> Configure {mode === "answer" ? "RAG systems" : "research agents"} <ArrowRight size={15}/></button></div>
            </div>
            <footer className="chat-config-footer"><button type="button" onClick={() => setSettingsOpen(false)}>Done</button></footer>
          </section>
        </div>}
      </>}
    </main>

    {recentsDialog && <RecentChatsDialog kind={recentsDialog}
      items={recentsDialog === "research" ? conversations : paperHistory}
      activeId={recentsDialog === "research" ? (view === "research" ? sessionId : "") : (view === "literature" ? paperSessionId : "")}
      disabled={recentsDialog === "research" && working}
      onClose={() => setRecentsDialog(null)}
      onOpen={id => {if (recentsDialog === "research") {setRecentsDialog(null);void openConversation(id);} else openPaperChat(id);}}
      onRename={id => {if (recentsDialog === "research") {const chat = conversations.find(item=>item.id===id);if(chat) void renameChat(chat);} else renamePaperChat(id);}}
      onDelete={id => {if (recentsDialog === "research") {const chat = conversations.find(item=>item.id===id);if(chat) void removeChat(chat);} else deletePaperChat(id);}}
      hasMore={recentsDialog === "research" && listHasMore}
      loading={recentsDialog === "research" && listLoading}
      onMore={() => void loadMoreChats()}/>}
    {inspectorOpen && <div className="inspector-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) setInspectorOpen(false); }}><section className="inspector inspector-dialog" role="dialog" aria-modal="true" aria-label="Evidence and working context"><div className="inspector-header"><div><div className="eyebrow">RESEARCH CONTEXT</div><strong>Evidence & memory</strong></div><button className="icon-button" aria-label="Close inspector" onClick={() => setInspectorOpen(false)}><X size={18} /></button></div>
      <div className="inspector-tabs" role="tablist" aria-label="Inspector"><button role="tab" aria-selected={inspector === "evidence"} className={inspector === "evidence" ? "active" : ""} onClick={() => setInspector("evidence")}><FileSearch size={14}/> Evidence</button><button role="tab" aria-selected={inspector === "memory"} className={inspector === "memory" ? "active" : ""} onClick={() => setInspector("memory")}><BookOpen size={14}/> Notes <span>{context.notes.length}</span></button><button role="tab" aria-selected={inspector === "verify"} className={inspector === "verify" ? "active" : ""} onClick={() => setInspector("verify")}><ShieldCheck size={14}/> Check</button></div>
      <div className="inspector-body">
        {inspector === "evidence" && (evidenceBusy ? <div className="loading-panel"><LoaderCircle className="spin" size={20} /> Loading source…</div> : selectedEvidence ? <div className="evidence-details"><button className="back-link" onClick={() => setSelectedEvidence(null)}><ArrowLeft size={14} /> Back to sources</button><div className="section-label">IMMUTABLE EVIDENCE</div><h3>{shortName(selectedEvidence.document_path || selectedEvidence.source_url || "Web evidence")}</h3><div className="evidence-badges"><span>{selectedEvidence.source_kind === "web_snippet" ? "Unverified web snippet" : selectedEvidence.source_kind === "web_page" ? "Captured web page (unverified claims)" : selectedEvidence.source_kind === "academic_abstract" ? "Academic abstract (not full paper)" : `Page ${selectedEvidence.page ?? "?"}`}</span><span>{selectedEvidence.document_version ? `v${selectedEvidence.document_version}` : ""}</span><span className={selectedEvidence.source_active ? "source-active" : "source-stale"}>{selectedEvidence.source_active ? "Active" : "Inactive"}</span></div><div className="passage-title">RETRIEVED PASSAGE <span>select text to quote</span></div><div className="passage-content" onMouseUp={useQuoteSelection} onKeyUp={useQuoteSelection}>{selectedEvidence.text}</div>{selectedEvidence.source_url && <a href={selectedEvidence.source_url} target="_blank" rel="noopener noreferrer" className="evidence-link"><Globe2 size={14}/> Open original web URL <ArrowRight size={13}/></a>}<div className="evidence-id">SOURCE ID <code>{selectedEvidence.id}</code></div>{selectedQuote && <div className="quote-preview"><Check size={14} /> Selected {selectedQuote.trim().split(/\s+/).length} words for your note</div>}<button className="primary-wide" onClick={draftFromEvidence}><Plus size={16} /> Draft context note</button><div className="footnote-box"><Info size={15} /> Claims must be supported by an exact quote from this stored passage. Cosine similarity is only a heuristic.</div></div> : <div className="inspector-sources"><div className="section-label">CITED SOURCES</div>{currentSourceRefs.length ? renderSourceList(currentSourceRefs) : <EmptyHint title="No sources cited" detail="Only sources cited in the answer are listed here. Unused retrieval results are not citations." />}<div className="footnote-box"><Info size={15} /> Sources are immutable records. Context notes are editable, but evidence is not.</div></div>)}
        {inspector === "memory" && <div className="memory-panel"><div className="panel-heading"><div><span className="section-label">WORKING CONTEXT</span><h3>Saved research notes</h3></div><span className="revision-pill">REV {context.revision}</span></div><p className="panel-description">These notes are passed to the local model on each turn. Only active, source-backed notes are included.</p>
          {context.notes.length ? <div className="notes-list">{context.notes.map((note) => <div className="note-card" key={note.id}><div className="note-status"><span className={`tiny-status ${note.source_active ? "" : "stale"}`} />{note.source_active ? "SOURCE ACTIVE" : "STALE SOURCE"}</div><p>{note.claim}</p><div className="note-actions"><button onClick={() => { setNoteDraft({ claim: note.claim, quote: note.quote, evidenceId: note.evidence_id, replaceId: note.id }); setVerificationResult(null); }}>Edit</button><button onClick={() => void inspectEvidence(note.evidence_id)}>View source</button><button aria-label="Delete note" title="Delete note" onClick={() => void deleteNote(note.id)}><Trash2 size={14}/></button></div></div>)}</div> : <div className="muted-empty">No notes yet. Inspect a passage in Evidence and choose “Draft context note.”</div>}
          {noteDraft && <div className="note-editor"><div className="panel-heading"><h3>{noteDraft.replaceId ? "Edit research note" : "Propose context note"}</h3><button aria-label="Close editor" className="ghost-icon" onClick={() => setNoteDraft(null)}><X size={16}/></button></div><label>Original exact quote <span>(max 100 words)</span><textarea rows={4} value={noteDraft.quote} onChange={(e) => setNoteDraft({ ...noteDraft, quote: e.target.value })} placeholder="Paste or select exact source text…" /></label><label>Edited claim <span>(max 100 words)</span><textarea rows={4} value={noteDraft.claim} onChange={(e) => setNoteDraft({ ...noteDraft, claim: e.target.value })} placeholder="Write a faithful paraphrase…" /></label><div className="note-source-id">Evidence: {shortId(noteDraft.evidenceId)}</div><button className="primary-wide" disabled={savingNote || !noteDraft.quote.trim() || !noteDraft.claim.trim()} onClick={() => void saveNote()}>{savingNote ? <LoaderCircle className="spin" size={15}/> : <ShieldCheck size={15}/>} Verify and save</button><p className="small-muted">The backend verifies the quote and records accepted or rejected proposals in the audit log.</p></div>}
          {verificationResult && <div className="verification-banner"><CircleCheck size={16}/> Accepted at revision {verificationResult.revision}</div>}
          <div className="audit-section"><div className="section-label"><History size={13}/> RECENT CONTEXT EVENTS</div>{events.length ? events.slice(-6).reverse().map((event) => <div key={event.id} className="audit-entry"><span className={`audit-dot ${event.status === "rejected" ? "rejected" : ""}`} /><div><strong>{event.operation} · {event.status}</strong><small>rev {event.previous_revision} → {event.new_revision}</small><p>{event.reason}</p></div></div>) : <p className="small-muted">No context edits recorded.</p>}</div>
        </div>}
        {inspector === "verify" && <div className="check-panel"><div className="section-label">SIMILARITY LAB</div><h3>Compare context edits</h3><p className="panel-description">Compare paraphrases and contradictions using the selected checker. TEV1 is experimental, not guaranteed factual verification.</p><label>Original passage<textarea rows={5} value={comparison.original} onChange={(e) => { setComparison({ ...comparison, original: e.target.value }); setCompareResult(null); }} placeholder="Text from a source passage…" /></label><label>Edited passage<textarea rows={5} value={comparison.edited} onChange={(e) => { setComparison({ ...comparison, edited: e.target.value }); setCompareResult(null); }} placeholder="Proposed reformulation…" /></label><p className="small-muted">Checker: {verificationMode === "tev1" ? "TEV1 experimental decision model" : verificationMode === "extractive" ? "Exact textual support" : "MiniLM cosine similarity"}</p><button className="primary-wide" onClick={() => void compareTexts()} disabled={comparing || !comparison.original.trim() || !comparison.edited.trim()}>{comparing ? <LoaderCircle className="spin" size={15}/> : <ShieldCheck size={15} />} Run verification</button>{compareResult && <div className="similarity-result"><div className="result-header"><span>{compareResult.verification_method}</span><span className={compareResult.accepted ?? compareResult.above_threshold ? "pass-label" : "fail-label"}>{compareResult.accepted ?? compareResult.above_threshold ? "Passed" : "Not verified"}</span></div>{compareResult.cosine_similarity != null && <strong>{compareResult.cosine_similarity.toFixed(3)}</strong>}{compareResult.supported_probability != null && <strong>{(compareResult.supported_probability * 100).toFixed(1)}% supported</strong>}{compareResult.reason && <p>{compareResult.reason}</p>}{compareResult.threshold != null && <small>Cosine threshold: {compareResult.threshold.toFixed(2)}</small>}</div>}<div className="footnote-box"><CircleAlert size={15} /> Similarity does not establish factual entailment. Negations and missing caveats can score highly.</div></div>}
      </div>
      <div className="inspector-footer"><ShieldCheck size={14} /> Local-first · evidence-grounded</div>
    </section></div>}
  </div>;
}
