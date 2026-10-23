"use client";

import { useRef } from "react";
import { BookOpen, FileText, LoaderCircle, RefreshCw, Upload } from "lucide-react";
import type { DocumentInfo } from "@/lib/types";

export function LibraryWorkspace({ documents, uploading, syncing, upload, sync }: {
  documents: DocumentInfo[];
  uploading: boolean;
  syncing: boolean;
  upload: (file: File) => Promise<void>;
  sync: () => Promise<void>;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const active = documents.filter(doc => doc.active);
  return <section className="feature-page">
    <header className="feature-head">
      <div><div className="eyebrow">YOUR KNOWLEDGE BASE</div><h1>Paper library</h1><p>Manage the local sources used for retrieval, citation checking, and research conversations.</p></div>
      <div className="feature-actions">
        <input ref={fileInput} className="hidden-input" type="file" accept=".pdf,.txt,.md" onChange={e => {
          const file = e.target.files?.[0]; if (file) void upload(file);
          e.target.value = "";
        }}/>
        <button className="feature-action secondary" disabled={syncing} onClick={() => void sync()}>
          {syncing ? <LoaderCircle className="spin" size={16}/> : <RefreshCw size={16}/>} {syncing ? "Indexing…" : "Sync folder"}
        </button>
        <button className="feature-action primary" disabled={uploading} onClick={() => fileInput.current?.click()}>
          {uploading ? <LoaderCircle className="spin" size={16}/> : <Upload size={16}/>} {uploading ? "Uploading…" : "Upload paper"}
        </button>
      </div>
    </header>
    <div className="feature-stat"><BookOpen size={18}/><strong>{active.length}</strong> active documents <span>·</span> {active.reduce((n,d)=>n+d.chunks,0).toLocaleString()} indexed passages</div>
    {active.length ? <div className="library-page-list">{active.map(doc => <article className="library-page-row" key={doc.path}>
      <span className="library-page-icon"><FileText size={20}/></span>
      <div><strong title={doc.path}>{doc.path.split("/").at(-1)}</strong><p>{doc.path} · Version {doc.version ?? "—"} · {doc.chunks} passages</p></div>
    </article>)}</div> : <div className="feature-empty"><BookOpen size={25}/><h2>Your library is empty</h2><p>Upload a paper or copy PDFs into the local papers folder and select Sync folder.</p></div>}
  </section>;
}
