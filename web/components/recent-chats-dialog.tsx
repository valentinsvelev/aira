"use client";

import { useEffect, useId, useRef, useState } from "react";
import { BookOpen, Clock3, MessageSquareText, Pencil, Search, Trash2, X } from "lucide-react";

export type RecentChat = { id: string; title: string; updated_at: string };

export function RecentChatsDialog({kind, items, activeId, onClose, onOpen, onRename, onDelete, onMore, hasMore=false, loading=false, disabled=false}: {
  kind: "research" | "paper"; items: RecentChat[]; activeId: string; onClose: () => void;
  onOpen: (id: string) => void; onRename: (id: string) => void; onDelete: (id: string) => void;
  onMore?: () => void; hasMore?: boolean; loading?: boolean; disabled?: boolean;
}) {
  const titleId = useId();
  const dialog = useRef<HTMLElement>(null);
  const [filter, setFilter] = useState("");
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialog.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); closeRef.current(); }
      if (event.key !== "Tab" || !dialog.current) return;
      const elements = Array.from(dialog.current.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )).filter(element => element.getClientRects().length > 0);
      if (!elements.length) return;
      const first = elements[0]; const last = elements[elements.length - 1];
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) {
        event.preventDefault(); last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || document.activeElement === dialog.current)) {
        event.preventDefault(); first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => { document.removeEventListener("keydown", onKeyDown); previouslyFocused?.focus(); };
  }, []);
  const visible = items.filter(item => item.title.toLocaleLowerCase().includes(filter.toLocaleLowerCase().trim()));
  const title = kind === "research" ? "Recent research chats" : "Recent paper explorations";
  return <div className="recents-overlay" onMouseDown={event => {if(event.target === event.currentTarget) onClose();}}>
    <section className="recents-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} tabIndex={-1} ref={dialog}>
      <header className="recents-header"><div><span className="eyebrow">{kind === "research" ? "RESEARCH CHAT" : "PAPER EXPLORER"}</span>
        <h2 id={titleId}>{title}</h2><p>{kind === "research" ? "Your saved conversations" : "Saved on this device in your browser"}</p></div>
        <button type="button" className="recents-close" onClick={onClose} aria-label="Close recent chats"><X size={19}/></button>
      </header>
      <div className="recents-search"><Search size={16}/><input aria-label="Search recent chats" autoComplete="off" placeholder="Search conversations…" value={filter} onChange={event=>setFilter(event.target.value)}/></div>
      <div className="recents-list">
        {visible.length ? visible.map(item => <div key={item.id} className={`recents-item ${activeId === item.id ? "active" : ""}`}>
          <button type="button" className="recents-open" disabled={disabled} onClick={()=>onOpen(item.id)} title={item.title}>
            {kind === "research" ? <MessageSquareText size={19}/> : <BookOpen size={19}/>}
            <span className="recents-item-text"><strong>{item.title}</strong><small><Clock3 size={12}/> {new Date(item.updated_at).toLocaleString()}</small></span>
          </button>
          <div className="recents-actions"><button type="button" disabled={disabled} onClick={()=>onRename(item.id)} title="Rename" aria-label={`Rename ${item.title}`}><Pencil size={16}/></button>
          <button type="button" disabled={disabled} onClick={()=>onDelete(item.id)} title="Delete" aria-label={`Delete ${item.title}`}><Trash2 size={16}/></button></div>
        </div>) : <p className="recents-empty">{items.length ? "No matching conversations." : "No previous conversations yet. Start a new chat to create one."}</p>}
        {hasMore && !filter && <button type="button" className="recents-more" disabled={loading} onClick={onMore}>{loading ? "Loading…" : "Load more conversations"}</button>}
      </div>
    </section>
  </div>;
}
