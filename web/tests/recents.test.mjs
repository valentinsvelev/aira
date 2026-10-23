// Saved chats and Paper Explorer history in the Recents dialog.
import test from 'node:test';
import assert from 'node:assert/strict';
import { PAPER_HISTORY_KEY, parsePaperHistory, readPaperHistory, writePaperHistory, upsertPaperHistory, paperSessionTitle } from '../lib/paper-history.ts';
import { source } from './helpers.mjs';

const researchWorkspace = source('components/research-workspace.tsx');
const sample = (id, date = '2026-10-16T12:00:00.000Z') => ({id,title:`Topic ${id}`,created_at:date,updated_at:date,turns:[{
  question:'What changed?',result:{question:'What changed?',answer:'Findings [1].',scope:'abstracts_only',search_query:'research',provider_errors:{},papers:[{
    number:1,evidence_id:'a',title:'Relevant paper',authors:['A'],year:2026,url:'https://example.org',abstract:'Evidence text',provider:'arxiv',cited:true,
  }]},
}]});

test('Paper Explorer session history survives serialization with cited papers intact', () => {
  const history = [sample('one')];
  const parsed = parsePaperHistory(JSON.stringify(history));
  assert.deepEqual(parsed, history);
  assert.equal(parsed[0].turns[0].result.papers[0].abstract, 'Evidence text');
  assert.deepEqual(parsePaperHistory('bad json'), []);
  assert.deepEqual(parsePaperHistory(JSON.stringify([{id:'invalid',turns:[]}])), []);
});

test('History adds/reorders sessions, derives readable title, and persists in localStorage', () => {
  const orig = globalThis.window;
  const cache = new Map();
  globalThis.window = {localStorage:{getItem:k=>cache.get(k)??null,setItem:(k,v)=>cache.set(k,v)}};
  try {
    const older = sample('old','2026-10-15T12:00:00.000Z');
    const newer = sample('new','2026-10-16T12:00:00.000Z');
    const merged = upsertPaperHistory([older],newer);
    assert.deepEqual(merged.map(item=>item.id),['new','old']);
    assert.equal(upsertPaperHistory(merged,{...older,title:'Renamed'}).length,2);
    writePaperHistory(merged);
    assert.ok(cache.has(PAPER_HISTORY_KEY));
    assert.deepEqual(readPaperHistory(),merged);
    assert.equal(paperSessionTitle('  One  two  three  '),'One two three');
  } finally { globalThis.window=orig; }
});

test('research and paper sidebars have parallel new-chat and recents entries', () => {
  const src=source('components/research-workspace.tsx');
  assert.match(src, /id="sidebar-conversations"/);
  assert.match(src, /id="sidebar-paper-conversations"/);
  assert.match(src, /setRecentsDialog\("research"\)/);
  assert.match(src, /setRecentsDialog\("paper"\)/);
  assert.match(src, /<RecentChatsDialog/);
  assert.match(src, /onMore=\{\(\) => void loadMoreChats\(\)\}/);
  assert.doesNotMatch(src,/chat-history-items/); // no more always-expanded list
});

test('dialog provides rename/delete/search, Escape handling and focus restoration', () => {
  const src=source('components/recent-chats-dialog.tsx');
  assert.match(src,/role="dialog" aria-modal="true"/);
  assert.match(src,/Search recent chats/);
  assert.match(src,/event.key === "Escape"/);
  assert.match(src,/previouslyFocused\?\.focus\(\)/);
  assert.match(src,/onRename\(item.id\)/);
  assert.match(src,/onDelete\(item.id\)/);
  assert.match(src,/onOpen\(item.id\)/);
});

test('saved conversation highlighting is scoped to Research Chat, now in the Recents dialog', () => {
  const recentDialog = source('components/recent-chats-dialog.tsx');
  assert.match(researchWorkspace, /activeId=\{recentsDialog === "research" \? \(view === "research" \? sessionId : ""\)/);
  assert.match(recentDialog, /recents-item \$\{activeId === item.id \? "active" : ""\}/);
  assert.match(researchWorkspace, /nav-entry \$\{view === "research" \? "active" : ""\}/);
});
