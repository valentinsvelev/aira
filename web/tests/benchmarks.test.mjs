// Benchmark catalog, dataset upload parsing and the Experiments page.
import test from 'node:test';
import assert from 'node:assert/strict';
import { BENCHMARK_CATALOG, BENCHMARK_TRACKS } from '../lib/benchmark-catalog.ts';
import { parseBenchmarkFile } from '../lib/benchmark-upload.ts';
import { source } from './helpers.mjs';

const experimentsWorkspace = source('components/experiments-workspace.tsx');

test('catalog includes QASPER, DeepResearch Bench, BrowseComp-Plus and DRUID', () => {
  const ids = BENCHMARK_CATALOG.map(b => b.id);
  assert.deepEqual(ids, ['qasper', 'druid', 'browsecomp-plus', 'deepresearch-bench']);
  assert.deepEqual([...BENCHMARK_TRACKS], ['Scientific QA', 'Research agents']);
  assert.equal(new Set(ids).size, ids.length);
});

test('each benchmark has an explicit protocol, metrics, official links, and honest caveats', () => {
  for (const b of BENCHMARK_CATALOG) {
    assert.ok(b.name.trim() && b.summary.trim());
    assert.ok(b.protocol.length >= 50, `protocol missing for ${b.id}`);
    assert.ok(b.metrics.trim() && b.caveat.trim());
    assert.ok(b.year >= 2018 && b.year <= 2026);
    assert.ok(BENCHMARK_TRACKS.includes(b.track));
    for (const link of [b.officialUrl, b.datasetUrl]) {
      assert.match(link, /^https:\/\/[\w.-]+(?:\/|$)/);
    }
  }
});

test('catalog does not list commercial judging as a free built-in capability', () => {
  const deepresearch = BENCHMARK_CATALOG.find(b => b.id === 'deepresearch-bench');
  assert.match(deepresearch.caveat, /proprietary|not \$0/i);
});

test('custom evaluation parses labeled JSON/JSONL and optional context', () => {
  assert.deepEqual(parseBenchmarkFile('test.json','[{"question":"What?","answer":"Evidence","context":"A paper"}]'),
    [{question:'What?',answer:'Evidence',context:'A paper'}]);
  assert.deepEqual(parseBenchmarkFile('test.jsonl','{"query":"Why?","reference":"Because"}\n'),
    [{question:'Why?',answer:'Because',context:''}]);
});

test('quoted CSV, newlines and embedded commas are parsed properly', () => {
  assert.deepEqual(parseBenchmarkFile('cases.csv','question,answer,context\n"Which, model?","LoRA","A\nscientific, paper"\n'),
    [{question:'Which, model?', answer:'LoRA', context:'A\nscientific, paper'}]);
});

test('invalid rows or parquet do not silently produce fake evaluations', () => {
  assert.throws(() => parseBenchmarkFile('bad.parquet','binary'), /not supported/);
  assert.throws(() => parseBenchmarkFile('empty.json','[{"query":"What?"}]'), /No valid rows/);
});

test('experiment view remains a simple dataset selector with disclosure of protocol', () => {
  assert.match(experimentsWorkspace, /BENCHMARK_CATALOG\.map/);
  assert.match(experimentsWorkspace, /Your own dataset/);
  assert.match(experimentsWorkspace, /<details>/);
  assert.match(experimentsWorkspace, /Evaluation protocol and requirements/);
  assert.doesNotMatch(experimentsWorkspace, /benchmark-search|benchmark-filters|visibleBenchmarks/);
});

test('small-sample benchmark explicitly starts evaluation and discloses its protocol', () => {
  assert.match(experimentsWorkspace, /postJSON<RunResult>\("\/benchmarks\/run"/);
  assert.match(experimentsWorkspace, /\[1,3,5,10\]/);
  assert.match(experimentsWorkspace, /onClick=\{\(\) => void startRun\(\)\}/);
  assert.match(experimentsWorkspace, /Only selected rows leave your browser/);
  assert.match(experimentsWorkspace, /result\.protocol_note/);
});

test('benchmark presets use two columns with a full-width local dataset choice', () => {
  const css = source('app/globals.css');
  assert.match(css, /experiments-simple-options\{display:grid;grid-template-columns:repeat\(2,minmax\(0,1fr\)\)/);
  assert.match(css, /experiments-option-custom\{grid-column:1 \/ -1\}/);
  assert.match(experimentsWorkspace, /experiments-option-custom/);
});

test('four benchmark references, with explicit limited evaluation', () => {
  assert.deepEqual(BENCHMARK_CATALOG.map(item => item.id), ['qasper','druid','browsecomp-plus','deepresearch-bench']);
  const view = source('components/experiments-workspace.tsx');
  assert.match(view, /Start run/);
  assert.match(view, /postJSON<RunResult>/);
  assert.match(view, /QASPER, DRUID/);
});

test('benchmark UI requires an explicit start, with an overridable local save checkbox', () => {
  assert.match(experimentsWorkspace, /save_results: saveResults/);
  assert.match(experimentsWorkspace, /Save benchmark predictions locally/);
  assert.match(experimentsWorkspace, /useState\(true\)/);
  assert.match(experimentsWorkspace, /onClick=\{\(\) => void startRun\(\)\}/);
});

test('saved result IDs and storage errors are visible, not silently swallowed', () => {
  assert.match(experimentsWorkspace, /result\.run_id/);
  assert.match(experimentsWorkspace, /result\.save_error/);
});

test('benchmark random sampling and all-mode with no starting row', () => {
  const src = source('components/experiments-workspace.tsx');
  assert.doesNotMatch(src, /Benchmark starting row|Starting row/);
  assert.match(src, /All examples/);
  assert.match(src, /random_sample: true/);
  assert.match(src, /\/benchmarks\/jobs/);
});
