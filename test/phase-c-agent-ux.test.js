const fs = require("node:fs");
const test = require("node:test");
const assert = require("node:assert/strict");

const app = fs.readFileSync("src/app.js", "utf8");
const api = fs.readFileSync("src/api.js", "utf8");
const css = fs.readFileSync("src/styles.css", "utf8");

test("Phase C: ResultSet card never renders unknown total or zero-remaining CTA", () => {
  assert.match(app, /找到一批相关结果/);
  assert.match(app, /const totalKnown = ts\.result_total != null/);
  assert.doesNotMatch(app, /result_total != null \? ts\.result_total : "\?"/);
  assert.match(app, /hasMore = Boolean\(ts\.has_more\) && remaining > 0/);
});

test("Phase C: agent-style work trace auto-collapses after final and stays open on failure", () => {
  assert.match(app, /思考过程 · \$\{traceSteps\.length\} 步/);
  assert.match(app, /buildThinkingSteps/);
  assert.match(app, /agentStepHtml/);
  assert.match(app, /failureStatus/);
  assert.match(app, /blocked_by_guard/);
  assert.match(css, /details\.agent-trace-box summary/);
  assert.match(css, /\.agent-step\.running/);
});

test("Phase C: original evidence fold defaults open with item count", () => {
  assert.match(app, /<summary>原始证据\$\{evidenceCount/);
  assert.match(app, /hasGap \|\| evidenceCount > 0 \? " open" : ""/);
});

test("Phase C/UX: tool-loop evidence samples feed the original-evidence fold", () => {
  assert.match(app, /function toolLoopEvidence/);
  assert.match(app, /\(tr\.observation \|\| \{\}\)\.samples/);
  assert.match(app, /hasToolEvidence \|\| hasResultSet/);
});

test("Phase C: SSE events and polling fallback are wired", () => {
  assert.match(api, /assistantTurnEventsUrl/);
  assert.match(app, /new EventSource\(/);
  assert.match(app, /pollTurnEvents/);
  assert.match(fs.readFileSync("server.js", "utf8"), /text\/event-stream/);
});

test("Phase C: selected photo flows into next turn and original delivery", () => {
  assert.match(api, /selected_asset_handle/);
  assert.match(api, /selected_result_set_id/);
  assert.match(app, /select-result-photo/);
  assert.match(app, /open-selected-original/);
  assert.match(app, /state\.selectedAsset/);
});

test("Phase C/C8: inspected thumbnail badge and observation notes render", () => {
  assert.match(app, /已复核/);
  assert.match(app, /result-set-check inspected/);
  assert.match(app, /result-set-inspect-notes/);
  assert.match(app, /tr\.tool === "inspect_photo" && tr\.inspect_handle/);
  assert.match(css, /\.result-set-thumb\.inspected img/);
});

test("Phase C/C9: guard debug detail is admin-only and layered", () => {
  assert.match(app, /function guardDebug/);
  assert.match(app, /Guard 校验明细/);
  assert.match(app, /l1_codes/);
  assert.match(app, /L2 评审/);
  assert.match(app, /恢复步数/);
  assert.match(app, /adminDebug\(\)/);
});
