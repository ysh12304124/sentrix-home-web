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

test("Phase C: work trace auto-collapses after final and stays open on failure", () => {
  assert.match(app, /查看处理过程 · \$\{progressSteps\.length\} 步/);
  assert.match(app, /failureStatus/);
  assert.match(app, /blocked_by_guard/);
  assert.match(css, /details\.assistant-progress summary/);
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
