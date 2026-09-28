const fs = require("fs");
const path = require("path");
const test = require("node:test");
const assert = require("node:assert/strict");

const root = path.resolve(__dirname, "..");
const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
const apiSource = fs.readFileSync(path.join(root, "src", "api.js"), "utf8");

test("people UI separates role confirmation from naming", () => {
  assert.match(appSource, /data-action="confirm-suggested-role"/);
  assert.match(appSource, /data-action="save-person-name"/);
  assert.match(appSource, /无法判断/);
  assert.match(appSource, /朋友/);
  assert.match(appSource, /同事/);
  assert.match(appSource, /照护者/);
});

test("people UI merges insight into unified person cards", () => {
  assert.match(appSource, /cluster-samples-inline/);
  assert.match(appSource, /data-action="open-person-insight"/);
  assert.match(appSource, /suggestion-badge">重要</);
  assert.match(appSource, /系统建议/);
});

test("portrait interaction controls exist", () => {
  assert.match(appSource, /data-action="portrait-feedback"/);
  assert.match(appSource, /data-action="edit-portrait"/);
  assert.match(appSource, /data-action="lock-portrait"/);
  assert.match(appSource, /历史版本/);
});

test("API client exposes person insight methods", () => {
  assert.match(apiSource, /personInsights:/);
  assert.match(apiSource, /startPersonInsightRun:/);
  assert.match(apiSource, /personInsightRun:/);
  assert.match(apiSource, /decidePersonRole:/);
  assert.match(apiSource, /savePersonName:/);
  assert.match(apiSource, /decideRelationshipHypothesis:/);
  assert.match(apiSource, /personPortrait:/);
  assert.match(apiSource, /sendPortraitFeedback:/);
  assert.match(apiSource, /updatePortrait:/);
});

test("legacy batch confirm and identity photo entry remain", () => {
  assert.match(appSource, /data-action="batch-confirm"/);
  assert.match(appSource, /data-action="open-person-profile"/);
});
