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

test("people UI has three tiers with recommended relationships", () => {
  assert.match(appSource, /系统认为最重要的人/);
  assert.match(appSource, /其他常见人物/);
  assert.match(appSource, /一次性或低证据人物/);
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
