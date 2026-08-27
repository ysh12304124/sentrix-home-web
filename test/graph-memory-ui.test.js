const fs = require("fs");
const path = require("path");
const test = require("node:test");
const assert = require("node:assert/strict");

const root = path.resolve(__dirname, "..");
const app = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
const api = fs.readFileSync(path.join(root, "src", "api.js"), "utf8");

test("people page exposes the video memory graph task link", () => {
  assert.match(app, /任务动态联系/);
  assert.match(app, /dataset\.action = "graph-memory"/);
  assert.match(app, /graph-memory-stats/);
  assert.match(app, /graphMemoryQuery/);
  assert.match(app, /EPISODE.*SESSION.*EVENT.*ENTITY/s);
});

test("browser API exposes graph memory detail endpoints", () => {
  assert.match(api, /graphMemoryStats/);
  assert.match(api, /graphMemoryQuery/);
  assert.match(api, /graphMemoryNode/);
  assert.match(api, /graphMemorySubgraph/);
});
