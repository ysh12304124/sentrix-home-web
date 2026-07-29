const test = require("node:test");
const assert = require("node:assert/strict");

const {
  normalizeDashboard,
  normalizePersons,
  buildAgentContext,
} = require("../src/normalizers");

test("normalizes Sentrix dashboard events into the portal event shape", () => {
  const result = normalizeDashboard({
    pipeline: { assets: 2, observations: 4, events: 1, facts: 0 },
    l2: {
      events: [{
        id: "evt_1",
        title: "周末公园",
        type_label: "日常",
        time_start: "2026-07-01T10:00:00",
        place_text: "城市公园",
        participants: [{ id: "person_1", name: "小明" }],
        asset_count: 3,
        summary: "一家人在公园散步。",
      }],
    },
  });

  assert.equal(result.stats.events, 1);
  assert.deepEqual(result.events[0], {
    id: "evt_1",
    title: "周末公园",
    date: "2026.07.01",
    place: "城市公园 · 小明",
    type: "日常",
    count: "3 项",
    text: "一家人在公园散步。",
    tone: "mint",
  });
});

test("normalizes confirmed and pending persons without losing source ids", () => {
  const result = normalizePersons([
    { id: "p1", display_name: "妈妈", name: "妈妈", confirmed: true, sample_count: 4 },
    { id: "p2", display_name: "待命名成员 2", name: null, confirmed: false, sample_count: 1 },
  ]);

  assert.equal(result[0].id, "p1");
  assert.equal(result[0].confirmed, true);
  assert.equal(result[1].id, "p2");
  assert.equal(result[1].confirmed, false);
  assert.equal(result[1].count, "1");
});

test("builds an agent context with events, people, and explicit evidence rules", () => {
  const context = buildAgentContext({
    dashboard: { l2: { events: [{ id: "evt_1", title: "生日", summary: "切蛋糕", type_label: "生日", time_start: "2026-06-01T18:00:00", place_text: "家里", participants: [{ name: "妈妈" }], asset_count: 5 }] }, l3: { facts: [] } },
    persons: [{ id: "p1", display_name: "妈妈", confirmed: true }],
    observations: [{ asset: { id: "asset_1" }, caption: { caption: "桌上有蛋糕" }, event: { id: "evt_1", title: "生日" } }],
    cognee: ["家庭成员在家中庆祝生日"],
  });

  assert.match(context, /evt_1/);
  assert.match(context, /妈妈/);
  assert.match(context, /asset_1/);
  assert.match(context, /证据不足时明确说明/);
});
