const test = require("node:test");
const assert = require("node:assert/strict");

const {
  toGraphModel,
  partitionPeople,
  layoutPeopleGraph,
  relationChoices,
} = require("../src/people_graph.js");

const graph = {
  people: [
    { id: "father", display_name: "爸爸", membership: { membership: "family" } },
    { id: "daughter", display_name: "小雨", membership: { membership: "family" } },
    { id: "friend", display_name: "小陈", membership: { membership: "friend" } },
    { id: "unknown", display_name: "路人", membership: { membership: "unknown" } },
  ],
  relationships: [
    { subject_entity_id: "father", predicate: "父亲", object_entity_id: "daughter", inverse_predicate: "女儿" },
    { subject_entity_id: "daughter", predicate: "女儿", object_entity_id: "father", inverse_predicate: "父亲" },
    { subject_entity_id: "daughter", predicate: "朋友", object_entity_id: "friend", inverse_predicate: "朋友" },
    { subject_entity_id: "friend", predicate: "朋友", object_entity_id: "daughter", inverse_predicate: "朋友" },
  ],
};

test("folds inverse records into one directed visible edge", () => {
  const model = toGraphModel(graph);
  assert.equal(model.edges.filter((edge) => edge.predicate === "父亲").length, 1);
  assert.deepEqual(model.edges.find((edge) => edge.predicate === "父亲"), {
    subjectId: "father", objectId: "daughter", predicate: "父亲", inversePredicate: "女儿",
  });
});

test("keeps only family and connected friends in graph", () => {
  const groups = partitionPeople(toGraphModel(graph));
  assert.deepEqual(groups.family.map((person) => person.id), ["father", "daughter"]);
  assert.deepEqual(groups.friends.map((person) => person.id), ["friend"]);
  assert.deepEqual(groups.pending.map((person) => person.id), ["unknown"]);
});

test("produces stable generational layout", () => {
  const model = toGraphModel(graph);
  assert.deepEqual(layoutPeopleGraph(model, 900, 600), layoutPeopleGraph(model, 900, 600));
  const positions = layoutPeopleGraph(model, 900, 600);
  assert.ok(positions.father.y < positions.daughter.y);
});

test("offers only valid stored relation pairs", () => {
  assert.deepEqual(relationChoices().find((choice) => choice.label === "父亲 → 女儿"), {
    label: "父亲 → 女儿", predicate: "父亲", inversePredicate: "女儿",
  });
});
