(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.SentrixPeopleGraph = api;
})(typeof globalThis !== "undefined" ? globalThis : window, function () {
  const relationPriority = new Map([
    ["父亲", 1], ["母亲", 1], ["丈夫", 2], ["妻子", 2],
    ["祖父", 3], ["祖母", 3], ["外祖父", 3], ["外祖母", 3],
    ["哥哥", 4], ["姐姐", 4], ["弟弟", 4], ["妹妹", 4],
    ["叔叔", 5], ["伯伯", 5], ["姑姑", 5], ["舅舅", 5], ["姨妈", 5], ["姨母", 5],
    ["朋友", 8], ["密友", 8], ["其他亲属", 9],
  ]);
  const parentRelations = new Set(["父亲", "母亲", "祖父", "祖母", "外祖父", "外祖母"]);
  const partnerRelations = new Set(["丈夫", "妻子"]);
  const friendRelations = new Set(["朋友", "密友"]);

  function relationChoices() {
    return [
      ["父亲", "儿子"], ["父亲", "女儿"], ["母亲", "儿子"], ["母亲", "女儿"],
      ["丈夫", "妻子"], ["妻子", "丈夫"], ["哥哥", "弟弟"], ["姐姐", "妹妹"],
      ["朋友", "朋友"], ["密友", "密友"], ["其他亲属", "其他亲属"],
    ].map(([predicate, inversePredicate]) => ({ label: `${predicate} → ${inversePredicate}`, predicate, inversePredicate }));
  }

  function normalisePeople(graph) {
    return (graph.people || []).map((person) => ({
      ...person,
      membershipValue: person.membership && person.membership.membership || "unknown",
    }));
  }

  function isBetterEdge(left, right) {
    const leftPriority = relationPriority.get(left.predicate) || 100;
    const rightPriority = relationPriority.get(right.predicate) || 100;
    if (leftPriority !== rightPriority) return leftPriority < rightPriority;
    return String(left.predicate).localeCompare(String(right.predicate), "zh-CN") < 0;
  }

  function toGraphModel(graph) {
    const people = normalisePeople(graph);
    const ids = new Set(people.map((person) => person.id));
    const byPair = new Map();
    for (const relationship of (graph.relationships || [])) {
      const subjectId = relationship.subject_entity_id;
      const objectId = relationship.object_entity_id;
      if (!ids.has(subjectId) || !ids.has(objectId) || subjectId === objectId) continue;
      const candidate = {
        subjectId, objectId, predicate: relationship.predicate, inversePredicate: relationship.inverse_predicate,
      };
      const pairKey = [subjectId, objectId].sort().join("\u0000");
      const existing = byPair.get(pairKey);
      if (!existing || isBetterEdge(candidate, existing)) byPair.set(pairKey, candidate);
    }
    return { people, edges: [...byPair.values()] };
  }

  function partitionPeople(model) {
    const family = model.people.filter((person) => person.membershipValue === "family");
    const familyIds = new Set(family.map((person) => person.id));
    const friendIds = new Set(
      model.edges.filter((edge) => friendRelations.has(edge.predicate) && (familyIds.has(edge.subjectId) || familyIds.has(edge.objectId)))
        .flatMap((edge) => [edge.subjectId, edge.objectId]),
    );
    familyIds.forEach((id) => friendIds.delete(id));
    const friends = model.people.filter((person) => person.membershipValue === "friend" && friendIds.has(person.id));
    const visibleIds = new Set([...family.map((person) => person.id), ...friends.map((person) => person.id)]);
    return { family, friends, pending: model.people.filter((person) => !visibleIds.has(person.id)) };
  }

  function layoutPeopleGraph(model, width, height) {
    const { family, friends } = partitionPeople(model);
    const familyIds = new Set(family.map((person) => person.id));
    const level = new Map(family.map((person) => [person.id, 1]));
    for (let pass = 0; pass < family.length; pass += 1) {
      for (const edge of model.edges) {
        if (!familyIds.has(edge.subjectId) || !familyIds.has(edge.objectId) || !parentRelations.has(edge.predicate)) continue;
        level.set(edge.objectId, Math.max(level.get(edge.objectId) || 1, (level.get(edge.subjectId) || 1) + 1));
      }
    }
    const groups = new Map();
    family.forEach((person) => {
      const row = level.get(person.id) || 1;
      if (!groups.has(row)) groups.set(row, []);
      groups.get(row).push(person);
    });
    const positions = {};
    const rows = [...groups.keys()].sort((a, b) => a - b);
    rows.forEach((row, index) => {
      const people = groups.get(row).sort((a, b) => a.display_name.localeCompare(b.display_name, "zh-CN"));
      people.forEach((person, column) => {
        positions[person.id] = {
          x: width * (column + 1) / (people.length + 1),
          y: height * (index + 1) / (rows.length + 1),
        };
      });
    });
    friends.sort((a, b) => a.display_name.localeCompare(b.display_name, "zh-CN")).forEach((person, index) => {
      const angle = (-Math.PI / 2) + (Math.PI * 2 * index / Math.max(friends.length, 1));
      positions[person.id] = { x: width / 2 + Math.cos(angle) * width * .38, y: height / 2 + Math.sin(angle) * height * .34 };
    });
    return positions;
  }

  return { toGraphModel, partitionPeople, layoutPeopleGraph, relationChoices, partnerRelations, friendRelations };
});
