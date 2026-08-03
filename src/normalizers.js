const toneNames = ["mint", "peach", "blue", "lime", "lavender"];

function firstValue(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function formatDate(value) {
  if (!value) return "未标注时间";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10).replace(/-/g, ".");
  const parts = new Intl.DateTimeFormat("en-CA", { year: "numeric", month: "2-digit", day: "2-digit", timeZone: "Asia/Shanghai" }).formatToParts(date);
  const values = Object.fromEntries(parts.filter((part) => part.type !== "literal").map((part) => [part.type, part.value]));
  return `${values.year}.${values.month}.${values.day}`;
}

function participantNames(participants) {
  return (Array.isArray(participants) ? participants : [])
    .map((person) => firstValue(person.name, person.display_name, person.label))
    .filter(Boolean);
}

function normalizeEvent(event, index = 0) {
  const people = participantNames(event.participants || event.people);
  const count = firstValue(event.asset_count, event.observation_count, event.count, 0);
  return {
    id: String(firstValue(event.id, event.event_id, `event_${index + 1}`)),
    title: String(firstValue(event.title, event.name, event.activity, "未命名事件")),
    date: formatDate(firstValue(event.time_start, event.start_time, event.date)),
    place: [firstValue(event.place_text, event.place, event.location, "未标注地点"), people.length ? people.join("、") : "暂无人物"].join(" · "),
    type: String(firstValue(event.type_label, event.event_type, event.type, "家庭事件")),
    count: `${count} 项`,
    text: String(firstValue(event.summary, event.description, event.text, "暂无事件摘要。")),
    tone: toneNames[index % toneNames.length],
  };
}

function sourceEvents(payload) {
  return payload?.events || payload?.l2?.events || payload?.data?.events || [];
}

function normalizeDashboard(payload = {}) {
  const pipeline = payload.pipeline || payload.stats || {};
  const events = sourceEvents(payload).map(normalizeEvent);
  return {
    stats: {
      assets: Number(firstValue(pipeline.assets, pipeline.asset_count, 0)),
      observations: Number(firstValue(pipeline.observations, pipeline.observation_count, 0)),
      events: Number(firstValue(pipeline.events, pipeline.event_count, events.length)),
      facts: Number(firstValue(pipeline.facts, pipeline.fact_count, payload.l3?.facts?.length, 0)),
    },
    events,
  };
}

function normalizePersons(persons = []) {
  return (Array.isArray(persons) ? persons : []).map((person, index) => {
    const confirmed = Boolean(person.confirmed || person.status === "confirmed");
    const name = confirmed
      ? firstValue(person.display_name, person.name, person.label, `家庭成员 ${index + 1}`)
      : `待命名成员 ${index + 1}`;
    return {
      id: String(firstValue(person.id, person.person_id, `person_${index + 1}`)),
      name: String(name),
      role: confirmed ? "家庭成员" : "待确认人物",
      count: String(firstValue(person.sample_count, person.asset_count, person.count, 0)),
      tone: toneNames[index % toneNames.length],
      initial: String(name).slice(0, 1),
      confirmed,
    };
  });
}

function observationSummary(observation) {
  const caption = observation.caption;
  if (typeof caption === "string") return caption;
  return firstValue(caption?.caption, observation.text, observation.transcript, observation.summary, "暂无观察摘要。");
}

function buildAgentContext({ dashboard = {}, persons = [], entities = [], observations = [], relationships = [], vectors = [] } = {}) {
  const events = sourceEvents(dashboard);
  const facts = dashboard.l3?.facts || dashboard.facts || [];
  const lines = [
    "你是 Sentrix 家庭记忆 Agent。只能使用下面的证据回答问题。证据不足时明确说明，不得编造。",
    "[EVENTS]",
    ...events.map((event) => JSON.stringify({ id: event.id, title: event.title, time: firstValue(event.time_start, event.date), place: firstValue(event.place_text, event.place), participants: participantNames(event.participants), summary: firstValue(event.summary, event.description) })),
    "[ENTITIES]",
    ...persons.map((person) => JSON.stringify({ id: person.id, name: firstValue(person.display_name, person.name), confirmed: Boolean(person.confirmed) })),
    ...entities.map((entity) => JSON.stringify(entity)),
    "[OBSERVATIONS]",
    ...observations.map((observation) => JSON.stringify({ observation_id: observation.id, asset_id: observation.asset?.id || observation.asset_id, event_id: observation.event?.id || observation.event_id, summary: observationSummary(observation) })),
    "[FACTS]",
    ...facts.map((fact) => JSON.stringify(fact)),
    "[RELATIONSHIPS]",
    ...relationships.map((relationship) => JSON.stringify(relationship)),
    "[VECTOR_HITS]",
    ...vectors.map((vector) => JSON.stringify(vector)),
    "回答时返回答案、置信度、证据 ID 和是否证据不足。不要把推断伪装成事实。",
  ];
  return lines.join("\n");
}

module.exports = { normalizeDashboard, normalizePersons, buildAgentContext };
