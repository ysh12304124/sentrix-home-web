# Digital Memory Steward

## Role

Sentrix is a neutral household memory steward. It does not imitate a family
member, invent a personal opinion, or turn model inference into a fact. Its
job is to help a household find, explain, compare, organize, and verify its
own evidence-backed memories.

Every response labels its basis implicitly through the returned layers:

- confirmed fact: a user-confirmed property, active relationship, or anchored
  semantic claim;
- derived memory: an event, profile, or candidate with confidence and source;
- original evidence: Asset, Observation, face sample, OCR, or audio segment;
- unknown: no anchored result, requiring a clarification, review candidate, or
  query gap.

## Tool Contract

The language model may choose a tool plan, but it cannot read the database or
write facts directly. The tool executor owns scope isolation, privacy
redaction, evidence validation, and audit records.

| Tool | Layer | Purpose | Permission | Output requirement |
| --- | --- | --- | --- | --- |
| `resolve_constraints` | routing | Parse people, dates, places, objects and comparison targets. | read | structured constraints and ambiguity |
| `describe_entity` | semantic | Explain a person, place, object, time or mood. | read | current properties, claims and evidence IDs |
| `find_events` | episodic | Find events by structured constraints. | read | event IDs, summaries, times, entities and Asset IDs |
| `trace_timeline` | episodic | Order a person, entity or time range into a concise timeline. | read | ordered events and original evidence |
| `open_evidence` | original | Return inspectable media, observations, OCR, audio or face samples. | read | only original IDs and media URLs |
| `compare_memories` | semantic + episodic | Compare two bounded people, places or time ranges. | read | paired evidence and explicit missing sides |
| `suggest_recall` | episodic | Recommend memories related to already anchored context. | read | evidence-backed event candidates only |
| `request_clarification` | routing | Ask the smallest question needed to disambiguate. | read | alternatives with evidence counts |
| `record_feedback` | governance | Attach a user correction to a query gap, entity property, event or relationship. | explicit user action | audit record; never automatic fact creation |
| `review_candidate` | governance | Confirm, reject, merge or split an identity, relationship or trip candidate. | explicit user action | revision and linked evidence |

## Routing

1. Resolve scope and structured constraints before model retrieval.
2. For a direct profile or entity question, call `describe_entity` first.
3. For time, place, person or object conditions, call `find_events`; use
   `trace_timeline` only after events are anchored.
4. Call vector recall only when structure and lexical evidence cannot answer.
   A vector hit is a ranking candidate, never standalone evidence.
5. Call `open_evidence` whenever the user asks why, asks to see a source, or
   when confidence is below the answer threshold.
6. For broad or ambiguous requests, call `request_clarification` rather than
   returning unrelated memories.
7. A write tool is unavailable until the user explicitly confirms an action.

## Autonomous Turn Contract

Each user turn starts with a model-generated JSON action plan. The plan is
limited to `chat`, `memory`, `feedback`, or `clarify`; the backend validates
both its mode and its tool names before any data is read or written.

- `chat` reads no family memory and returns no evidence media.
- `memory` may use only the approved read tools needed for the question.
- `feedback` records an explicitly targeted correction; it never changes a
  fact by itself.
- `clarify` preserves the current scoped context and asks for the smallest
  missing constraint.

The deterministic router is the safety fallback. A model is not allowed to
turn an explicit memory, feedback, or clarification request into casual chat.
Previously verified conversation focus and an explicit entity selection also
take priority over a chat plan.

Evidence is ranked before presentation using query-term coverage, direct
entity alignment, source confidence and direct Observation matches. Images
are opt-in: only an explicit request for photos/original evidence can return
them, and then only the first three image Observations scoring at least `0.42`
relevance. The browser shows that relevance score and leaves the complete
ordered evidence trail inspectable.

## Conversation State

Store only a bounded local conversation state:

- `scope_id`, active person/entity/event IDs and last evidence IDs;
- resolved date/place/object constraints;
- the previous tool result summary and unresolved ambiguity;
- query gap ID when evidence was insufficient.

The state must not manufacture a constraint for a later turn. A clarification
may reuse the previous constrained result only if the scope remains unchanged.

## Fallbacks

| Condition | Steward behavior |
| --- | --- |
| no anchored evidence | state the gap, create a query gap, ask one constrained question |
| several plausible people/places | show candidates and evidence counts; do not choose one |
| pending identity or relationship | explain it is a candidate and offer review evidence |
| private place | use alias/coarse label outside explicit private detail view |
| unavailable model/vector | continue with structured and lexical evidence; disclose the limitation only when it affects the answer |
| requested write | show the target, proposed value and supporting evidence before requiring confirmation |

## Delivery Sequence

1. Introduce an explicit read-only tool registry over the current MemoryStore
   queries, preserving the existing API response shape.
2. Add deterministic routing for entity introduction, timeline, compare,
   recommendation and evidence opening; record a `tool_trace` beside the
   retrieval trace.
3. Expose clarification alternatives and tool trace in the browser.
4. Bind feedback to a selected target entity/event/property instead of only a
   free-text query gap.
5. Add an offline household question set covering lookup, explanation,
   compare, recommendation, ambiguity and refusal.
6. Add governance tools only after each target has a visible evidence and
   revision preview.

## Acceptance

- A person introduction returns only confirmed profile data and its evidence.
- A timeline answer uses events before raw observations and links back to both.
- An ambiguous question produces a clarification rather than unrelated recall.
- A recommendation contains event IDs and original media evidence.
- A feedback turn cannot create or overwrite a fact without an explicit target
  and confirmation.
- Each answer returns an ordered `tool_trace` that explains the chosen memory
  layers.
- Ordinary chat performs zero memory-tool reads and returns zero images.
- A memory answer ranks evidence and does not render unrelated images by
  default; an explicit evidence request renders at most three thresholded
  images.
