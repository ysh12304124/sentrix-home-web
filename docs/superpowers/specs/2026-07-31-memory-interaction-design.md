# Memory Interaction and P0 Correctness Design

## Goal

Make person confirmation reliable from the existing web UI, propagate the
confirmed name through person-centred semantic memory, and turn Agent recall
into a conversational interaction that distinguishes questions from feedback
and renders image evidence. At the same time, close the remaining P0 event
segmentation correctness gaps without introducing a second storage or Agent
runtime.

## Architecture

The existing FastAPI `backend.app` and `MemoryStore` remain authoritative.
`face-cluster confirm` is the canonical identity write path. The legacy
`persons` endpoints remain compatibility adapters only. Confirmation is an
atomic projection update across entity, cluster, mentions, events, semantic
claims, and profile data. Anonymous model descriptions remain observation
evidence and never become identity bridges.

The Agent receives a conversation turn containing an optional conversation ID,
message, and optional feedback target. A deterministic intent router identifies
query, feedback, or clarification. Query responses keep the current evidence
trace and add structured image results (`asset_id`, file name, caption and
media URL). Feedback is persisted through the existing query-gap and memory
feedback tables and can target a fact or semantic claim. The old `/api/search`
route calls the same turn implementation for backward compatibility.

The browser keeps one same-origin API wrapper and renders image evidence from
backend asset URLs. Conversation history is local UI state keyed by the server
conversation ID; no external session service is added.

## Scope

- Fix entity/person confirmation ID mismatch and return a unified entity detail.
- Rebuild person facts after naming so subject text uses the confirmed name,
  with evidence and revision history preserved.
- Add assistant turn/query-feedback routing, conversation context, structured
  image results, and image rendering.
- Add P0 event regression coverage for confirmed-person resegmentation,
  anonymous-description isolation, vector dimension mismatch, and indexes.
- Add an observable same-time/same-place event fixture for calibration; do not
  alter production thresholds without benchmark evidence.

## Non-goals

- No external LLM, vector database, or session service.
- No automatic family relationship inference from a name alone.
- No production rebuild until the controlled benchmark gate passes.
- No changes to FMA service port 5173 or shared Ollama 11434.

## Acceptance Criteria

1. Naming any pending entity from the current people page returns HTTP 200 and
   changes the canonical entity name; the old legacy route no longer returns a
   false 404 for a native entity.
2. A confirmed person's derived clothing/activity claims contain the confirmed
   name through their person join, link an event and observation, and do not
   leave a competing anonymous person fact active.
3. `POST /api/assistant/turn` returns `intent`, `conversation_id`, answer,
   evidence, image results, and retrieval trace. Feedback turns persist a
   resolution and do not invoke normal visual recall.
4. An image query renders original-image thumbnails with an asset endpoint and
   keeps the answer text concise rather than listing file names only.
5. P0 event regressions pass: confirmed-person bridges can re-evaluate affected
   events, model `people` descriptions cannot bridge events, incompatible
   vector dimensions are unavailable, and the event lookup indexes exist.
6. Full Python/Node suites, syntax/compile checks, and controlled API checks
   pass on 153.

