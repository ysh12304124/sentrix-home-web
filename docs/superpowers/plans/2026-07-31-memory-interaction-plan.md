# Memory Interaction and P0 Correctness Implementation Plan

## Goal

Fix native-person naming, propagate names into semantic memory, add
query/feedback conversation turns with image evidence, and finish the P0 event
correctness checks.

## Architecture

Keep FastAPI + SQLite + browser JavaScript as the authoritative stack. The
canonical confirmation route is `/api/face-clusters/{cluster_id}/confirm`; a
legacy entity adapter resolves `/api/persons/{person_id}/confirm` to the
native cluster when possible. `MemoryAgent` owns deterministic intent and
evidence routing; `backend.app` exposes the turn endpoint. The UI uses the
same-origin proxy and renders backend asset URLs.

## Steps

1. **Synchronize and restore the clean 153 fixture**
   - Copy the current `psh` backend, frontend, tests, and project memory to
     `/Users/rm001/.codex/tmp/sentrix-edit`.
   - Restore `/tmp/sentrix.db.before-person-appearance-20260730-201048` on 153
     because a diagnostic probe accidentally confirmed one pending cluster.
   - Verify the restored cluster/entity status before testing.

2. **Write failing naming tests**
   - Add an API-level regression that resolves a native entity returned by
     `/api/people` and confirms it without a legacy `persons` row.
   - Add a store regression asserting confirmation rebuilds named activity and
     clothing claims and preserves observation/event evidence.
   - Run the focused tests and record the expected 404/anonymous failure.

3. **Implement naming compatibility and propagation**
   - Add `MemoryStore.confirm_person_entity` to resolve an entity to its active
     face cluster and delegate to `confirm_face_cluster`.
   - Change the legacy FastAPI confirmation route to use that resolver before
     falling back to the old table.
   - Ensure confirmation refreshes event participants, resegments only through
     confirmed entities, rebuilds derived claims, and refreshes event summaries
     after all joins are committed.
   - Add an explicit named subject projection for facts/claims where the API
     presents text, while keeping normalized `person_id` authoritative.

4. **Verify naming tests and frontend contract**
   - Run focused Python tests and `node --test test/*.test.js`.
   - Update the UI action to prefer the cluster endpoint and render the returned
     entity/profile/claims; keep a compatibility fallback only for legacy rows.
   - Add a Node regression covering the route string and native entity ID.

5. **Write failing Agent turn tests**
   - Add tests for `query`, `feedback`, and `clarification` intent detection.
   - Add a test that an image query returns structured image results with asset
     IDs and media URLs instead of only filenames.
   - Add a conversation test proving a second turn receives prior turn context.
   - Add a feedback test proving a query gap/fact correction is persisted and
     normal query retrieval is not called.

6. **Implement Agent conversation and image evidence**
   - Add a small deterministic intent router in `backend/agent.py` with model
     fallback only for ambiguous natural-language turns.
   - Add `answer_turn(message, conversation_id, feedback)` and bounded in-memory
     conversation context; use existing SQLite feedback/query-gap persistence.
   - Add `POST /api/assistant/turn`; make `/api/search` delegate to it with a
     query-only turn.
   - Add `image_results` to result payloads and include asset URL metadata for
     image observations.
   - Add API wrapper methods and a conversation panel/image result rendering in
     `src/api.js` and `src/app.js`.

7. **Implement and test remaining P0 event guards**
   - Add vector dimension validation in `_event_visual_similarity`.
   - Add event-observation and visual-vector lookup indexes in the additive
     schema migration.
   - Ensure event scoring only uses confirmed entity IDs and never anonymous
     `people` descriptions.
   - Add tests for confirmed-person resegmentation and anonymous-description
     non-bridging, plus a same-time/same-place observable event fixture for the
     benchmark evaluator.

8. **Run complete verification on the local copy**
   - `.venv/bin/python -m unittest discover -s backend/tests -v`
   - `node --test test/*.test.js`
   - `node --check src/app.js && node --check src/api.js`
   - `.venv/bin/python -m compileall -q backend scripts`
   - `git diff --check`

9. **Transfer, validate, and commit on 153**
   - Preserve unrelated files and never copy credentials, logs, databases, or
     runtime state.
   - Transfer verified source/tests/docs to `/home/asus/Github/Sentrix-Home-Web`.
   - Run the full suite and focused API checks on 153 with Sentrix Ollama at
     11435; leave FMA 5173 and shared Ollama 11434 untouched.
   - Update `docs/PROJECT_MEMORY.md` with measured results and remaining P0
     calibration status.
   - Commit the complete change on 153 `psh` with a Conventional Commit.

