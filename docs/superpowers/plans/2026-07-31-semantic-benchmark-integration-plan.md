# Semantic Memory and Benchmark Integration Plan

## Goal

Implement the approved person-centred semantic memory projection, fix person
evidence review and confirmation feedback, add memory-space switching, and run
the three-album benchmark end to end without importing benchmark identities or
answers.

## Architecture

FastAPI, native SQLite, browser JavaScript, buffalo_l detection, AdaFace
identity embeddings, CLIP visual vectors, and the dedicated Sentrix Ollama
runtime remain unchanged. `MemoryStore` remains the authoritative persistence
boundary. Benchmark source files stay outside Git; only the fixture parser,
manifest contract, evaluator, tests, and documentation are committed.

## Tasks

### 1. Documentation checkpoint

Files:

- `docs/PROJECT_MEMORY.md`
- `docs/superpowers/specs/2026-07-31-semantic-benchmark-integration-design.md`
- `docs/superpowers/plans/2026-07-31-semantic-benchmark-integration-plan.md`

Record the person-event graph, pattern projection, evidence modal behavior,
memory-space boundary, benchmark file intersection rules, evaluation-only
labels, and non-video scope. Verify Markdown has no secrets or raw identity
data, run `git diff --check`, and commit the documentation checkpoint on 153
`psh`.

### 2. Synchronize a local implementation copy and write failing tests

Copy the current 153 `psh` source, tests, and documentation to
`/Users/rm001/.codex/tmp/sentrix-edit` without copying `.env`, databases, logs,
runtime state, or benchmark images. Add tests before implementation for:

- person-event memory and pattern grouping after confirmation;
- normalized clothing values retaining raw evidence;
- scope filtering preventing cross-album events and people;
- benchmark manifest intersection and evaluation-label isolation;
- evidence review returning face samples and linked assets;
- confirmation response containing refresh counts;
- merge and split actions returning updated cluster/entity state.

Run the focused tests and verify each new test fails for the current reason.

### 3. Add additive scope and semantic schema migrations

Files:

- `backend/db.py`
- `backend/tests/test_memory_store.py`
- `backend/tests/test_entities.py`

Add `memory_spaces` and additive `scope_id` columns to assets, observations,
events, entities, face clusters, memory vectors, semantic profiles, semantic
claims, and person appearance evidence. Existing records receive
`default-household`.

Add:

- `person_event_memory` for person-to-event roles and evidence;
- `person_patterns` for cross-event aggregates;
- normalized clothing fields and normalization version on semantic claims;
- indexes for scope, person-event, pattern, and evidence lookup.

Migrations must preserve existing databases and remain idempotent. No source
metadata, benchmark face name, or query answer is written by migration.

### 4. Implement person-centred semantic consolidation

Files:

- `backend/db.py`
- `backend/app.py`
- `backend/agent.py`

Refactor `rebuild_person_memory` to rebuild `PersonEventMemory` first, then
derive `PersonPattern`, profile summaries, and compatibility claims. Group
profile output by event and pattern, include co-person IDs and evidence IDs,
and retain revision/conflict status. Keep `person_id` authoritative; display
names are projections.

Add normalized Chinese clothing vocabulary for equivalent values such as
`深色西装外套`, `黑色西装外套`, and `西装外套`. Store normalized values for
aggregation while retaining raw clothing values in appearance evidence.
Scene-level clothing remains observation-only unless a confirmed face-scoped
appearance record exists.

Extend person/profile and knowledge responses with `event_memory`, `patterns`,
and structured evidence counts. Add optional `scope_id` filtering to events,
people, entities, knowledge, assets, and assistant turns. Keep `/api/search`
backward compatible.

### 5. Fix person evidence review and confirmation feedback

Files:

- `src/app.js`
- `src/api.js`
- `src/styles.css`
- `backend/app.py`
- `test/project-structure.test.js`

Separate the people-page `查看证据` action from `确认人物`. The evidence
modal must load and render face-cluster samples, original image links, linked
events, and current identity status. The confirmation modal remains the only
place for pending name/role input.

Return `refresh_counts` from confirmation with affected observations, events,
patterns, claims, and appearance records. Render a visible success notice with
the confirmed name and counts, then reload authoritative data. The notice must
also show server errors and leave the modal usable.

### 6. Complete merge/split UI against existing APIs

Files:

- `src/app.js`
- `src/api.js`
- `src/styles.css`
- `backend/app.py`
- `test/project-structure.test.js`

Add a merge target selector showing eligible clusters in the current
memory-space, a split confirmation showing the selected face and destination
cluster, and success/error feedback. Preserve the existing audited backend
merge/split operations and refresh entity/profile state after each operation.

### 7. Add benchmark manifest, scoped importer, and evaluator

Files:

- `scripts/benchmarks/prepare_household_benchmark.py`
- `scripts/benchmarks/evaluate_household_benchmark.py`
- `scripts/maintenance/rebuild_memory.py`
- `backend/tests/test_benchmark_fixture.py`
- `test/project-structure.test.js`

The preparation script accepts a source root and output manifest path. For
each album it intersects actual image filenames with metadata and face maps,
records unmatched metadata/face entries, preserves missing time/GPS as null,
and keeps face names and query ground truth in evaluation-only sections.

The rebuild command accepts the generated import section, assigns each album
to a `MemorySpace`, and imports only original files plus allowed time/location
and album provenance. It must reject event labels, identities, family roles,
and query answers at the import boundary.

The evaluator reads the generated manifest and database, then reports input
diagnostics, face clustering metrics, scope isolation, event candidate
diagnostics, and query retrieval metrics. It never writes evaluation labels to
SQLite.

### 8. Add scoped web selection and benchmark run controls

Files:

- `src/app.js`
- `src/api.js`
- `src/styles.css`
- `backend/app.py`
- `test/normalizers.test.js`

Add a persistent `MemorySpace` selector to the portal. Switching it reloads
dashboard, timeline, people, knowledge, assets, and search data. Display the
active space in evidence and event views. Do not mix records across spaces.

### 9. Run controlled benchmark and full verification

Transfer the non-Git benchmark source to an ignored 153 runtime directory,
generate the manifest, run a clean rebuild per memory space, and record only
aggregate metrics and unmatched-file counts in `docs/PROJECT_MEMORY.md`.

Run locally and on 153:

```bash
.venv/bin/python -m unittest discover -s backend/tests -v
node --test test/*.test.js
node --check src/app.js && node --check src/api.js
.venv/bin/python -m compileall -q backend scripts
git diff --check
```

Run the benchmark evaluator for all three spaces and perform one manual
confirmation per selected test cluster. Verify the returned profile, event
memory, patterns, evidence, and query results. Verify Sentrix `8090` and web
`4174` remain healthy, FMA `5173` remains untouched, and both Ollama
listeners have zero resident models after processing.

### 10. Commit verified implementation on 153

Transfer only verified source, tests, scripts, and documentation to 153.
Preserve user changes, do not transfer databases or credentials, run the final
verification again, update the project memory with measured results and open
follow-up items, and commit the complete change on `psh` with a Conventional
Commit.
