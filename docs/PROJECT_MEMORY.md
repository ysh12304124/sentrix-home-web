# Sentrix Home Web - Project Memory

## Purpose

Sentrix is a local-first household memory system. It turns original media into
evidence-backed event, semantic, and visual memory. It is implemented natively
in this repository; Cognee concepts informed the design but Cognee is not a
runtime dependency.

## Runtime

- Project on 153: `/home/asus/Github/Sentrix-Home-Web`
- Web UI: `http://192.168.0.153:4174`
- Sentrix API: port `8090`
- Do not use, modify, or stop the FMA web service on port `5173`.
- Main local models: `gemma4:12b`, FunASR (`paraformer-zh`, FSMN-VAD,
  CT-Punc), InsightFace `buffalo_l`, CLIP ViT-B/32.
- Video extraction is intentionally reserved through `VideoMemoryAdapter`.

### Sentrix Model Runtime Isolation

- Sentrix owns a dedicated Ollama process on `127.0.0.1:11435`; the shared
  system Ollama remains on `127.0.0.1:11434` for other projects.
- The dedicated process uses the existing shared read-only model store and
  `gemma4:12b`; it has no private copy of model blobs. Its PID file and logs
  are under `.ollama-sentrix/`, which is runtime state and remains ignored.
- Sentrix processes must set `OLLAMA_BASE_URL=http://127.0.0.1:11435`.
  `OLLAMA_KEEP_ALIVE=0` is required so the process unloads the model after each
  completed request. Before starting a long Sentrix run, check both `/api/ps`
  endpoints and ensure no other 12B model is resident.
- Validated on 2026-07-30: the dedicated runner recognized the RTX 3090,
  offloaded all `49/49` Gemma layers to `CUDA0`, and then unloaded cleanly.
  The host's `nvidia-smi` remains unavailable because NVML userspace and the
  loaded kernel module differ; Ollama's own CUDA runner log is the runtime
  verification source.

## Memory Model

1. Asset and Observation are immutable evidence-oriented records. Every event,
   semantic claim, and answer must link back to an Observation and its original
   Asset.
2. Events are clustered first from original capture time and capture location.
   The title, event type, activity, and summary are generated only after the
   relevant observations are grouped. Import metadata must never supply an
   event name or inferred activity.
3. Semantic memory is person-centred. `semantic_profiles` and
   `semantic_claims` describe confirmed people through supported events and
   evidence. Activity, place, attendance, and capture can be event-level
   multi-value claims. Clothing is a person attribute only when an explicit
   `PersonAppearanceEvidence` record joins a confirmed person to a detected
   face instance, its upper-body crop, and the original observation/asset.
   Scene-level clothing remains on `Observation`; identity attributes remain
   versioned single-value claims.
4. Visual memory stores CLIP vectors, face embeddings, face clusters, and
   original-media pointers. Face clusters use `buffalo_l`, global re-clustering,
   and a medoid representative. A cluster is only a candidate until the user
   confirms it.

## Module Map

Sentrix is organized as one evidence pipeline with replaceable processing
adapters. The major modules and ownership boundaries are:

- **Web UI and API**: imports, timeline, event evidence, person review,
  semantic-memory views, Agent answers, feedback, and task status.
- **Asset and provenance boundary**: original file identity, capture time,
  capture location, album owner/device/album ID, deduplication, and derived
  media pointers.
- **Media processing plane**: image, audio, text, and document observation
  extraction; processing jobs, retries, versions, and failure cleanup.
- **Identity and entity governance**: face instances, face clusters, person
  entities, avatars, confirmation, rejection, merge/split history, and
  identity mentions.
- **Event memory**: observation grouping, event candidates, event boundaries,
  event participants, event revisions, summaries, and evidence membership.
- **Semantic memory**: person profiles, event-level claims, stable identity
  attributes, time-varying activities/places/clothing, relationships, conflict
  status, and claim revision history.
- **Visual memory**: image/observation/face vectors, media similarity, original
  asset references, and evidence-level visual lookup.
- **Agent and retrieval orchestration**: query interpretation, person/event/
  claim/vector/evidence retrieval, evidence validation, answer generation,
  retrieval trace, query gaps, and feedback-driven enrichment.
- **Video memory boundary**: reserved adapter for video fragments, key frames,
  temporal evidence, and video vectors; no video extraction is enabled in the
  current phase.
- **Operations and validation**: rebuild runs, model/version state, audit
  records, benchmark data, acceptance metrics, and service health.

### Module Implementation Reference

| Module | Implementation method | Authority / output |
| --- | --- | --- |
| `backend/app.py` | FastAPI routes construct the store, pipeline and agent once; file ingestion runs through the pipeline; cluster confirmation triggers participant, semantic and event-summary refresh. | HTTP API and application orchestration. |
| `backend/db.py` | SQLite schema migration plus transactional CRUD/projections. `MemoryStore` owns all evidence joins, event candidate scoring, clustering persistence, semantic rebuilds, and audit status. | The only authoritative memory database. |
| `backend/pipeline.py` | Normalizes allowed source provenance, calls modality adapters, persists immutable observations and vectors, then proposes/updates events. | `Asset`, `Observation`, visual vectors, event candidates. |
| `backend/model_clients.py` | Thin adapters for Gemma visual/text JSON extraction, FunASR, buffalo_l detection, AdaFace embeddings and CLIP. Models never write to SQLite directly. | Structured model outputs with model/version provenance. |
| `backend/face_embeddings.py` and `backend/face_clustering.py` | AdaFace face vectors plus quality/pose-aware multi-prototype global clustering. Low-quality detections remain evidence-only. | `FaceInstance`, `FaceCluster`, prototype and metric contracts. |
| `backend/person_appearance.py` | Deterministic bounded expansion of a detected face to head-and-upper-body crop. | Crop coordinates used by person appearance evidence. |
| `backend/agent.py` | Parses query constraints, retrieves claims/events/observations/vectors, validates evidence IDs, and answers deterministically when structured evidence is sufficient. | Answer, evidence layers, retrieval trace and query gaps. |
| `src/` | Plain browser JavaScript renders backend-authoritative event, person, knowledge, asset and search views; `src/api.js` is the only browser API wrapper. | Local UI state only. |
| `server.js` | Serves static files and proxies all `/api/*` requests to Sentrix FastAPI. | Same-origin web gateway; no alternate memory implementation. |

## End-to-End Data Pipeline

The complete data path is the following. Each arrow represents a persisted
contract, not an implicit model call:

```text
Original photo/audio/text/video
  │
  ▼
Asset Intake
  ├─ content identity and duplicate decision
  ├─ capture time/location
  └─ album owner/device/album provenance
  │  output: Asset
  ▼
Media Processing Job
  ├─ image observation extraction
  ├─ audio transcript observation extraction
  ├─ text/document observation extraction
  └─ video adapter boundary (reserved)
  │  output: Observation + derived media references
  ▼
Evidence Normalization
  ├─ canonical Chinese observation fields
  ├─ object, clothing, spatial, OCR, transcript signals
  ├─ face instances and face embedding provenance
  └─ CLIP/image/audio/text vector records
  │  output: immutable Observation, FaceInstance, MemoryVector
  ▼
Identity Candidate Layer
  ├─ face cluster candidate
  ├─ representative face samples/avatar
  └─ pending/confirmed/rejected entity state
  │  output: EntityMention and auditable identity candidate
  ▼
Event Builder
  ├─ capture-time and capture-location candidate grouping
  ├─ source album/device and visual evidence compatibility
  ├─ event boundary and ambiguity record
  └─ post-group model summary
  │  output: Event -> Observation/Evidence and EventParticipant
  ▼
Person-Centred Semantic Consolidation
  ├─ Person -> Event -> Evidence chain
  ├─ event-level activity/place/capture claims
  ├─ person-level appearance claims only when face-to-attribute evidence exists
  ├─ scene-level clothing retained on Observation when attribution is unknown
  ├─ confirmed identity/role claims
  └─ profile projection and revision history
  │  output: SemanticProfile, SemanticClaim, Relationship, Fact
  ▼
Three Memory Views
  ├─ episodic: what happened, when, where, who, and evidence
  ├─ semantic: what is known about each person across events
  └─ visual: which original image/region/vector can verify it
  │  output: native SQLite records + vector spaces
  ▼
Agent Recall
  ├─ parse person/time/place/activity/object/clothing intent
  ├─ retrieve claims and profiles first for person questions
  ├─ locate events and original observations
  ├─ validate every answer reference against evidence
  └─ return answer, evidence layers, trace, and confidence
  │  output: Answer + Evidence + RetrievalTrace
  ▼
Feedback and Maintenance Loop
  ├─ user names/rejects/merges/splits a person candidate
  ├─ user confirms or corrects a fact/answer
  ├─ query gap requests targeted visual enrichment
  └─ affected events, claims, profiles, vectors, and answers rebuild
```

### Pipeline Contracts

| Stage | Reads | Writes | Downstream consumers |
| --- | --- | --- | --- |
| Intake | original file and allowed provenance | `Asset` | processing jobs, evidence UI |
| Observation | `Asset`, modality payload | `Observation`, derived references | event builder, visual memory |
| Identity | face evidence and model provenance | `FaceInstance`, `FaceCluster`, `Entity`, `EntityMention` | person UI, events, semantic memory |
| Person appearance | confirmed person, face instance, upper-body crop | `PersonAppearanceEvidence` | clothing claims, person profile, Agent |
| Event | observations, asset metadata, identity mentions | `Event`, `EventObservation`, `EventParticipant` | timeline, semantic consolidation, Agent |
| Semantic | confirmed entities, events, observations | `SemanticProfile`, `SemanticClaim`, `Fact`, `Relationship` | person pages, Agent |
| Visual | original media and derived visual features | `MemoryVector`, evidence pointers | Agent fallback and verification |
| Recall | query plus all memory views | `Answer`, `Evidence`, `RetrievalTrace`, `QueryGap` | web UI and feedback |
| Feedback | user correction or query-gap resolution | revisions and rebuild tasks | every affected memory view |

The required logical chain is always:

```text
Asset -> Observation -> Event -> Person/Entity -> SemanticClaim/Profile
  \________________________ Evidence ________________________/\
```

An answer may start from a claim, event, vector, or raw observation, but it
must be able to traverse this chain back to the original asset. A person
profile is therefore a cross-event summary, not a second copy of image
captions; an event is the unit that binds people, activity, time, place, and
evidence together.

For a person appearance claim, the stronger required chain is:

```text
Asset -> Observation -> FaceInstance -> PersonAppearanceEvidence -> SemanticClaim
                             \-> confirmed Person -> Event
```

`PersonAppearanceEvidence` stores the selected face, source asset and
observation, deterministic crop coordinates, model name, confidence, and the
target-only clothing array. The crop expands from the detected face down to
the upper body; it never falls back to the full scene. On person confirmation,
the API analyzes at most the highest-quality confirmed face per event. Empty
model results remain evidence records but create no clothing claim.

## Project Memory Maintenance

`docs/PROJECT_MEMORY.md` is the living structural record for this repository.
Every architecture, data-contract, model-boundary, validation, or deployment
decision updates the relevant section. Structural sections describe modules,
records, and data flow; implementation status records only the current state;
validation entries record commands, dates, and measured results. Do not place
secrets, transient prompts, raw user identity data, or unverified benchmark
claims in this file.

### Repository Ownership

This file is the single current-state handoff document. `README.md` contains
only runnable entry points and links here. Approved historical designs and
implementation plans remain under `docs/superpowers/`; Git history retains
superseded root-level drafts and duplicate implementations. Do not create a
second current architecture, implementation plan, or test-data document.

The stable repository layout is:

```text
backend/                 authoritative API, persistence, pipeline, models, tests
src/                     browser code and CSS
scripts/runtime/         service lifecycle utilities
scripts/maintenance/     explicit destructive or long-running operations
scripts/benchmarks/      controlled evaluation tools
scripts/fixtures/        reproducible test-data and metadata generators
test/                    frontend and repository-layout regression tests
docs/                    current memory and approved historical records
```

`server.js` is a static-file server and same-origin proxy to `backend.app`.
It contains no mock API or Cognee fallback path. `backend.app` and its
`MemoryStore` are the authoritative behavior and persistence boundaries.

## Source Data Contract

Imported photos may contain only these external facts:

- original file;
- capture time;
- capture location;
- source album owner, device, and album ID.

Album ownership is provenance, not a visible person, family role, event label,
or relationship. The system must infer event content from observations and
must wait for user confirmation before treating a face cluster as a named
person.

## Identity Workflow

1. `buffalo_l` detects faces and groups embeddings.
2. The UI shows cropped face samples from `face_instances.bbox_json`; clicking
   a sample opens its original image.
3. The user supplies a name and optional family role.
4. The system writes entity mentions, event participant roles, semantic claims,
   and a person profile, then re-summarizes related events with confirmed names.

## Validation Baseline

The current automated baseline is 78 Python tests and 7 Node tests. Before
claiming a change complete, run:

```bash
.venv/bin/python -m unittest discover -s backend/tests -v
node --test test/*.test.js
node --check src/app.js
node --check src/api.js
.venv/bin/python -m compileall -q backend scripts
```

## Test Data

`data/full-chain` is public test imagery with synthetic, auditable capture
time/location and neutral source-album provenance. It must not contain
prewritten event names, family identities, roles, or relationships. Regenerate
its metadata with `scripts/fixtures/prepare_test_metadata.py` before a clean
rebuild.

## Current Decisions (2026-07-29)

The selected face-recognition direction is option 3: add a quality-aware face
embedding model, with AdaFace as the primary candidate and MagFace as the
comparison model. The adapter and quality-aware path are implemented, but the
real checkpoint is not installed yet. The existing `buffalo_l`
detector remains the initial detection boundary; the embedding model is kept
behind a replaceable Sentrix adapter so the detector and identity model can be
changed independently.

The face solution must include more than a model swap:

- combine feature quality, detection confidence, face size, pose, sharpness,
  and occlusion signals;
- maintain several high-quality prototypes per person instead of one mean
  embedding, so frontal, left-profile, and right-profile views are represented;
- use quality-weighted global clustering and treat low-quality faces as weak
  evidence rather than allowing them to create or redefine a strong identity;
- support user-audited cluster merge and split operations with revision history;
- evaluate pairwise precision, recall, F1, singleton ratio, missed merges, and
  false merges on the same benchmark after every clustering change.

## Current Findings

The 153 Sentrix service was checked on 2026-07-29. The Sentrix API is on port
8090, the web portal is on port 4174, and the unrelated FMA service on port 5173
must remain untouched. The service reported Gemma, FunASR, InsightFace, and
CLIP as ready. The 153 Sentrix repository is currently on `main` at commit
`6ad4ae8`.

The current 153 database contains 54 assets, 54 observations, 4 events, 80
face instances, and 16 face clusters. It contains no confirmed people, event
participant roles, or semantic profiles yet. Existing automated tests pass (23
Python tests and 4 Node tests), but the test suite does not yet cover profile
versus frontal matching, low-quality faces, bridge samples, or real event
over-aggregation.

The current clustering root causes are:

1. Online ingestion uses a single-threshold greedy assignment at cosine 0.55.
2. A cluster representative is updated with an unweighted mean, so profile or
   noisy samples can move the representative away from the identity.
3. Quality and pose are recorded but do not gate or weight clustering.
4. Offline re-clustering uses threshold-connected components; one bridge sample
   can join otherwise separate identities, and online ingestion does not use
   the global result immediately.

The current event aggregation also needs correction. It considers a six-hour
window plus location/album matches, then takes the first candidate. It does not
score activity, object, visual-place, person overlap, or source-device evidence,
so unrelated activities at the same place can be merged.

## Cognee-Inspired Boundaries

Cognee is a design reference, not a Sentrix runtime dependency. The useful
ideas are explicit memory operations and retrieval orchestration:

- `remember` maps to Asset -> Observation -> Event -> semantic claim writes;
- `cognify` maps to deterministic batch rebuilding of events, entities,
  profiles, claims, and vectors;
- `recall` maps to hybrid retrieval across people, events, claims, graph edges,
  vectors, and original evidence;
- `improve` maps to user confirmation, query-gap resolution, and claim feedback;
- session context and retrieval traces should remain local and auditable;
- every generated answer must retain a path back to Observation and Asset.

Sentrix must continue to own the SQLite evidence store, native vector index,
entity relationships, identity confirmation, and privacy boundary. Do not add
Cognee as an online dependency or project household evidence into an opaque
external graph.

## Remaining Work

The following is the active backlog. P0 items are required before the next
meaningful 153 acceptance; P1 items follow the core correctness work; P2 items
are deliberately deferred.

| Priority | Area | Work | Status |
| --- | --- | --- | --- |
| P0 | Face | AdaFace adapter, model/version configuration, quality-aware embedding, and benchmark harness | Partial: adapter and quality path implemented; real checkpoint not installed |
| P0 | Face | Multi-view prototypes, quality-weighted global clustering, and confirmed-person constraints | Implemented locally; 153 rebuild pending |
| P0 | Face | Cluster merge/split APIs, audit revisions, and person-memory rebuild after confirmation | Implemented locally; 153 sync validation pending |
| P0 | Face | Face benchmark with pairwise precision/recall/F1, singleton, false-merge, and missed-merge metrics | Implemented locally; real AdaFace benchmark blocked by checkpoint |
| P0 | Events | Candidate scoring using time, location, source, activity, objects, visual place, and person overlap | Implemented locally; needs 153 data validation |
| P0 | Events | Event boundaries and deterministic handling of multiple candidates | Implemented locally; candidate ambiguity metadata and UI display added |
| P0 | Entities | Unify legacy `persons` and native `entities` so two person systems cannot diverge | Core sync implemented; legacy endpoints remain compatibility wrappers |
| P0 | Semantic | Evidence-backed Person -> Event -> Claim/Profile rebuild and versioned conflict handling | Core rebuild implemented; richer conflict policy remains P1 |
| P0 | Agent | Query parsing, hybrid retrieval, evidence validation, and relevance re-ranking | Evidence validation and structured retrieval trace implemented; ranking remains to tune |
| P0 | Tests | Regression tests for side/front faces, bridge samples, event over-aggregation, and person confirmation | Implemented locally; 57 Python tests and 4 Node tests pass |
| P1 | Semantic | Chinese canonical fields, normalized predicates, confidence provenance, and habit thresholds | Common predicates and confidence source implemented; habit thresholds remain |
| P1 | Agent | Query gaps, targeted visual enrichment, feedback updates, and retrieval trace detail | Query gaps, feedback, lexical re-ranking, structured trace, and evidence layers implemented |
| P1 | UI | Reorder results as answer -> people/events -> claims -> observations -> original assets -> gaps | Implemented locally; 153 sync validation pending |
| P1 | Data | SHA-256 deduplication, real EXIF time/GPS/device extraction, and source provenance validation | SHA-256, duplicate reuse, EXIF time/GPS/device boundary implemented; broader provenance validation remains |
| P1 | Operations | Model/rebuild versioning, failed-stage retry, clean rebuild checks, and 153 small-batch acceptance | Idempotency, failure cleanup, schema migration, retry endpoint, and local checks implemented; 153 acceptance pending |
| P2 | Face | MagFace comparison and possible production deployment if AdaFace is insufficient | Planned |
| P2 | Video | Keep `video_memory_adapter` reserved; no video encoding memory in this phase | Intentionally deferred |
| P2 | Session | Local session memory inspired by Cognee, after core evidence retrieval is stable | Planned |

## Implementation Status (2026-07-30)

Implementation has started from the approved option 3 direction. No production
database rebuild has been run. The detailed first-phase plan is:

`docs/superpowers/plans/2026-07-30-adaptive-face-memory-plan.md`

The first implementation slice is complete locally for AdaFace/MagFace adapter
boundaries, quality metadata, multi-view prototype behavior, model/version
isolation, confirmed-person protection, quality-aware global clustering, event
candidate scoring, schema migration, Asset idempotency, and failed-pipeline
cleanup. The working sequence was red test -> minimal implementation -> full
regression verification. 153 services remain running and 153 data remains
untouched; no AdaFace checkpoint or production rebuild has been applied there.

Local verification on 2026-07-30: 57 Python tests and 4 Node tests passed; JS
syntax check, Python compile check, and `git diff --check` passed. A direct local
FastAPI health import was not available because the macOS Python environment
lacks the optional `fastapi` package; 153 remains the API runtime validation
environment.

Important safety fixes included in this slice:

- different embedding model/version spaces cannot cluster together;
- two confirmed people cannot be automatically merged;
- low-quality faces cannot join an existing identity cluster;
- repeated Asset processing is idempotent;
- failed processing removes only that Asset's derived records;
- legacy person confirmation synchronizes or creates its native entity;
- event aggregation stores an explainable score breakdown;
- schema indexes are created after additive migrations;
- user face-cluster merge and single-sample split operations write entity
  revision audit records;
- uploaded assets receive SHA-256 content identity, duplicate uploads reuse the
  existing Asset, and image EXIF time/GPS/device metadata is preserved when
  explicit import metadata is absent;
- Agent responses expose structured retrieval stages and evidence layers for
  answer, people/events, claims, observations, assets, and query gaps;
- the UI exposes cluster merge/split actions and ordered evidence layers.
- semantic claims normalize common Chinese predicates and retain a confidence
  source;
- rebuild scripts write versioned `rebuild_runs` audit records with final
  statistics.

The next blocking step is to install and validate a real AdaFace checkpoint on a
controlled environment, run the benchmark, then perform a small 153 validation
without touching FMA 5173 or rebuilding the production database. Non-blocking
follow-up work includes richer Chinese predicate normalization, habit
thresholds, retrieval ranking calibration, MagFace comparison, and broader
provenance validation.

## 153 Validation (2026-07-30)

The verified local code and tests were synchronized to
`/home/asus/Github/Sentrix-Home-Web` and Sentrix services were reloaded. The
following checks passed on 153:

- `57` Python tests;
- `4` Node tests;
- Python compile check;
- 8090 and 4174 health endpoints;
- FMA 5173 health endpoint remained normal.

The live Sentrix health response now reports `identityModel=adaface`,
`identityConfigured=false`, and `identityReady=false` with
`adaface checkpoint is unavailable`. This is the expected controlled state:
InsightFace detection remains ready, but no identity embedding or new face
cluster is created until a real AdaFace checkpoint is installed and validated.
The 153 database remains unchanged in business data at 54 assets, 54
observations, 4 events, 80 face instances, 16 face clusters, 0 semantic
profiles, and 0 event participant rows. The additive schema migration added
`confidence_source` and `rebuild_runs`; no production rebuild was run.

The final 153 acceptance after the next implementation slice also passed:

- `57` Python tests and `4` Node tests;
- JavaScript syntax, Python compile, and `git diff --check`;
- Sentrix API `8090` health, Sentrix Web `4174` HTTP 200, and FMA `5173` HTTP
  200;
- business data counts remained unchanged; `rebuild_runs=0` confirms no rebuild
  was executed;
- the active branch is still `main` with uncommitted, verified worktree
  changes. The formal `psh` commit destination has not been created.

## Execution Checklist (2026-07-30)

- [x] Adapter boundary, quality metadata, model/version isolation, and
  multi-view face prototypes.
- [x] Quality-aware global clustering, bridge protection, confirmed-person
  protection, pairwise metrics, and regression tests.
- [x] User face-cluster merge/split APIs, UI actions, and entity revision audit.
- [x] Event candidate scoring, ambiguity metadata, deterministic boundaries, and
  regression tests.
- [x] Asset SHA-256 identity, duplicate reuse, EXIF time/GPS/device boundary,
  and failure cleanup.
- [x] Legacy person/native entity synchronization and person-centered semantic
  profile/claim rebuild.
- [x] Chinese predicate normalization, claim confidence source, query gaps,
  feedback, lexical ranking, structured retrieval trace, and evidence layers.
- [x] Query UI order: answer, people/events, claims, observations, assets, and
  gaps.
- [x] Local and 153 verification with FMA `5173` isolation.
- [ ] Obtain and checksum-verify the official AdaFace checkpoint, run real
  aligned-face inference and benchmark, then perform controlled 153 validation.
- [ ] Create/verify the formal 153 `psh` branch and commit the verified backend
  changes there.
- [ ] After AdaFace metrics pass, decide whether to run a controlled derived
  memory rebuild; do not rebuild production data before that gate.

## AdaFace Installation Attempt (2026-07-30)

The 153 SSH alias was found to resolve incorrectly to `0.0.0.153`; the valid
internal target is `192.168.0.153` with user `asus`. Direct SSH access to the
valid target was restored without changing the project SSH configuration.

The 153 `stmem` environment was verified with Python 3.10.20, PyTorch 2.5.1
CUDA 12.1, and an available NVIDIA RTX 3090. `torch.cuda.is_available()` is
true and one GPU is visible. The earlier CUDA/NVML warning does not prevent
PyTorch CUDA initialization in this environment.

Official AdaFace source was copied to the independent directory
`/home/asus/models/AdaFace` at upstream commit
`c60eaa786a42c03444f3df7096dbaf9d57ae010d`. Only `net.py`, `LICENSE`, and the
commit marker were copied; the source file hashes match the local download.
The source is not part of the Sentrix repository and no production data was
changed.

At the time of this entry, the official R50/MS1MV2 checkpoint was still
uninstalled. Its official source is
the AdaFace README Google Drive file ID
`1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI`. Both 153 and the local environment timed
out when reaching Google Drive. GitHub is reachable locally but large clone
and archive downloads are unreliable, so no unverified model mirror was used.
This historical constraint was superseded by the verified live checkpoint and
runtime validation recorded in the final acceptance entry below.

The adapter was tightened to support injected test backends correctly and to
load the official checkpoint's `model.*` state dictionary without falsely
labeling another model as AdaFace. Local AdaFace-related tests remain green.
The next acceptance step is to obtain the exact official checkpoint through a
network path that can be checksum-verified, load it on the 153 RTX 3090, run a
single aligned-face inference, then run a small benchmark before any rebuild.

## Execution Order

1. Add failing tests and a reproducible face benchmark for the current failure
   modes.
2. Implement AdaFace/MagFace adapter boundaries and quality-aware face records.
3. Implement multi-view prototypes, global clustering, merge/split, and metrics.
4. Implement event candidate scoring and event rebuild tests.
5. Complete person confirmation propagation and semantic profile/claim rebuild.
6. Complete Agent query parsing, hybrid retrieval, ranking, gaps, and feedback.
7. Reorder the portal evidence presentation and add API/UI regression coverage.
8. Rebuild a controlled 153 dataset, validate orphan/evidence invariants, and
   record the measured results here.

Whenever architecture, model, data-flow, or acceptance status changes, update
this file in the same change set. Do not record credentials, tokens, or
environment secrets here.

## Virtual Family Album Acceptance (2026-07-30)

The controlled 153 test dataset contains 120 LFW-derived images, four held-out
identity labels used only for evaluation, capture time/location, and source
album provenance. The Sentrix image prompt and event scorer do not read the
evaluation-only `event_id`, `activity_hint`, or `source_identity` fields. A
future import boundary should nevertheless whitelist allowed provenance fields
before storing raw metadata.

The first foreground rebuild stopped after `asset_108.jpg` because its
interactive parent session expired. A second detached `nohup` rebuild completed
all 120 assets with zero processing failures, six event summaries, 148 AdaFace
face instances, and 34 pending clusters after global re-clustering.

Acceptance results:

- Face clustering: **not passed**. Four known identities produced 34 clusters;
  pairwise precision was 0.9985, recall 0.4837, F1 0.6517, and singleton ratio
  0.6176. The failure is identity fragmentation, not broad false merging.
- Initial semantic state: expected zero profiles/claims before a person is
  confirmed.
- Confirmation propagation: **passed** for a pure 29-sample cluster confirmed
  as `测试成员甲`. The system created an avatar, profile, 51 evidence-backed
  claims, and participant roles in all six linked events.
- Agent query: **not passed**. The query `测试成员甲参与过哪些活动？` returned
  validated evidence and a structured retrieval trace, but its final answer
  fell back to a list of 47 observations instead of aggregating the person's
  semantic activity claims.

These results are the failed baseline. They were superseded by the final
controlled rerun below; retain them only as a regression comparison.

## Current Remediation (2026-07-30)

The complete end-to-end pipeline above remains the architectural source of
truth. The current implementation work addresses two failed gates without
changing the data contract:

- **Identity candidate admission:** detected faces remain observation evidence,
  but weak/small detections no longer automatically create person candidates.
  Only identity-eligible face instances enter `FaceCluster` and person-entity
  projection; evidence-only face instances remain traceable to their original
  asset but carry no `cluster_id`. This prevents detection noise from becoming
  dozens of pending people while preserving multi-person photos.
- **Cluster lifecycle:** re-clustering retires obsolete candidate clusters and
  their unconfirmed entities together. Confirmed entities remain protected from
  automatic merging.
- **Semantic recall:** person confirmation rebuilds derived activity/place
  claims at Event -> Evidence granularity, and the Agent prefers those claims
  and the person profile before raw observations for person activity questions.
- **Controlled rerun rule:** the 120-image virtual album must be rebuilt from
  source after the admission policy and dedicated `11435` model runtime are
  both active. Previous 34/35-cluster runs remain historical controls only.

## Final Controlled Acceptance (2026-07-30)

The controlled virtual-family album was rebuilt from its 120 original files
after all runtime and data-flow corrections. Processing completed with 120/120
assets successful, zero failures, six event summaries, and a completed rebuild
record. Evaluation labels were read only from the external manifest after the
run; they were not imported into Sentrix assets, observations, events, or
semantic memory.

### Runtime Gate

- `scripts/runtime/start_sentrix_ollama.sh` owns the project-local Ollama listener on
  `127.0.0.1:11435` and refuses to start it if shared `11434` has a resident
  model.
- The dedicated runner was verified to use the RTX 3090 (`CUDA0`) with all
  49 Gemma layers offloaded. It shares the existing model store but has a
  separate process, listener, PID/log directory, and lifecycle.
- Sentrix calls include `keep_alive=0`; a completed request unloads the 12B
  model, preventing a persistent duplicate model from competing with other
  projects. At final service verification both `11434` and `11435` had no
  resident model.
- The rebuild now aborts before deleting derived data when AdaFace is enabled
  but unconfigured. This prevents a misleading successful rebuild with zero
  identity evidence.

### Face Acceptance

- Detection evidence: 148 detected face instances.
- Identity candidates: 84 high-quality face instances were eligible for
  automatic person clustering.
- Evidence-only faces: 64 weaker detections were retained as face evidence
  tied to their original observations and assets, but did not create a person
  candidate or pending entity.
- Candidate result: four active pending clusters and four pending person
  entities for four held-out identities; no singleton candidate clusters.
- External-manifest pairwise evaluation of the 84 candidate samples: precision
  `1.0000`, recall `1.0000`, F1 `1.0000`, false positives `0`, false negatives
  `0`.

The admission contract is intentional: `FaceInstance` is an evidence record;
only `identity_eligible` instances obtain a `cluster_id` and can project into a
pending `Entity`. It preserves all detected faces without turning weak
detections into fragmented people.

### Semantic, Feedback, and Query Acceptance

- Before confirmation: zero person profiles and zero active semantic claims,
  as required by the user-confirmation boundary.
- Confirmation propagation: one externally verified pure candidate cluster was
  confirmed as `测试成员甲`. The system created its avatar/profile, five
  event-backed activity claims, and participant roles in all five linked
  events.
- Event-backed semantic consistency: the confirmed person has five activity
  claims; every claim has both an event ID and observation evidence. Each is
  traceable as `Asset -> Observation -> Event -> Person -> SemanticClaim`.
- Agent query: `测试成员甲参与过哪些活动？` returned all five activity claims and
  all five supporting event IDs, rather than falling back to a list of image
  captions. The retrieval trace included lexical, semantic, vector, and
  evidence-validation stages.
- Structured recall: the agent now filters date and place questions to matching
  events, uses original observation objects for object questions, and answers
  from evidence without starting a model call when that evidence is sufficient.
  On final verification, `家中餐厅发生了什么？` and `2025-05-10发生了什么？`
  each returned only the two matching events; `麦克风相关证据` returned 12
  original image references without loading `gemma4:12b`.
- Clothing boundary: scene-level `Observation.clothing` never becomes a person
  fact by co-occurrence. That conservative baseline was replaced by the
  face-scoped adapter described below.
- Visual vectors: the project-local CLIP ViT-B/32 checkpoint at
  `data/models/clip/ViT-B-32.bin` is auto-discovered by the adapter and was
  verified as a complete 302/302-key state dict with a successful 512-dimension
  image embedding. The API health check reports CLIP ready after service restart.

### Ongoing Maintenance Gate

For every architecture, model, data-contract, service-runtime, or acceptance
change: update the relevant Module Map / End-to-End Data Pipeline section and
append a dated measured result here. Before claiming a rebuild valid, verify:

1. AdaFace is configured and completes a real aligned-face inference.
2. Exactly one 12B model is resident across `11434` and `11435`.
3. Rebuild completion, processing failures, cluster/face admission counts,
   and external controlled metrics.
4. A confirmation propagation check and an intent-specific Agent query with
   event and observation evidence.
5. Verify that person-level attributes have person-level evidence; do not
   promote scene-level properties such as clothing to a person by co-occurrence.

## Person Appearance Acceptance (2026-07-30)

The person-level clothing adapter is now enabled and requires a confirmed
identity mention. It stores `person_appearance_evidence` as a distinct record:
`PersonAppearanceEvidence -> FaceInstance -> Observation -> Asset`; only these
records can project `SemanticClaim(dimension=clothing, predicate=穿着)`.
Scene-level `Observation.clothing` remains separate and cannot pass this gate.

For the already externally verified confirmed person `测试成员甲`, re-confirming
the same cluster selected one high-quality face in each of five linked events.
The dedicated Sentrix Ollama on `11435` produced five target-only appearance
records and twelve clothing/accessory claims, including suit jackets, shirts,
ties, an eyeglass frame, and a ring. The query `测试成员甲穿过什么衣服？` returned
the claims plus all five `PersonAppearanceEvidence` records and their original
asset IDs. Both `11434` and `11435` reported an empty resident-model list after
the request, satisfying the one-12B residency rule.

Automated validation after this change: 78 Python tests, 4 Node tests, JS
syntax checks, and Python compile checks passed. The known test-time warning
that a synthetic `ClipAdapter` has no pretrained weights is limited to unit
test construction; the deployed health endpoint reports the project checkpoint
as ready.

## Current Work Queue

The following work is intentionally not represented as complete:

1. **Event segmentation quality**: current event grouping combines capture
   time/location, source provenance, activity/object overlap, visual place, and
   confirmed-person overlap. It needs a controlled split/merge benchmark for
   same-time same-place but distinct activities before its production threshold
   is widened.
2. **Appearance normalization**: face-scoped clothing evidence is correct and
   traceable, but semantically equivalent values such as `深色西装外套` and
   `黑色西装外套` are still separate claims. Add evidence-preserving attribute
   normalization only after a reviewed Chinese vocabulary and evaluation set
   exist.
3. **Identity operations UI**: merge and split API/audit behavior exists;
   complete the corresponding user review controls and regression coverage.
4. **Video adapter**: the interface boundary is reserved only. No video
   decoding, keyframe extraction, temporal evidence, or vector retrieval is
   currently implemented.
5. **Operationalization**: add a managed service definition for the Sentrix
   API and dedicated Ollama runner after confirming host-level ownership and
   restart policy. Do not alter the shared Ollama process.

## Completed Work Snapshot

- Evidence-first native SQLite memory model: assets, observations, events,
  entities, face instances/clusters/prototypes, semantic profiles/claims,
  person appearance evidence, vectors, feedback, and rebuild audit records.
- Source metadata allowlist: capture time/location and album provenance only;
  event labels, activities, identities, and relationships are rejected at
  import.
- AdaFace identity embeddings with buffalo_l face detection, quality-aware
  candidate admission, multi-view global clustering, and controlled external
  evaluation.
- User naming propagation from face cluster to entity mentions, events,
  profiles, claims, event summaries, and evidence-backed Agent recall.
- Target-only person appearance extraction with an independent evidence record
  before clothing claims can be created.
- Dedicated CUDA-capable Sentrix Ollama runtime on `11435`, auto-unloading its
  12B model to coexist with the shared `11434` service.
- 120-image controlled acceptance, full backend/frontend regression suite, and
  repository cleanup that removed stale root duplicate implementations, mock
  gateway behavior, and duplicate test suites.
