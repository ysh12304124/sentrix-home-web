# Semantic Entity Roadmap

## Goal

Build an evidence-first memory graph for Person, Place, Time, Object, Event,
Trip, and Mood. Every current value must identify its source, confidence,
revision history, and original media evidence. User edits take precedence over
derived values.

## Architecture

```text
Asset -> Observation -> EntityEvidence -> EntityProperty -> Event/Relationship
                                      -> Profile/Trip/Mood projection
```

- `entities` remains the stable cross-observation identity table.
- `entity_properties` stores versioned fields such as place alias, privacy,
  person role, and derived scene attributes.
- `entity_evidence` is represented by existing `entity_observations` and
  links every entity back to observations and assets.
- Existing `entity_revisions` records names/cluster changes; P0 adds explicit
  property revisions without replacing that audit trail.

## Technology

- Python 3, FastAPI, SQLite, unittest
- Plain browser JavaScript and CSS
- Existing local Gemma, AdaFace, buffalo_l, and CLIP adapters

## P0: Stable Entity Contract And User Governance

1. Add an `entity_properties` table with `id`, `entity_id`, `key`,
   `value_json`, `source`, `confidence`, `status`, `evidence_ids_json`,
   `revision`, `supersedes_property_id`, timestamps, and a unique active
   property key per entity.
2. Add `MemoryStore.maintain_entity_property()` for derived values. It appends
   a pending revision when a derived value conflicts with a current active
   value and never overwrites a user value.
3. Add `MemoryStore.set_entity_property()` for user corrections. It supersedes
   the current property revision and stores the user source and evidence.
4. Expose property history and current properties in `/api/entities/{id}`.
5. Add UI controls for changing a place alias and private flag, then show the
   current value, source, confidence, and original files in entity detail.
6. Aggregate identical relationships by their stable subject/predicate/object
   tuple, preserving evidence count and time range rather than inserting a
   visible duplicate for every observation.

Acceptance: user edits persist as current values after a rebuild; derived
updates do not overwrite them; every property can open at least one evidence
file when evidence exists.

## P1: Place And Time

1. Split place data into immutable coordinates, derived scene type, canonical
   place identity, and optional user alias.
2. Add local reverse-geocode/POI adapter behind a disabled-by-default boundary;
   persist provider, precision, and evidence instead of treating a POI as raw
   EXIF truth.
3. Derive `part_of_day`, season, year/month/day, and local calendar festival
   from timestamps. Birthday and life-stage stay user-maintained properties.
4. Add place privacy. Hidden places expose a user alias or coarse scene label
   to normal pages and Agent responses, never precise coordinates.
5. Add Place -> Event, Place -> Person and Time -> Event aggregate views with
   representative media selected from evidence.

Acceptance: new assets append to a stable place ID; aliases survive reindex;
private coordinates do not appear in standard entity/API responses.

## P1: Event And Object Semantics

1. Maintain explicit event-to-person, event-to-place, event-to-object,
   event-to-time, and event-to-mood projections from existing evidence.
2. Add an explainable event cover selector using usable resolution, sharpness,
   representative entity coverage, and event recency; persist score inputs.
3. Normalise objects with raw label plus controlled category. OCR text is
   stored as observation evidence, not silently turned into a permanent object
   name.
4. Add object salience only after a detector supplies a bounding box or a
   bounded visual score. Do not infer object ownership without user approval.
5. Allow users to revise event title, type, location and cover with a revision
   chain.

Acceptance: event detail presents stable entity links and a cover-selection
reason; object grouping preserves the original model terms and files.

## P1: Person Profiles And Relationships

1. Continue confirmed-only profile projections for first/last seen, frequent
   places, frequent activities, and representative face.
2. Add user-confirmed `is_self`, role, circle/group, and relationship fields
   through `entity_properties`.
3. Generate co-occurrence and intimacy candidates as pending evidence-backed
   values only. Never expose them as facts without confirmation.
4. Keep person appearance attributes tied to face/body evidence. Scene-level
   clothing cannot become a person property.

Acceptance: confirming, splitting, or rejecting a face cluster rebuilds all
derived person projections without changing user-authored properties.

## P2: Trip And Mood

1. Create a `trips` table only when a candidate spans multiple days or places
   with an evidence-backed event sequence. Persist candidate status first.
2. Trip fields include time range, ordered events, cities/places, companions,
   type candidate, evidence and revision history. User naming/merge/split is
   required for active trip identity.
3. Introduce controlled mood vocabulary with raw model mood, normalized mood,
   confidence source, and evidence. Visual style stays image-level.
4. Background music and annual-recap suggestions remain presentation outputs;
   they do not become factual memory properties.

Acceptance: an ordinary single-day event cannot be promoted to a trip; mood
normalization retains raw labels and evidence.

## P1: Semantic Entity Consolidation

1. Generate same-type semantic entity clusters for Place, Object and Mood from
   normalized labels, text similarity and shared Observation/Event context.
2. Keep every source entity stable until a user confirms a merge. A candidate
   stores canonical suggestion, aliases, confidence, similarity rationale and
   all original evidence IDs.
3. Never cluster people with this mechanism. Identity remains face-evidence
   governed. Never merge entities across MemorySpace boundaries.
4. Exclude private place aliases from cross-entity similarity text; use their
   stable IDs only after permission checks.
5. Add merge, reject and split review with revision history; a rejected
   candidate must not reappear unchanged.

Acceptance: near-equivalent labels such as "湖边" and "水边" can be presented
as a review candidate with source observations, while raw labels and stable
IDs remain intact until confirmation.

### Implemented Browse Projection

The default entity directory and semantic recall now read an automatically
maintained semantic-group projection. Every non-person source entity remains
stable, while same-type labels that resolve to the same controlled semantic
key appear in one card. The card and detail view show its member labels,
cumulative evidence, events and original media. This automatic grouping needs
no user confirmation because it is the default semantic memory view, not a
destructive rewrite of source entities.

## P1: Digital Memory Steward

The accepted product role is a neutral household memory steward, not an
imitation of a family member. Full protocol: `digital-memory-steward.md`.

1. Introduce a read-only tool registry for constraint resolution, entity
   introduction, event lookup, timeline, comparison, recommendation and
   original evidence opening.
2. Route structured constraints before semantic/vector retrieval. Record the
   selected tools in an ordered `tool_trace` alongside `retrieval_trace`.
3. Return clarification candidates on ambiguity; return a query gap on missing
   evidence; never use a vector score as standalone evidence.
4. Bind user feedback to an explicit entity, event, property or relationship.
   Review/confirmation are separate, audited write tools.
5. Build an offline household question set for lookup, explanation, compare,
   recommendation, ambiguity and refusal acceptance gates.

Acceptance: all answers expose their tool route and evidence layers; no
feedback or model output can create a fact without an explicit audited target.

## Cross-Cutting Delivery Gaps

1. Produce a reliable labeled household face benchmark before claiming a
   95-percent household clustering result; LFW alone is not a substitute.
2. Publish a reproducible end-to-end performance benchmark with fixed hardware
   and data before claiming the five-times processing target.
3. Add local, auditable POI data before reverse geocoding; do not send family
   coordinates to a network service. Backfill historical `geo` only through a
   separately backed-up maintenance run.
4. Complete Time festivals/life stages, Object salience/OCR presentation, Mood
   score/style/user correction, and Trip cities/companions only with evidence.
5. Add service supervision, task progress, retry/recovery metrics, backup
   restore drills and a bounded video evidence plan.

## Verification Sequence

1. Add a failing unittest for each property/revision rule.
2. Run the focused test before implementation and confirm failure.
3. Implement the smallest storage/API/UI change.
4. Run focused Python, Node, syntax, and diff checks.
5. Use an isolated SQLite copy for migrations/reindexing.
6. Back up production SQLite through the SQLite backup API before a formal
   reindex; verify identity, entity, relationship, and source-media counts.
7. Restart only Sentrix `8090`, then verify `4174` and untouched FMA `5173`.
