# Semantic Memory and Benchmark Integration Design

## Goal

Make household semantic memory person-centred and explainable, fix the person
evidence review flow, isolate the three benchmark albums as separate memory
spaces, and evaluate ingestion, face clustering, event grouping, and image
retrieval without leaking benchmark labels into memory.

## Design Alternatives

### A. Keep the flat claim projection

Continue storing one `SemanticClaim` for each person, event, dimension, and
value, then improve only the UI grouping. This is the smallest change, but it
does not make the person-to-event chain a first-class persisted object.

### B. Native person-event graph with pattern projection (selected)

Persist a first-class `PersonEventMemory` link for each confirmed person's
participation in an event, and derive cross-event `PersonPattern` records from
those links. Keep `SemanticClaim` as an evidence-backed compatibility detail.
This makes the person profile, event timeline, query retrieval, and evidence
navigation share one explicit chain while keeping SQLite authoritative.

### C. Add an external graph database

Project people, events, and claims into a graph service. This adds deployment,
privacy, and synchronization cost without improving the local evidence
contract, so it is rejected.

## Canonical Memory Chain

```text
MemorySpace / Household
  -> Asset
  -> Observation
  -> Event
  -> PersonEventMemory
  -> PersonPattern / SemanticClaim
  -> Evidence Asset or FaceInstance
```

`Asset` and `Observation` remain immutable evidence. `Event` remains the
episodic unit. A confirmed `Person` is the root of semantic memory; places,
activities, clothing, and relationships are person-related dimensions rather
than an unrelated flat entity list.

## Semantic Records

`PersonEventMemory` stores `scope_id`, `person_id`, `event_id`, role, event
activity, place, capture interval, co-person IDs, evidence IDs, confidence,
and revision state. It is rebuilt whenever identity confirmation, event
membership, or event summary changes.

`PersonPattern` stores `scope_id`, `person_id`, pattern type, canonical value,
support event IDs, evidence IDs, support count, first/last observed time,
confidence, and revision state. Examples include recurring activity, repeated
place, co-person, and normalized clothing attributes.

`SemanticClaim` remains available for compatibility and detailed review. It
must reference its supporting events and evidence, and its normalized value
must not discard the raw observation value.

The original observation caption is never overwritten after naming. Event
summaries, person-event memory, patterns, and Agent answers use confirmed
names when a confirmed identity is linked to the evidence.

## Person Review UI

The people page has separate actions:

- `查看证据`: opens avatar, face-cluster samples, linked events, and original
  image evidence;
- `确认人物`: opens the name and family-role form for pending candidates;
- confirmation success: shows the returned entity name, affected event count,
  profile/pattern count, and a refresh state before reloading the page.

The knowledge page separates confirmed people from non-person related entities
and displays a person profile as summary, event timeline, cross-event patterns,
appearance timeline, relationships, and evidence. Flat claims remain a
drill-down view.

## Benchmark Boundary

Each `album1`, `album2`, and `album3` becomes a separate `MemorySpace` and is
selectable in the web UI. Assets, observations, events, entities, vectors,
claims, and Agent retrieval are scope-filtered; an event cannot combine assets
from different spaces.

The input adapter reads only files present under each album's `images/`:

- metadata may be `metadata.json` at the album root or
  `metadata/metadata.json`;
- metadata and face mappings for missing image files are reported as
  unmatched and ignored for ingestion;
- missing metadata is preserved as unknown, never synthesized from the
  evaluation labels;
- `face_info_cn.json` and `face_info_en.json` names are evaluation-only;
- `query.json` ground-truth filenames are evaluation-only and never become
  event names, people, facts, or relationships.

The generated manifest contains allowed import metadata, scope identity,
unmatched-file diagnostics, face evaluation labels, and query evaluation
cases in separate sections. Only the import section is accepted by the memory
pipeline.

## Evaluation Contract

The benchmark runner reports separately:

- input intersection and missing-file diagnostics;
- face cluster pairwise precision, recall, F1, false merges, missed merges,
  and singleton ratio using external face IDs;
- scope isolation and event candidate diagnostics using capture time/location;
- query image retrieval precision, recall, hit rate, and per-query evidence
  trace against `ground_truth` filenames;
- post-confirmation propagation from a user-confirmed cluster into person,
  event, pattern, claim, and evidence records.

Benchmark names and answers are loaded only by the evaluator. They are never
used to auto-confirm a cluster or write a semantic fact.

## Non-goals

Video extraction remains behind the existing adapter boundary. The benchmark
phase evaluates images, metadata, face clustering, event grouping, semantic
projection, and retrieval only.
