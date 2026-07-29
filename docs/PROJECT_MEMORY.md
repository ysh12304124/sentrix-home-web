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
   evidence. Activity, place, clothing, attendance, and capture are multi-value
   claims; identity attributes remain versioned single-value claims.
4. Visual memory stores CLIP vectors, face embeddings, face clusters, and
   original-media pointers. Face clusters use `buffalo_l`, global re-clustering,
   and a medoid representative. A cluster is only a candidate until the user
   confirms it.

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

The current automated baseline is 23 Python tests and 4 Node tests. Before
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
its metadata with `scripts/prepare_test_metadata.py` before a clean rebuild.
