# Sentrix Agent 2.0 Design

## Goal

Improve generalization for evidence-grounded family-memory tasks without
encoding a growing list of question types in deterministic routing rules.
The system must handle novel wording and new combinations of known evidence
needs while preserving Sentrix's local-data, MemorySpace, provenance, budget,
and authorization constraints.

The target is not a shorter prompt. The target is fewer unconstrained model
decisions and a runtime that makes every accepted decision executable,
auditable, budgeted, and evidence-bound.

## Non-goals

- Do not replace the production Tool-Loop with a fixed question-type router.
- Do not train or tune against PhotoBench answer labels in production memory.
- Do not make an arbitrary SQL endpoint available to the planner.
- Do not grant direct memory-write authority to the model.
- Do not claim a benchmark gain as generalization without held-out paraphrase,
  composition, and failure-path evidence.

## Current Problem

The current runtime has a useful Tool Registry, budget manager, ResultSet,
answer nucleus, Final Guard, capability matrix, and CompletionState. However,
CompletionState derives mandatory next steps from overlapping keyword signals.
For example, a text/sign request may activate both `inspect_photo` and
`read_photo_text`, while both are expensive and the normal `tool_loop` profile
permits one inspection. The failure is not insufficient prompt detail; it is a
workflow conflict encoded outside an explicit evidence state.

## Architecture

```text
User message
  -> LLM Goal Interpreter
  -> TaskState(goal, evidence requirements, constraints, open references)
  -> LLM Planner chooses one capability and arguments
  -> Capability Runtime validates and executes the capability
  -> EvidenceLedger appends typed, scoped provenance
  -> TaskState update
  -> Planner: continue, clarify, or final
  -> Grounded Final (nucleus + guards + writer)
```

The planner remains open-ended. It may decompose a novel question into
requirements and combine capabilities. The runtime does not decide that a
particular user phrase must use a particular capability.

### TaskState

The Goal Interpreter returns a schema-validated declaration, not a tool call:

```json
{
  "goal": "determine the steak price at the referenced restaurant",
  "constraints": {"scope_id": "album1", "time": "2025"},
  "requirements": [
    {"id": "restaurant", "evidence_type": "memory_reference", "status": "open"},
    {"id": "steak_price", "evidence_type": "visible_text", "status": "open"}
  ]
}
```

Requirements are model-proposed and use a closed vocabulary of evidence
types. The runtime validates schema, scope and state transitions, but does not
infer a user intent from keywords. A requirement can be `open`, `running`,
`satisfied`, `unsupported`, `blocked_budget`, or `ambiguous`.

Initial evidence types are:

- `structured_fact`
- `memory_asset`
- `memory_reference`
- `visual_observation`
- `visible_text`
- `temporal_metadata`
- `location_metadata`
- `confirmed_identity`
- `user_statement`
- `transcript`

Adding an evidence type requires a contract test and benchmark coverage; it is
not added for one user question.

### Capability Contract

Extend the existing `ToolSpec`; do not add a parallel registry. Each capability
declares:

- stable identity and version;
- typed input and output schema;
- evidence types it can produce and cannot establish;
- preconditions, scope requirement, risk, authorization and readiness;
- cost vector: model calls, images or media seconds, tokens, wall time;
- deterministic budget unit and result limits;
- evidence provenance fields and partial/failure semantics;
- allowed state transitions and whether the output can satisfy a requirement.

Examples:

- `read_photo_text` can establish `visible_text`, returning exact raw text,
  text regions, confidence, source/provider and partial coverage. It cannot
  establish confirmed identity or an unseen object.
- `inspect_photo` can establish `visual_observation`, returning scoped visible
  attributes and confidence. It cannot establish exact OCR text or confirmed
  identity.
- `search_memories` can establish `memory_asset` or retrieval support, but not
  unseen fine-grained visual claims.

Text data is preserved as `raw`, optional `normalized`, and optional
`translation`; derived forms never replace raw evidence. Region coordinates and
provider confidence stay attached to the raw text.

### EvidenceLedger and Completion

Every tool result is recorded as an immutable ledger entry with call id, scope,
input references, evidence type, asset or observation provenance, coverage,
certainty, and failure reason.

Completion is requirement-driven:

- The runtime marks a requirement satisfied only when a capability produces a
  compatible evidence type with valid provenance.
- The planner may request multiple compatible capabilities when the question
  genuinely needs them. `visual_observation` and `visible_text` are therefore
  composable, not globally exclusive.
- The runtime prevents duplicate calls, nonexistent handles, invalid schema,
  budget overruns, cross-space reads and unauthorized writes.
- If requirements are ambiguous, unsupported or blocked by budget, the planner
  emits a structured clarify or partial terminal state. A model cannot silently
  convert missing evidence into a fact.

`clarify` is a typed terminal action containing `missing_slots`, candidate
references from evidence, and a code-templated or model-written question. It
does not contain arbitrary entity identifiers.

### Result Delivery

The planner decides selection intent, for example `representative`, `all`, or
`specific_assets`. ResultSet/API/UI own pagination and transfer mechanics.
An all-results request returns a result-set reference, total, access state and
delivery mode to the UI; the model never burns its reasoning budget retrieving
HTTP pages one at a time.

### Bounded Batch Capabilities

Batch visual or OCR work is introduced only through a general batch contract.
It has limits on items, media tokens, wall time and cost. Its response contains
per-item provenance and a coverage summary:

```json
{
  "items": [{"handle": "photo_1", "status": "supported", "evidence": []}],
  "coverage": {"requested": 6, "processed": 4, "skipped_budget": 2}
}
```

The final answer may make a population-level claim only when coverage is
sufficient for that claim; otherwise it must name the uninspected remainder.

### Query Capability Evolution

`query_memory_facts` remains the deterministic executor for known structured
facts. Its expansion path is a restricted, versioned Memory Query IR, not
model-generated SQL. Version 1 supports a typed source, filters, `And`, `Not`,
grouping, count, ordering, `TopK`, and person co-occurrence. A compiler maps
only validated IR nodes to scoped queries and returns evidence rows.

## Rollout

### Phase 0: Instrument and Compare

Keep the production Tool-Loop unchanged. Add trace fields that classify planned
requirements, chosen capability, accepted or rejected transitions, evidence
coverage, budget outcome and terminal reason. Run the candidate architecture in
shadow mode against the existing PhotoBench execution path. No production
answer or memory mutation changes in this phase.

### Phase 1: Minimal Goal and Evidence Slice

Implement typed TaskState, EvidenceLedger adapters for existing tools, and
schema-validated planner output. Exercise one retrieval-plus-verification
workflow without keyword-mandatory CompletionState rules. The candidate must
fall back to the current runtime on malformed plans, unsupported requirements,
or unavailable capabilities.

### Phase 2: Contract Consolidation

Extend `ToolSpec`, derive model-visible capability summaries and profile
filtering from it, and migrate CompletionState to requirement compatibility.
Introduce structured clarify and ResultSet delivery intent. Remove a legacy
rule only after the replacement has shadow evidence and regression coverage.

### Phase 3: General Composition

Add bounded batch capabilities and Memory Query IR v1. These features stay
behind profiles until they improve held-out composition tasks without exceeding
latency, budget-exhaustion or unsupported-claim limits.

### Phase 4: New Modalities and Controlled Mutation

Add video/audio evidence capabilities through the same contracts. Add only
`propose_write -> user confirmation -> commit_write` transitions, with audit,
revisions and no direct model writes.

## Evaluation and Gates

Use the current PhotoBench as the principal comparable baseline. Preserve its
current cases and traces, then partition evaluation so design examples never
become the only acceptance data:

- existing benchmark: regression, score, image delivery and trace comparison;
- paraphrase holdout: same latent task expressed with unseen wording;
- compositional holdout: multiple evidence types, negation, references and
  multi-step aggregation;
- failure injection: missing handles, zero retrieval, partial OCR, capability
  timeout, exhausted budget and malformed planner output;
- adversarial provenance checks: cross-space references, unsupported exact
  values and attempts to answer beyond coverage.

Report, per model profile and task stratum:

- task success and clarify success;
- capability-selection precision/recall and confusion matrix;
- unsupported-claim rate, hard-value consistency and evidence coverage;
- model decisions, model steps, tool calls, expensive budget use and P95 latency;
- parse failures, timeout rate, recovery rate and budget exhaustion rate.

`LLM decisions per successful task` is a primary efficiency metric. A lower
number is an improvement only when held-out task success and grounding are not
worse. A phase may advance only when it has no new hard grounding or scope
violation, no regression on existing benchmark gates, and a pre-declared gain
on held-out generalization measures.

## Safety and Migration Rules

- Preserve Asset -> Observation provenance and MemorySpace isolation.
- Never copy benchmark labels, answers, or identities into production memory.
- Keep current Tool-Loop as an explicit fallback through Phase 2.
- Treat a planner declaration as untrusted input; schema validation and runtime
  authorization remain deterministic.
- Do not compile a Fast Path until traces show the same plan is stable across
  independent paraphrases and compositions. Compiled paths are optimizations,
  not the source of semantics.
