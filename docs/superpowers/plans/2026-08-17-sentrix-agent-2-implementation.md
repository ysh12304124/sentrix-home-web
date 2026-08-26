# Sentrix Agent 2.0 Implementation Plan

## Goal

Evolve the production AgentRuntime from a free Tool-Loop with keyword-derived
completion rules into an open Planner with typed evidence requirements and a
capability-constrained runtime. Improve generalization for unfamiliar wording
and novel compositions without implementing a deterministic question-type
router or allowing the model to bypass scope, authorization, budget, or
provenance rules.

## Architecture

The planner emits a schema-validated `TaskDeclaration` and subsequent typed
actions. `TaskState` tracks model-proposed goals and requirements. Existing
`ToolSpec` becomes the sole capability contract. `EvidenceLedger` records tool
outputs with provenance and coverage. Runtime code validates state transitions,
inputs, permissions, budgets and evidence compatibility; it never maps a user
phrase directly to a mandatory tool. Existing Tool-Loop remains the production
fallback through Phase 2.

## Technology

- Python 3, FastAPI, SQLite, `unittest`
- Existing `backend/agent_runtime/` Tool Registry, Tool Policy, ResultSet,
  Final Guard, answer nucleus, profiles and traces
- Existing PhotoBench orchestrator and its persisted runtime traces
- Node test runner for web/API contracts

## Delivery Rules

- Work only in `/Users/rm001/Sentrix-Home-Web-psh-work` until the user grants
  separate authorization for any 153 transfer, commit, service change or test.
- Never copy PhotoBench prompts, answers, identity labels or event labels into
  production memory or runtime prompts.
- Keep MemorySpace isolation, immutable Asset -> Observation provenance and
  `memory_write=false` unchanged through Phase 3.
- Do not remove a legacy completion rule until the replacement has passed its
  shadow gate and focused regression tests.

## Phase 0: Baseline and Trace Contract

### Task 1: Add stable Agent 2 trace types

Files:
- Create `backend/agent_runtime/task_state.py`
- Create `backend/agent_runtime/evidence_ledger.py`
- Create `backend/tests/test_agent_task_state.py`
- Create `backend/tests/test_agent_evidence_ledger.py`

Steps:
1. Define immutable dataclasses and JSON serializers for `EvidenceRequirement`,
   `TaskDeclaration`, `TaskState`, `LedgerEntry`, and `Coverage`.
2. Limit requirement evidence types to the vocabulary in the approved design:
   `structured_fact`, `memory_asset`, `memory_reference`,
   `visual_observation`, `visible_text`, `temporal_metadata`,
   `location_metadata`, `confirmed_identity`, `user_statement`, `transcript`.
3. Make every state transition explicit: `open -> running -> satisfied`, or
   `open/running -> ambiguous|unsupported|blocked_budget`; reject all others.
4. Require a ledger entry to include tool call id, scope id, capability name,
   input references, evidence type, provenance references, certainty, coverage,
   and failure reason when not supported.
5. Write unit tests for invalid evidence types, cross-scope references,
   duplicate call ids, invalid transitions, partial coverage and JSON round
   trips.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_agent_task_state backend.tests.test_agent_evidence_ledger -v
```

Expected result: all new tests pass; no database, network or model dependency.

### Task 2: Preserve current behavior while emitting shadow fields

Files:
- Modify `backend/agent_runtime/runtime.py`
- Modify `backend/app.py`
- Modify `backend/tests/test_tool_loop_truth_contract.py`
- Create `backend/tests/test_agent_runtime_trace_contract.py`

Steps:
1. Add optional, empty-by-default Agent 2 trace fields to `RuntimeTurn` and API
   payloads: `task_declaration`, `task_state`, `evidence_ledger`,
   `planner_decisions`, `terminal_reason` and `budget_outcome`.
2. Populate these fields only when the shadow profile is selected; preserve the
   existing production `tool_loop` response shape and answer path.
3. Verify that internal evidence references are redacted from non-debug output
   exactly as existing traces are.
4. Add regression tests that existing traces and FinalGuard behavior are
   unchanged when Agent 2 is disabled, and serialization remains valid when it
   is enabled with an empty declaration.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_tool_loop_truth_contract backend.tests.test_agent_runtime_trace_contract -v
```

Expected result: all focused tests pass and legacy runtime API tests retain
their current assertions.

### Task 3: Extend PhotoBench trace extraction without changing scoring

Files:
- Modify `services/photobench/backend/benchmark_orchestrator.py`
- Create `services/photobench/backend/test_trace_contract.py`
- Modify `services/photobench/README.md`

Steps:
1. Extract and persist Agent 2 trace fields when present in an assistant turn.
2. Attach planner decision count, requirement status counts, evidence coverage,
   blocked reasons and budget outcome to each runtime turn without overwriting
   historical runs missing those fields.
3. Include aggregate summaries by model and task category while preserving
   existing TTFT, token, latency and tool-trace measurements.
4. Test old fixture runs, new trace runs and absent optional fields.

Commands:
```bash
PYTHONPATH=. python3 -m unittest services.photobench.backend.test_trace_contract -v
```

Expected result: all trace fixtures pass; no historical result data is deleted
or rewritten.

### Task 4: Create a generalization-aware comparison dataset

Files:
- Create `scripts/benchmarks/agent2_shadow_cases.json`
- Create `scripts/benchmarks/agent2_gate_config.json`
- Create `scripts/benchmarks/evaluate_agent2_shadow.py`
- Create `scripts/benchmarks/summarize_agent2_eval.py`
- Create `backend/tests/test_agent2_benchmark_contract.py`

Steps:
1. Reference existing PhotoBench case ids; do not duplicate their answer labels
   into runtime inputs.
2. Add separate strata for baseline regression, paraphrase holdout,
   composition holdout, failure injection and provenance adversarial cases.
3. Store expected evidence requirements and permitted terminal outcomes only in
   the benchmark fixture, never in production code.
4. Make the evaluator run legacy and shadow profiles on identical inputs and
   emit per-case trace-derived metrics.
5. Define gates in JSON before each comparison: zero new scope, provenance or
   hard-value violations; no more than 1 percentage point regression in the
   existing benchmark score; non-negative held-out grounding delta; no increase
   in tool-call-limit rate; P95 latency no greater than 1.10 times baseline.
6. Test that the evaluator refuses a missing stratum, missing baseline, or a
   fixture that attempts to pass expected answers to the runtime.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_agent2_benchmark_contract -v
python3 scripts/benchmarks/evaluate_agent2_shadow.py --help
python3 scripts/benchmarks/summarize_agent2_eval.py --help
```

Expected result: fixture contract tests pass and both CLI tools print usage
without requiring a model service.

## Phase 1: Open Planner and Evidence Requirements Shadow Slice

### Task 5: Introduce schema-validated planner declarations

Files:
- Create `backend/agent_runtime/planner_contracts.py`
- Create `backend/agent_runtime/goal_planner.py`
- Modify `backend/agent_runtime/runtime.py`
- Create `backend/tests/test_goal_planner.py`
- Create `backend/tests/test_planner_contracts.py`

Steps:
1. Define strict JSON schemas for `declare`, `tool_call`, `clarify` and `final`.
2. Require `declare` to contain a goal, optional scoped constraints and one or
   more typed evidence requirements; prohibit raw SQL, untrusted asset ids and
   arbitrary evidence type strings.
3. Implement parser repair only for syntactic JSON recovery; reject semantic
   schema violations into the legacy Tool-Loop fallback.
4. Add a shadow-only planner turn before the current first tool decision. The
   planner proposes requirements but cannot execute a capability itself.
5. Test novel paraphrases, compositional requirements, malformed JSON,
   unsupported evidence types and prompt-injected capability names using a
   deterministic fake `chat_fn`.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_planner_contracts backend.tests.test_goal_planner -v
```

Expected result: all valid declarations parse, invalid declarations cannot
reach a tool executor, and fallback is observable in the trace.

### Task 6: Extend ToolSpec into the single capability contract

Files:
- Modify `backend/agent_runtime/tool_registry.py`
- Modify `backend/agent_runtime/tools.py`
- Modify `backend/agent_runtime/tool_policy.py`
- Modify `backend/agent_runtime/capability.py`
- Create `backend/tests/test_tool_capability_contract.py`

Steps:
1. Add typed produced evidence, prohibited claims, preconditions, cost vector,
   budget unit, output coverage schema and state-transition declarations to
   `ToolSpec`.
2. Declare `read_photo_text -> visible_text`, `inspect_photo ->
   visual_observation`, `search_memories -> memory_asset|memory_reference`,
   and structured tools' corresponding metadata or fact evidence.
3. Remove text-reading claims from the `inspect_photo` model-visible
   description; do not hard-code a user phrase to either capability.
4. Make ToolPolicy validate arguments, scope, budget and TaskState compatibility
   before execution, then emit a ledger entry from the normalized tool result.
5. Keep the current capability matrix as measured readiness data, not an
   alternative contract source.
6. Test every registered capability has a complete contract, cannot satisfy an
   incompatible requirement, and reports partial coverage faithfully.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_tool_capability_contract backend.tests.test_tool_loop_truth_contract -v
```

Expected result: all registered tools validate and existing truth contracts
remain unchanged.

### Task 7: Replace keyword-mandatory completion only in shadow mode

Files:
- Modify `backend/agent_runtime/completion.py`
- Modify `backend/agent_runtime/runtime.py`
- Modify `backend/agent_runtime/intent.py`
- Create `backend/tests/test_requirement_completion.py`
- Create `backend/tests/test_agent2_shadow_runtime.py`

Steps:
1. Preserve current `CompletionState` for legacy profiles.
2. Add a requirement-driven completion implementation for the shadow profile:
   it checks evidence compatibility and state, not user-message regex matches.
3. Permit the planner to request both visual and text evidence when its declared
   requirements need both; reject only duplicate, invalid or budget-infeasible
   transitions.
4. Return `clarify` with validated candidate references when state is ambiguous;
   return a partial terminal reason when a requirement is unsupported or budget
   blocked.
5. Test text-only, visual-only, mixed text-plus-visual, zero-retrieval,
   malformed-plan and budget-exhausted cases without asserting specific user
   keyword paths.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_requirement_completion backend.tests.test_agent2_shadow_runtime -v
```

Expected result: the shadow runtime produces no simultaneous mandatory actions
from overlapping text/visual wording and does not regress legacy profiles.

### Task 8: Run local shadow comparison and gate review

Files:
- Generated under `services/photobench/results/` only
- Generated under `docs/baseline/` only after review

Steps:
1. Start only the local validated stack and the configured vLLM Manager path;
   do not use or alter 153 services.
2. Execute legacy and `goal_driven_shadow` profiles against the exact same
   PhotoBench scope and case manifest.
3. Run all benchmark strata, produce JSON reports and compare them with
   `agent2_gate_config.json`.
4. Inspect every safety violation, planner fallback, budget-exhaustion increase
   and held-out degradation before considering Phase 2.
5. Save raw traces, summary and gate decision with model id, commit id, config
   hash and timestamp.

Commands:
```bash
python3 scripts/benchmarks/evaluate_agent2_shadow.py --profile legacy --out services/photobench/results/agent2-legacy.json
python3 scripts/benchmarks/evaluate_agent2_shadow.py --profile goal_driven_shadow --out services/photobench/results/agent2-shadow.json
python3 scripts/benchmarks/summarize_agent2_eval.py --baseline services/photobench/results/agent2-legacy.json --candidate services/photobench/results/agent2-shadow.json --gates scripts/benchmarks/agent2_gate_config.json
```

Expected result: a nonzero exit status on any gate failure and a machine-readable
pass/fail report. This task requires a local model/service environment; it does
not authorize 153 testing.

## Phase 2: Controlled Integration

### Task 9: Add profile-gated candidate execution and ResultSet delivery intent

Files:
- Modify `backend/agent_runtime/profile.py`
- Modify `backend/agent_runtime/runtime.py`
- Modify `backend/agent_runtime/result_set.py`
- Modify `backend/app.py`
- Modify `src/api.js`
- Modify `src/app.js`
- Create `backend/tests/test_result_delivery_intent.py`
- Create `test/agent-result-delivery.test.js`

Steps:
1. Add an opt-in `goal_driven_candidate` profile; leave `tool_loop` default.
2. Add a planner selection intent (`representative`, `all`, `specific_assets`)
   to ResultSet output. Keep pagination and original-media URLs in API/UI code.
3. Make the UI render all-results delivery from ResultSet metadata without
   requiring the model to call every page.
4. Test API response compatibility, result-set scope isolation, no arbitrary
   asset disclosure, and UI handling of incomplete delivery coverage.
5. Repeat the complete local gate comparison before enabling the profile for any
   non-shadow traffic.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_result_delivery_intent -v
node --test test/agent-result-delivery.test.js
```

Expected result: focused Python and Node tests pass; default profile behavior
does not change.

### Task 10: Generate model-visible descriptions and documentation from ToolSpec

Files:
- Create `backend/agent_runtime/tool_manifest.py`
- Modify `backend/agent_runtime/runtime.py`
- Modify `backend/app.py`
- Create `scripts/maintenance/export_tool_manifest.py`
- Create `backend/tests/test_tool_manifest.py`
- Modify `docs/PROJECT_MEMORY.md`

Steps:
1. Generate the compact planner-visible capability summary from ToolSpec.
2. Expose a versioned manifest through health/debug diagnostics without leaking
   private asset data.
3. Export a checked-in documentation snapshot that contains capabilities and
   contracts but no benchmark labels or runtime secrets.
4. Test that every profile's visible tools are a filtered view of the registry,
   and that missing contract fields fail CI.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_tool_manifest -v
python3 scripts/maintenance/export_tool_manifest.py --check
```

Expected result: manifest check is deterministic and profile/tool count drift
causes a failing test.

## Phase 3: General Composition

### Task 11: Implement bounded batch evidence capability

Files:
- Create `backend/agent_runtime/batch_verification.py`
- Modify `backend/agent_runtime/tools.py`
- Modify `backend/agent_runtime/tool_policy.py`
- Modify `backend/agent_runtime/runtime.py`
- Create `backend/tests/test_batch_verification.py`
- Modify `scripts/benchmarks/agent2_shadow_cases.json`

Steps:
1. Implement a generic bounded batch adapter with item, media-token and wall
   time limits from ToolSpec.
2. Return per-item evidence and coverage (`requested`, `processed`,
   `skipped_budget`, `failed`) for every batch result.
3. Allow population-level final claims only when ledger coverage supports them.
4. Add composition holdout cases with varying item order and paraphrases; do
   not add production rules keyed to their wording.
5. Gate the capability behind the candidate profile until held-out and failure
   strata pass.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_batch_verification -v
python3 scripts/benchmarks/evaluate_agent2_shadow.py --profile goal_driven_shadow --stratum composition --out services/photobench/results/agent2-composition.json
```

Expected result: partial coverage never becomes a universal claim and the
composition report satisfies the predeclared gates.

### Task 12: Implement Memory Query IR v1

Files:
- Create `backend/memory_query_ir.py`
- Create `backend/memory_query_compiler.py`
- Modify `backend/agent_runtime/tools.py`
- Modify `backend/db.py`
- Create `backend/tests/test_memory_query_ir.py`
- Create `backend/tests/test_memory_query_compiler.py`
- Modify `scripts/benchmarks/agent2_shadow_cases.json`

Steps:
1. Define a versioned AST with typed source, filter, `And`, `Not`, `GroupBy`,
   `Count`, `OrderBy`, `TopK`, and person co-occurrence nodes.
2. Validate node types, scope, operands, limits and allowed joins before any SQL
   compilation.
3. Compile only to parameterized, scoped SQLite queries and return source rows
   needed for the EvidenceLedger.
4. Keep existing `query_memory_facts` operations as compatibility adapters;
   migrate no operation until equivalence tests pass.
5. Test injection attempts, cross-space predicates, unsupported joins,
   deterministic ordering and provenance rows.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_memory_query_ir backend.tests.test_memory_query_compiler -v
```

Expected result: all invalid ASTs fail before database execution and equivalent
legacy queries preserve their answer facts.

## Phase 4: New Modalities and Controlled Mutation

### Task 13: Add media capabilities through the same contracts

Files:
- Create `backend/agent_runtime/media_verification.py`
- Modify `backend/agent_runtime/tools.py`
- Modify `backend/agent_runtime/tool_registry.py`
- Create `backend/tests/test_media_verification_contract.py`

Steps:
1. Add transcript and bounded video/audio segment capabilities only after their
   input region, time range, coverage and evidence contracts are defined.
2. Reuse TaskState, EvidenceLedger, budgets and final grounding checks.
3. Add holdout and failure cases before exposing a capability in a profile.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_media_verification_contract -v
```

Expected result: media capabilities cannot claim evidence outside an inspected
segment and never bypass scope checks.

### Task 14: Introduce proposal-only writes

Files:
- Create `backend/agent_runtime/write_proposals.py`
- Modify `backend/agent_runtime/tool_registry.py`
- Modify `backend/app.py`
- Modify `backend/db.py`
- Create `backend/tests/test_write_proposals.py`

Steps:
1. Define proposal, preview, explicit user confirmation, commit and revision
   audit states; do not register a direct write capability.
2. Require an existing scoped evidence reference and a user confirmation token
   for every commit.
3. Preserve immutable observations and create revision records for projections.
4. Test rejected confirmation, expired tokens, cross-space references, repeated
   commits and audit reconstruction.

Commands:
```bash
PYTHONPATH=. python3 -m unittest backend.tests.test_write_proposals -v
```

Expected result: no model output alone can mutate family memory.

## Phase Advancement and 153 Handoff

Before each phase transition, record local focused test results and the full
local PhotoBench gate report in Project Memory. Do not transfer any source,
fixture, generated result, database, log, `.env`, model weight or backup to
153 without explicit user authorization. After authorization, compare both Git
statuses, transfer only reviewed source/test/document files, rerun the approved
tests on 153, and commit only on 153 `psh`.
