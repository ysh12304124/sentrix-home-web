const fs = require("fs");
const path = require("path");
const test = require("node:test");
const assert = require("node:assert/strict");

const root = path.resolve(__dirname, "..");
const exists = (...parts) => fs.existsSync(path.join(root, ...parts));

test("runtime and maintenance entry points use the documented layout", () => {
  for (const file of [
    ["src", "styles.css"],
    ["scripts", "runtime", "start_sentrix_ollama.sh"],
    ["scripts", "maintenance", "rebuild_memory.py"],
    ["scripts", "benchmarks", "evaluate_lfw_clusters.py"],
    ["scripts", "benchmarks", "ingest_household_face_benchmark.py"],
    ["scripts", "benchmarks", "evaluate_household_memory_steward.py"],
    ["scripts", "benchmarks", "evaluate_household_end_to_end.py"],
    ["scripts", "benchmarks", "evaluate_memory_steward.py"],
    ["scripts", "fixtures", "build_virtual_family_album.py"],
  ]) assert.equal(exists(...file), true, file.join("/") + " must exist");
  const apiStartScript = fs.readFileSync(path.join(root, "scripts", "runtime", "start_sentrix_api.sh"), "utf8");
  assert.match(apiStartScript, /ADAFACE_MODEL_PATH/, "API startup must configure the AdaFace checkpoint");
  assert.match(apiStartScript, /ADAFACE_REPO_ROOT/, "API startup must configure the AdaFace repository root");
  assert.match(apiStartScript, /OLLAMA_BASE_URL/, "API startup must use the Sentrix Ollama endpoint");
  assert.match(fs.readFileSync(path.join(root, "index.html"), "utf8"), /href="\/src\/styles\.css"/);
});

test("scene-type backfill is explicit and requires a SQLite backup", () => {
  const source = fs.readFileSync(path.join(root, "scripts", "maintenance", "backfill_scene_types.py"), "utf8");
  assert.match(source, /--apply requires --backup/);
  assert.match(source, /scene_type_backfill/);
  assert.match(source, /maintain_observation_entities/);
  assert.match(source, /--reproject-only/);
});

test("root directory contains only application entry points and project metadata", () => {
  for (const file of [
    "agent.py", "app.py", "db.py", "pipeline.py", "model_clients.py", "rebuild_memory.py",
    "api.js", "app.js", "normalizers.js", "styles.css", "architecture.md", "implementation-plan.md",
    "test-datasets.md", "download_test_data.py", "evaluate_lfw_clusters.py", "ingest_face_benchmark.py",
  ]) assert.equal(exists(file), false, file + " must not remain as a duplicate root entry point");
});

test("web gateway only proxies the authoritative Sentrix API", () => {
  const source = fs.readFileSync(path.join(root, "server.js"), "utf8");
  assert.doesNotMatch(source, /COGNEE_BASE_URL|mockSearch|function handleApi/);
  assert.match(source, /return proxyBackend\(req, res, url\);/);
  assert.match(source, /127\.0\.0\.1:8091/, "web must use the Agent-capable API by default");
  assert.match(source, /\/api\/model-profiles\/switch/);
  assert.match(source, /1_000_000/, "model switching must outlive the vLLM ready timeout");
});

test("settings exposes the four benchmark model profiles through the switch API", () => {
  const apiSource = fs.readFileSync(path.join(root, "src", "api.js"), "utf8");
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  for (const profile of ["gemma4-12b-it", "gemma4-e2b-it", "gemma4-e2b-it-lora-v2", "qwen3.5-0.8b-it"]) {
    assert.match(appSource, new RegExp(profile.replace(/[.]/g, "\\.")));
  }
  for (const label of ["Gemma-4-12B", "Gemma-4-E2B 蒸馏前", "Gemma-4-E2B 蒸馏后（加 LoRA 头）", "Qwen-3.5-0.8B"]) {
    assert.match(appSource, new RegExp(label.replace(/[()]/g, "\\$&")));
  }
  assert.match(apiSource, /getModelProfiles/);
  assert.match(apiSource, /switchModelProfile/);
  assert.match(apiSource, /\/api\/model-profiles\/switch/);
  assert.match(appSource, /switchModelProfile\(target\)/);
});

test("portal exposes all optimized album scopes and labels their evidence", () => {
  const source = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(source, /全部相册/);
  assert.match(source, /visibleSpaces/);
  assert.match(source, /albumLabel/);
  assert.match(source, /album-badge/);
  assert.match(source, /source_album_id/);
  assert.match(source, /assistant-scope.*albumLabel\(state\.scopeId\)/s);
});

test("virtual fixture keeps evaluation labels outside imported metadata", () => {
  const source = fs.readFileSync(path.join(root, "scripts", "fixtures", "build_virtual_family_album.py"), "utf8");
  const metadataSection = source.split('"sentrix_metadata.json"')[1];
  assert.ok(metadataSection);
  assert.doesNotMatch(metadataSection, /activity_hint|source_identity|family_member|photographer/);
});

test("native confirmation and assistant turn routes are exposed by the browser API", () => {
  const backendSource = fs.readFileSync(path.join(root, "backend", "app.py"), "utf8");
  const source = fs.readFileSync(path.join(root, "src", "api.js"), "utf8");
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(source, /\/api\/persons\/\$\{encodeURIComponent\(id\)\}\/confirm/);
  assert.match(source, /\/api\/assistant\/turn/);
  assert.match(source, /family_role/);
  assert.match(backendSource, /toolTrace/);
  assert.match(appSource, /本轮判断与工具/);
  assert.match(appSource, /toolTrace/);
  assert.match(source, /entityGroups/);
  assert.match(backendSource, /\/api\/entity-groups/);
  assert.match(backendSource, /evidencePresentation/);
  assert.match(backendSource, /originalEvidenceRequested/);
});

test("family memory assistant presents an evidence-backed conversation surface", () => {
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(appSource, /家庭记忆助手/);
  assert.match(appSource, /assistantMessages/);
  assert.match(appSource, /assistant-message/);
  assert.match(appSource, /clarification_candidates/);
  assert.match(appSource, /continue-assistant/);
  assert.match(appSource, /evidence_order/);
  assert.match(appSource, /证据顺序与可信度/);
  assert.match(appSource, /dialogue_plan/);
});

test("memory answers always expose evidence state and direct original media", () => {
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(appSource, /evidence_presentation/);
  assert.match(appSource, /original_evidence_requested/);
  assert.match(appSource, /证据缺口/);
  assert.match(appSource, /直接查看原始证据/);
});

test("assistant renders claim-level evidence through stable ids and segments", () => {
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  const backendSource = fs.readFileSync(path.join(root, "backend", "app.py"), "utf8");
  assert.match(appSource, /assistantAnswer/);
  assert.match(appSource, /claimEvidence/);
  assert.match(appSource, /claim_evidence_index/);
  assert.match(appSource, /data-claim-id/);
  assert.match(appSource, /逐句查看依据/);
  assert.match(backendSource, /claimVerifications/);
  assert.match(backendSource, /claimEvidenceIndex/);
  assert.match(backendSource, /segments/);
});

test("proactive recall is gated, viewer-scoped, and user-dismissible", () => {
  const agentSource = fs.readFileSync(path.join(root, "backend", "agent.py"), "utf8");
  const backendSource = fs.readFileSync(path.join(root, "backend", "app.py"), "utf8");
  const apiSource = fs.readFileSync(path.join(root, "src", "api.js"), "utf8");
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(agentSource, /SENTRIX_PROACTIVE_MEMORY/);
  assert.match(agentSource, /proactivity_sensitive/);
  assert.match(agentSource, /record_proactivity_outcome/);
  assert.match(agentSource, /memory_intensity=\"probe\"/);
  assert.match(backendSource, /viewer_id/);
  assert.match(apiSource, /viewer_id/);
  assert.match(appSource, /accept-proactive/);
  assert.match(appSource, /disable-proactive/);
});

test("entity property corrections are exposed with evidence-aware UI controls", () => {
  const apiSource = fs.readFileSync(path.join(root, "src", "api.js"), "utf8");
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(apiSource, /setEntityProperty/);
  assert.match(apiSource, /\/api\/entities\/\$\{encodeURIComponent\(id\)\}\/properties\/\$\{encodeURIComponent\(propertyKey\)\}/);
  assert.match(appSource, /entity-property-edit/);
  assert.match(appSource, /edit-entity-properties/);
  assert.match(appSource, /property_history/);
  assert.match(appSource, /evidence_ids/);
});

test("event detail exposes its evidence-backed entity projection", () => {
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(appSource, /eventEntities/);
  assert.match(appSource, /事件实体/);
  assert.match(appSource, /event-entity-row/);
});

test("person profiles expose user-maintained identity properties", () => {
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(appSource, /person-property-edit/);
  assert.match(appSource, /edit-person-properties/);
  assert.match(appSource, /relation_to_user/);
  assert.match(appSource, /is_self/);
  assert.match(appSource, /groups/);
});

test("relationship candidates expose evidence counts before confirmation", () => {
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(appSource, /关系候选与证据/);
  assert.match(appSource, /evidence_ids_json/);
  assert.match(appSource, /确认关系/);
});

test("pending trip candidates are loaded as evidence-backed semantic memory", () => {
  const apiSource = fs.readFileSync(path.join(root, "src", "api.js"), "utf8");
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(apiSource, /trips:/);
  assert.match(appSource, /行程候选/);
  assert.match(appSource, /trip-candidate/);
  assert.match(appSource, /evidence_ids_json/);
});

test("trip candidates have explicit user confirmation and rejection actions", () => {
  const apiSource = fs.readFileSync(path.join(root, "src", "api.js"), "utf8");
  const appSource = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(apiSource, /confirmTrip/);
  assert.match(apiSource, /rejectTrip/);
  assert.match(appSource, /confirm-trip/);
  assert.match(appSource, /reject-trip/);
});

test("event edit exposes type, end time, and evidence-backed cover controls", () => {
  const source = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(source, /name="event_type"/);
  assert.match(source, /name="time_end"/);
  assert.match(source, /name="cover_asset_id"/);
  assert.match(source, /封面选择依据/);
  assert.match(source, /cover_selection/);
});

test("entity reindex maintenance isolates its SQLite connection and rejects concurrent runs", () => {
  const source = fs.readFileSync(path.join(root, "backend", "app.py"), "utf8");
  assert.match(source, /maintenance_lock/);
  assert.match(source, /MemoryStore\(store\.path\)/);
  assert.match(source, /status_code=409/);
});

test("memory-space and evidence governance are wired into the portal", () => {
  const source = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(source, /space-select/);
  assert.match(source, /person-evidence/);
  assert.match(source, /cluster-split/);
  assert.match(source, /refresh_counts/);
});

test("semantic directory is semantic-first and evidence-backed", () => {
  const source = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(source, /semantic-group-card/);
  assert.match(source, /语义摘要/);
  assert.match(source, /细节语义/);
  assert.match(source, /原始证据/);
  assert.match(source, /semantic_details/);
  assert.match(source, /technical-evidence/);
  assert.match(source, /data-action="open-asset"/);
});

test("default evidence tiles do not expose filenames or internal identifiers", () => {
  const source = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  const imageResults = source.slice(source.indexOf("function imageResults"), source.indexOf("function traceLabel"));
  assert.doesNotMatch(imageResults, /item\.file_name/);
  assert.match(source, /技术信息/);
  assert.match(source, /technical-evidence/);
});

test("image imports automatically schedule event summarization after semantic enrichment", () => {
  const backendSource = fs.readFileSync(path.join(root, "backend", "app.py"), "utf8");
  assert.match(backendSource, /enrich_fast_image\(asset_id, summarize_event=True\)/);
});
