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
    ["scripts", "fixtures", "build_virtual_family_album.py"],
  ]) assert.equal(exists(...file), true, file.join("/") + " must exist");
  assert.match(fs.readFileSync(path.join(root, "index.html"), "utf8"), /href="\/src\/styles\.css"/);
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
});

test("virtual fixture keeps evaluation labels outside imported metadata", () => {
  const source = fs.readFileSync(path.join(root, "scripts", "fixtures", "build_virtual_family_album.py"), "utf8");
  const metadataSection = source.split('"sentrix_metadata.json"')[1];
  assert.ok(metadataSection);
  assert.doesNotMatch(metadataSection, /activity_hint|source_identity|family_member|photographer/);
});

test("native confirmation and assistant turn routes are exposed by the browser API", () => {
  const source = fs.readFileSync(path.join(root, "src", "api.js"), "utf8");
  assert.match(source, /\/api\/persons\/\$\{encodeURIComponent\(id\)\}\/confirm/);
  assert.match(source, /\/api\/assistant\/turn/);
  assert.match(source, /family_role/);
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

test("memory-space and evidence governance are wired into the portal", () => {
  const source = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
  assert.match(source, /space-select/);
  assert.match(source, /person-evidence/);
  assert.match(source, /cluster-split/);
  assert.match(source, /refresh_counts/);
});
