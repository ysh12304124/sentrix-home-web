import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { compileScript, compileTemplate, parse } from "@vue/compiler-sfc";

test("the evaluation template only uses bindings defined by its script", () => {
  const filename = new URL("./src/App.vue", import.meta.url);
  const { descriptor, errors } = parse(readFileSync(filename, "utf8"), { filename: filename.pathname });
  assert.deepEqual(errors, []);
  const script = compileScript(descriptor, { id: "photobench-app" });
  const template = compileTemplate({
    source: descriptor.template.content,
    filename: filename.pathname,
    id: "photobench-app",
    compilerOptions: { bindingMetadata: script.bindings },
  });
  assert.deepEqual(template.errors, []);
  const missing = [...new Set([...template.code.matchAll(/_ctx\.([A-Za-z_$][\w$]*)/g)].map((match) => match[1]))].sort();
  assert.deepEqual(missing, [], `Undefined template bindings: ${missing.join(", ")}`);
});

test("merged graph and QA metrics remain visible without an extra expansion or graph snapshot", () => {
  const source = readFileSync(new URL("./src/App.vue", import.meta.url), "utf8");
  assert.match(source, /<details open class="phase-card result-phase-card aggregate-result-card">/);
  assert.match(source, /id="graph-quality-section"/);
  assert.match(source, /graphQuestionTypeRows\(\)\.length" class="graph-quality-type-wrap"/);
  for (const label of ["人脸身份聚类质量", "可验证边 P/R/F1", "按问题类型的图记忆 QA 表现", "图效果"]) {
    assert.ok(source.includes(label), `${label} panel is missing`);
  }
});

test("startup isolates run-list failures from dataset manifests and selects the latest completed report", () => {
  const source = readFileSync(new URL("./src/App.vue", import.meta.url), "utf8");
  assert.match(source, /Promise\.allSettled\(\[\s*loadJudgePrompts\(\),\s*api\("\/api\/manifests"\),\s*loadRuns\(1\)/);
  assert.match(source, /manifestResult\.status === "fulfilled"\) manifests\.value/);
  assert.match(source, /\["completed", "completed_with_errors"\]\.includes\(run\.status\)/);
});
