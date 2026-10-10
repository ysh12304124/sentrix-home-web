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
