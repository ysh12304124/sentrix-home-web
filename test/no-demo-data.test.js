const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");

test("portal does not contain the previous demonstration family records", () => {
  const source = fs.readFileSync("src/app.js", "utf8");
  for (const value of ["周末公园散步", "妈妈的生日", "春节回家", "宝宝第一次走路", "我们的第一个家", "38,426"]) {
    assert.equal(source.includes(value), false, `demo value remains: ${value}`);
  }
});
