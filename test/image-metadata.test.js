const test = require("node:test");
const assert = require("node:assert/strict");
const { parseTiff, parsePng } = require("../src/image-metadata.js");

function tiffFixture() {
  const buffer = new ArrayBuffer(320);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  const entry = (offset, tag, type, count, value) => {
    view.setUint16(offset, tag, true); view.setUint16(offset + 2, type, true); view.setUint32(offset + 4, count, true); view.setUint32(offset + 8, value, true);
  };
  text(0, "II"); view.setUint16(2, 42, true); view.setUint32(4, 8, true); view.setUint16(8, 2, true);
  entry(10, 0x8769, 4, 1, 100); entry(22, 0x8825, 4, 1, 150);
  view.setUint16(100, 1, true); entry(102, 0x9003, 2, 20, 120); text(120, "2025:05:20 20:30:00\0");
  view.setUint16(150, 4, true); entry(152, 1, 2, 2, 78); entry(164, 2, 5, 3, 210); entry(176, 3, 2, 2, 80); entry(188, 4, 5, 3, 234);
  text(78, "N\0"); text(80, "E\0");
  [[30, 1], [15, 1], [0, 1]].forEach(([n, d], i) => { view.setUint32(210 + i * 8, n, true); view.setUint32(214 + i * 8, d, true); });
  [[120, 1], [0, 1], [0, 1]].forEach(([n, d], i) => { view.setUint32(234 + i * 8, n, true); view.setUint32(238 + i * 8, d, true); });
  return buffer;
}

test("reads EXIF capture time and GPS from TIFF payload", () => {
  assert.deepEqual(parseTiff(tiffFixture()), {
    capturedAt: "2025-05-20T20:30:00",
    latitude: 30.25,
    longitude: 120,
  });
});

test("PNG without eXIf returns empty metadata", () => {
  const png = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 0]);
  assert.deepEqual(parsePng(png.buffer), {});
});
