(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.sentrixImageMetadata = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  const MAX_METADATA_BYTES = 4 * 1024 * 1024;

  function ascii(view, offset, length) {
    let value = "";
    for (let index = 0; index < length && offset + index < view.byteLength; index += 1) {
      const code = view.getUint8(offset + index);
      if (!code) break;
      value += String.fromCharCode(code);
    }
    return value.trim();
  }

  function parseTiff(buffer, start = 0, length = buffer.byteLength - start) {
    const view = new DataView(buffer, start, length);
    if (view.byteLength < 8) return {};
    const marker = ascii(view, 0, 2);
    const little = marker === "II";
    if (!little && marker !== "MM") return {};
    if (view.getUint16(2, little) !== 42) return {};

    const u16 = (offset) => view.getUint16(offset, little);
    const u32 = (offset) => view.getUint32(offset, little);
    const typeSize = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 };

    function entryValue(type, count, entryOffset) {
      const bytes = (typeSize[type] || 0) * count;
      if (!bytes) return null;
      const valueOffset = bytes <= 4 ? entryOffset + 8 : u32(entryOffset + 8);
      if (valueOffset < 0 || valueOffset + bytes > view.byteLength) return null;
      if (type === 2) return ascii(view, valueOffset, count);
      if (type === 3) return count === 1 ? u16(valueOffset) : Array.from({ length: count }, (_, index) => u16(valueOffset + index * 2));
      if (type === 4) return count === 1 ? u32(valueOffset) : Array.from({ length: count }, (_, index) => u32(valueOffset + index * 4));
      if (type === 5) return Array.from({ length: count }, (_, index) => {
        const denominator = u32(valueOffset + index * 8 + 4);
        return denominator ? u32(valueOffset + index * 8) / denominator : 0;
      });
      return null;
    }

    function readIfd(offset) {
      const entries = new Map();
      if (!Number.isInteger(offset) || offset < 0 || offset + 2 > view.byteLength) return entries;
      const count = u16(offset);
      for (let index = 0; index < count; index += 1) {
        const entryOffset = offset + 2 + index * 12;
        if (entryOffset + 12 > view.byteLength) break;
        const tag = u16(entryOffset);
        entries.set(tag, entryValue(u16(entryOffset + 2), u32(entryOffset + 4), entryOffset));
      }
      return entries;
    }

    function normalizeTime(value, offset) {
      const match = String(value || "").match(/^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})/);
      if (!match) return null;
      const suffix = /^[+-]\d{2}:\d{2}$/.test(String(offset || "")) ? offset : "";
      return `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${match[6]}${suffix}`;
    }

    function degrees(values, reference) {
      if (!Array.isArray(values) || values.length < 3) return null;
      const result = Number(values[0]) + Number(values[1]) / 60 + Number(values[2]) / 3600;
      if (!Number.isFinite(result)) return null;
      return /[SW]/i.test(String(reference || "")) ? -result : result;
    }

    const rootIfd = readIfd(u32(4));
    const exifIfd = readIfd(rootIfd.get(0x8769));
    const gpsIfd = readIfd(rootIfd.get(0x8825));
    const capturedAt = normalizeTime(exifIfd.get(0x9003) || exifIfd.get(0x9004) || rootIfd.get(0x0132), exifIfd.get(0x9011));
    const latitude = degrees(gpsIfd.get(0x0002), gpsIfd.get(0x0001));
    const longitude = degrees(gpsIfd.get(0x0004), gpsIfd.get(0x0003));
    return {
      ...(capturedAt ? { capturedAt } : {}),
      ...(latitude !== null && longitude !== null ? { latitude, longitude } : {}),
    };
  }

  function parseJpeg(buffer) {
    const view = new DataView(buffer);
    if (view.byteLength < 4 || view.getUint16(0) !== 0xffd8) return {};
    let offset = 2;
    while (offset + 4 <= view.byteLength) {
      if (view.getUint8(offset) !== 0xff) break;
      const marker = view.getUint8(offset + 1);
      offset += 2;
      if (marker === 0xda || marker === 0xd9) break;
      const size = view.getUint16(offset);
      if (size < 2 || offset + size > view.byteLength) break;
      if (marker === 0xe1 && size >= 8 && ascii(view, offset + 2, 6) === "Exif") {
        return parseTiff(buffer, offset + 8, size - 8);
      }
      offset += size;
    }
    return {};
  }

  function parsePng(buffer) {
    const view = new DataView(buffer);
    const signature = [137, 80, 78, 71, 13, 10, 26, 10];
    if (view.byteLength < 12 || signature.some((byte, index) => view.getUint8(index) !== byte)) return {};
    let offset = 8;
    while (offset + 12 <= view.byteLength) {
      const length = view.getUint32(offset);
      const type = ascii(view, offset + 4, 4);
      const dataOffset = offset + 8;
      if (dataOffset + length + 4 > view.byteLength) break;
      if (type === "eXIf") return parseTiff(buffer, dataOffset, length);
      if (type === "IEND") break;
      offset = dataOffset + length + 4;
    }
    return {};
  }

  async function extract(file) {
    if (!file || typeof file.slice !== "function") return {};
    const name = String(file.name || "").toLowerCase();
    const type = String(file.type || "").toLowerCase();
    const supported = /image\/(jpeg|jpg|png)/.test(type) || /\.(jpe?g|png)$/.test(name);
    if (!supported) return {};
    try {
      const buffer = await file.slice(0, MAX_METADATA_BYTES).arrayBuffer();
      return (type.includes("png") || name.endsWith(".png")) ? parsePng(buffer) : parseJpeg(buffer);
    } catch (_) {
      return {};
    }
  }

  return { extract, parseJpeg, parsePng, parseTiff };
});
