const fs = require("fs");
const path = require("path");
const test = require("node:test");
const assert = require("node:assert/strict");

const root = path.resolve(__dirname, "..");
const app = fs.readFileSync(path.join(root, "src", "app.js"), "utf8");
const api = fs.readFileSync(path.join(root, "src", "api.js"), "utf8");
const css = fs.readFileSync(path.join(root, "src", "styles.css"), "utf8");
const server = fs.readFileSync(path.join(root, "server.js"), "utf8");

test("video scenes render as keyframe stacks on the existing timeline", () => {
  assert.match(app, /source_type === "video_scene"/);
  assert.match(app, /videoSceneStack/);
  assert.match(app, /keyframe_assets/);
  assert.match(app, /source_scene_index/);
  assert.match(app, /source_asset_id/);
  assert.match(app, /source_timestamp_sec/);
  assert.match(css, /scene-frame-stack/);
  assert.match(css, /transform: translate/);
});

test("scene evidence seeks the preserved source video", () => {
  assert.match(app, /scene-video-player/);
  assert.match(app, /data-action="seek-video"/);
  assert.match(app, /player\.currentTime/);
  assert.match(api, /\/api\/videos\/\$\{encodeURIComponent\(id\)\}\/scenes/);
  assert.match(api, /reprocessVideo/);
  assert.match(server, /"range"/);
  assert.match(server, /"content-range"/);
  assert.match(server, /"accept-ranges"/);
});

test("upload UI exposes real processing stages without debug jargon", () => {
  for (const label of ["读取视频元数据", "关键帧与场景切分", "场景图片语义理解", "事件记忆构建"]) {
    assert.match(app, new RegExp(label));
  }
  assert.doesNotMatch(app, /WorldMM Scene|VIDEO SCENE|LOCAL PIPELINE/);
  assert.doesNotMatch(app, /video-extraction-reserved|视频编码：预留/);
});
