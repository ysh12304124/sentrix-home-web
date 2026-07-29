const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { URL } = require("node:url");

const root = __dirname;
const port = Number(process.env.PORT || 4173);
const cogneeBaseUrl = (process.env.COGNEE_BASE_URL || "").replace(/\/$/, "");
const backendBaseUrl = (process.env.SENTRIX_BACKEND_URL || "http://127.0.0.1:8090").replace(/\/$/, "");

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
};

const json = (res, status, payload) => {
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  res.end(JSON.stringify(payload));
};

const readBody = (req) => new Promise((resolve) => {
  let body = "";
  req.on("data", (chunk) => { body += chunk; });
  req.on("end", () => {
    try { resolve(body ? JSON.parse(body) : {}); }
    catch { resolve({}); }
  });
});

const readRawBody = (req) => new Promise((resolve, reject) => {
  const chunks = [];
  req.on("data", (chunk) => chunks.push(chunk));
  req.on("end", () => resolve(Buffer.concat(chunks)));
  req.on("error", reject);
});

async function proxyBackend(req, res, url) {
  try {
    const body = req.method === "GET" || req.method === "HEAD" ? undefined : await readRawBody(req);
    const response = await fetch(`${backendBaseUrl}${url.pathname}${url.search}`, {
      method: req.method,
      headers: { "content-type": req.headers["content-type"] || "application/json" },
      body,
      signal: AbortSignal.timeout(240000),
    });
    const payload = await response.arrayBuffer();
    res.writeHead(response.status, { "content-type": response.headers.get("content-type") || "application/json; charset=utf-8", "cache-control": "no-store" });
    res.end(Buffer.from(payload));
  } catch (error) {
    json(res, 502, { error: "Sentrix backend unavailable", detail: error.message });
  }
}

async function searchCognee(query) {
  if (!cogneeBaseUrl) return null;
  const response = await fetch(`${cogneeBaseUrl}/api/v1/search`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      searchType: process.env.COGNEE_SEARCH_TYPE || "GRAPH_COMPLETION",
      query,
      includeReferences: true,
    }),
    signal: AbortSignal.timeout(18000),
  });
  if (!response.ok) throw new Error(`Cognee returned ${response.status}`);
  const payload = await response.json();
  const answer = Array.isArray(payload)
    ? payload.map((item) => typeof item === "string" ? item : JSON.stringify(item)).join("\n")
    : JSON.stringify(payload);
  return {
    query,
    answer,
    confidence: 0.71,
    memories: ["episodic", "semantic"],
    retrievalTrace: [
      { type: "episodic", label: "事件记忆", detail: "Cognee 图检索已返回候选上下文" },
      { type: "semantic", label: "语义记忆", detail: "Cognee Graph Completion 已生成回答" },
      { type: "visual-evidence", label: "视觉证据", detail: "图片证据接口已保留，视频编码接口暂未启用" },
    ],
    source: "cognee",
  };
}

function mockSearch(query) {
  return {
    query,
    answer: "我在家庭事件记忆中找到了 3 个相关片段，并用图片、语音转写和人物关系进行了交叉确认。",
    confidence: 0.86,
    memories: ["episodic", "semantic", "visual-evidence"],
    retrievalTrace: [
      { type: "episodic", label: "事件记忆", detail: "按时间、人物和地点召回 6 个候选事件" },
      { type: "semantic", label: "语义记忆", detail: "确认家庭成员与地点关系" },
      { type: "visual-evidence", label: "视觉证据", detail: "返回 5 张图片证据，视频接口暂未启用" },
    ],
    source: "demo",
  };
}

function handleApi(req, res, url) {
  if (req.method === "GET" && url.pathname === "/api/health") {
    return json(res, 200, {
      status: "ok",
      mode: cogneeBaseUrl ? "local-cognee-adapter" : "local-mock",
      cognee: cogneeBaseUrl ? "connected" : "adapter-ready",
      models: { gamma4_12B: "adapter-ready", whisper: "adapter-ready", face: "adapter-ready" },
      videoExtraction: "reserved",
    });
  }

  if (req.method === "POST" && url.pathname === "/api/search") {
    return readBody(req).then(async (body) => {
      const query = body.query || "";
      try {
        const result = await searchCognee(query);
        return json(res, 200, result || mockSearch(query));
      } catch (error) {
        return json(res, 200, { ...mockSearch(query), fallbackReason: error.message });
      }
    });
  }

  if (req.method === "POST" && url.pathname === "/api/import") {
    return readBody(req).then((body) => json(res, 202, {
      accepted: true,
      assetId: `asset_${Date.now()}`,
      fileName: body.fileName || "unknown",
      status: body.mediaType === "video" ? "video-extraction-reserved" : "queued",
    }));
  }

  return json(res, 404, { error: "Not found" });
}

function serveFile(req, res, url) {
  const requested = url.pathname === "/" ? "/index.html" : url.pathname;
  const filePath = path.normalize(path.join(root, requested));
  if (!filePath.startsWith(root)) return json(res, 403, { error: "Forbidden" });

  fs.readFile(filePath, (error, data) => {
    if (error) {
      if (error.code === "ENOENT") return serveFile(req, res, new URL("/index.html", "http://localhost"));
      return json(res, 500, { error: "Unable to read file" });
    }
    res.writeHead(200, { "content-type": contentTypes[path.extname(filePath)] || "application/octet-stream" });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  if (url.pathname.startsWith("/api/") && backendBaseUrl) return proxyBackend(req, res, url);
  if (url.pathname.startsWith("/api/")) return handleApi(req, res, url);
  return serveFile(req, res, url);
});

  server.listen(port, "0.0.0.0", () => {
  console.log(`Sentrix Home Web running at http://0.0.0.0:${port}`);
  console.log(`Cognee adapter: ${cogneeBaseUrl || "mock fallback"}`);
});
