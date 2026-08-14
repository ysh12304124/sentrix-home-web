const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { execFile } = require("node:child_process");
const { URL } = require("node:url");

const root = __dirname;
const port = Number(process.env.PORT || 11000);
// The web portal uses the Agent-capable API as its single authority.
// Default API is the project-local backend that reads ./data/sentrix.db.
const backendBaseUrl = (process.env.SENTRIX_BACKEND_URL || "http://127.0.0.1:11001").replace(/\/$/, "");
const photobenchPort = Number(process.env.PHOTOBENCH_PORT || 8771);
const photobenchDir = path.join(root, "services", "photobench");
const photobenchPython = process.env.PHOTOBENCH_PYTHON || "python3";

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".heic": "image/heic",
  ".heif": "image/heif",
};

const json = (res, status, payload) => {
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  res.end(JSON.stringify(payload));
};

const readRawBody = (req) => new Promise((resolve, reject) => {
  const chunks = [];
  req.on("data", (chunk) => chunks.push(chunk));
  req.on("end", () => resolve(Buffer.concat(chunks)));
  req.on("error", reject);
});

async function proxyBackend(req, res, url) {
  try {
    const body = req.method === "GET" || req.method === "HEAD" ? undefined : await readRawBody(req);
    const isSse = /text\/event-stream/.test(req.headers.accept || "") || url.pathname.endsWith("/events");
    const timeoutMs = url.pathname === "/api/model-profiles/switch" ? 1_000_000 : (isSse ? 600_000 : 240_000);
    const response = await fetch(`${backendBaseUrl}${url.pathname}${url.search}`, {
      method: req.method,
      headers: { "content-type": req.headers["content-type"] || "application/json" },
      body,
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (isSse) {
      // Phase C C13：SSE 必须流式转发（EventSource 需要增量 chunk），不能一次读完。
      res.writeHead(response.status, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache",
        "x-accel-buffering": "no",
        connection: "keep-alive",
      });
      const reader = response.body.getReader();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          res.write(Buffer.from(value));
        }
        res.end();
      } catch (streamError) {
        try { res.end(); } catch (_) { /* client closed */ }
      }
      return;
    }
    const payload = await response.arrayBuffer();
    res.writeHead(response.status, { "content-type": response.headers.get("content-type") || "application/json; charset=utf-8", "cache-control": "no-store" });
    res.end(Buffer.from(payload));
  } catch (error) {
    json(res, 502, { error: "Sentrix backend unavailable", detail: error.message });
  }
}

async function ensurePhotobench(req, res) {
  const probeUrl = `http://127.0.0.1:${photobenchPort}/api/config`;
  try {
    const probe = await fetch(probeUrl, { signal: AbortSignal.timeout(2000) });
    if (probe.ok) return json(res, 200, { status: "running" });
  } catch (_) { /* not running yet */ }

  const logFile = path.join(photobenchDir, "logs", "orchestrator.log");
  const command = `cd '${photobenchDir}' && mkdir -p logs && set -a && . ./.env.local 2>/dev/null; set +a; nohup '${photobenchPython}' backend/benchmark_orchestrator.py --host 0.0.0.0 --port ${photobenchPort} >> '${logFile}' 2>&1 &`;
  execFile("bash", ["-c", command], () => {});

  for (let i = 0; i < 60; i++) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    try {
      const probe = await fetch(probeUrl, { signal: AbortSignal.timeout(1500) });
      if (probe.ok) return json(res, 200, { status: "started" });
    } catch (_) { /* keep waiting */ }
  }
  return json(res, 500, { status: "timeout" });
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
    res.writeHead(200, {
      "content-type": contentTypes[path.extname(filePath)] || "application/octet-stream",
      "cache-control": "no-cache",
    });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
  if (url.pathname === "/api/photobench/ensure") return ensurePhotobench(req, res);
  if (url.pathname.startsWith("/api/") || url.pathname === "/qa") return proxyBackend(req, res, url);
  return serveFile(req, res, url);
});

server.listen(port, "0.0.0.0", () => {
  console.log(`Sentrix Home Web running at http://0.0.0.0:${port}`);
  console.log(`Sentrix API proxy: ${backendBaseUrl}`);
});
