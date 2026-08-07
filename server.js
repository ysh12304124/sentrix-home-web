const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { URL } = require("node:url");

const root = __dirname;
const port = Number(process.env.PORT || 11000);
// The web portal uses the Agent-capable API as its single authority.
// Default API is the project-local backend that reads ./data/sentrix.db.
const backendBaseUrl = (process.env.SENTRIX_BACKEND_URL || "http://127.0.0.1:11001").replace(/\/$/, "");

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
    const timeoutMs = url.pathname === "/api/model-profiles/switch" ? 1_000_000 : 240_000;
    const response = await fetch(`${backendBaseUrl}${url.pathname}${url.search}`, {
      method: req.method,
      headers: { "content-type": req.headers["content-type"] || "application/json" },
      body,
      signal: AbortSignal.timeout(timeoutMs),
    });
    const payload = await response.arrayBuffer();
    res.writeHead(response.status, { "content-type": response.headers.get("content-type") || "application/json; charset=utf-8", "cache-control": "no-store" });
    res.end(Buffer.from(payload));
  } catch (error) {
    json(res, 502, { error: "Sentrix backend unavailable", detail: error.message });
  }
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
  if (url.pathname.startsWith("/api/")) return proxyBackend(req, res, url);
  return serveFile(req, res, url);
});

server.listen(port, "0.0.0.0", () => {
  console.log(`Sentrix Home Web running at http://0.0.0.0:${port}`);
  console.log(`Sentrix API proxy: ${backendBaseUrl}`);
});
