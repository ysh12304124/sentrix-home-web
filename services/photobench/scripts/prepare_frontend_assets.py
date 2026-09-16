#!/usr/bin/env python3
"""Prepare PhotoBench production frontend assets for constrained HTTP servers.

By default this is a no-op.  Set --chunk-bytes (for example 30000 on Jetson
118) to split the Vite JS entry into byte-exact pieces and generate a loader
that reassembles the original UTF-8 bytes before evaluating it.  Splitting is
performed on bytes, not decoded text, so identifiers and UTF-8 sequences are
never changed at chunk boundaries.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCRIPT_RE = re.compile(r'<script[^>]+src="(?P<src>/assets/[^\"]+\.js)"[^>]*></script>')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=0)
    parser.add_argument("--loader-name", default="index-loader.js")
    args = parser.parse_args()
    dist = args.dist.resolve()
    index = dist / "index.html"
    if not index.is_file():
        raise SystemExit(f"dist index not found: {index}")
    if args.chunk_bytes <= 0:
        print("static asset splitting disabled")
        return 0
    if args.chunk_bytes < 1024:
        raise SystemExit("--chunk-bytes must be at least 1024")

    html = index.read_text(encoding="utf-8")
    match = SCRIPT_RE.search(html)
    if not match:
        raise SystemExit("Vite JS entry not found in dist/index.html")
    source = dist / match.group("src").lstrip("/")
    if not source.is_file():
        raise SystemExit(f"Vite JS entry not found: {source}")
    data = source.read_bytes()
    assets = dist / "assets"
    parts: list[str] = []
    for offset in range(0, len(data), args.chunk_bytes):
        name = f"{source.stem}.part{offset // args.chunk_bytes:03d}{source.suffix}"
        (assets / name).write_bytes(data[offset : offset + args.chunk_bytes])
        parts.append(name)

    loader = f"""const parts = {json.dumps(parts, ensure_ascii=False)};
(async () => {{
  try {{
    const buffers = [];
    let total = 0;
    for (const part of parts) {{
      const response = await fetch(`/assets/${{part}}`, {{ cache: "no-store" }});
      if (!response.ok) throw new Error(`${{part}}: HTTP ${{response.status}}`);
      const buffer = new Uint8Array(await response.arrayBuffer());
      buffers.push(buffer);
      total += buffer.byteLength;
    }}
    const combined = new Uint8Array(total);
    let offset = 0;
    for (const buffer of buffers) {{
      combined.set(buffer, offset);
      offset += buffer.byteLength;
    }}
    const source = new TextDecoder("utf-8", {{ fatal: true }}).decode(combined);
    new Function(`${{source}}\\n//# sourceURL=photobench-app.bundle.js`)();
  }} catch (error) {{
    console.error("PhotoBench 前端资源加载失败", error);
    document.body.innerHTML = `<pre style="padding:2rem;color:#b42318;white-space:pre-wrap">PhotoBench 前端资源加载失败：${{String(error)}}</pre>`;
  }}
}})();
"""
    loader_path = assets / args.loader_name
    loader_path.write_text(loader, encoding="utf-8")
    new_src = f"/assets/{args.loader_name}"
    html = html[: match.start("src")] + new_src + html[match.end("src") :]
    index.write_text(html, encoding="utf-8")
    print(f"split {source.name}: {len(data)} bytes into {len(parts)} parts of <= {args.chunk_bytes} bytes")
    print(f"loader: {loader_path.name} ({loader_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
