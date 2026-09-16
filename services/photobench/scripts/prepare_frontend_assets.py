#!/usr/bin/env python3
"""Prepare PhotoBench production frontend assets for constrained HTTP servers.

By default this is a no-op.  Set --chunk-bytes (for example 30000 on Jetson
118) to split the Vite JS entry into byte-exact pieces and generate a loader
that reassembles the original UTF-8 bytes before evaluating it.  Splitting is
performed on bytes, not decoded text, so identifiers and UTF-8 sequences are
never changed at chunk boundaries.

Stylesheet links in dist/index.html are preserved.  If a previous split
dropped them, the matching Vite CSS file is re-injected into index.html and
the loader.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCRIPT_RE = re.compile(r'<script[^>]+src="(?P<src>/assets/[^\"]+\.js)"[^>]*></script>')
STYLE_RE = re.compile(r'<link[^>]+rel="stylesheet"[^>]*href="(?P<href>/assets/[^"]+\.css)"[^>]*>')


def discover_stylesheet(dist: Path, html: str) -> str | None:
    match = STYLE_RE.search(html)
    if match:
        href = match.group("href")
        if (dist / href.lstrip("/")).is_file():
            return href
    assets = dist / "assets"
    candidates = sorted(assets.glob("index-*.css"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return f"/assets/{candidates[0].name}"
    return None


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
    css_href = discover_stylesheet(dist, html)
    data = source.read_bytes()
    assets = dist / "assets"
    parts: list[str] = []
    for offset in range(0, len(data), args.chunk_bytes):
        name = f"{source.stem}.part{offset // args.chunk_bytes:03d}{source.suffix}"
        (assets / name).write_bytes(data[offset : offset + args.chunk_bytes])
        parts.append(name)

    css_inject = ""
    if css_href:
        css_inject = f"""const cssHref = {json.dumps(css_href, ensure_ascii=False)};
if (!document.querySelector('link[rel="stylesheet"][href="' + cssHref + '"]')) {{
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = cssHref;
  document.head.appendChild(link);
}}
"""

    loader = css_inject + f"""const parts = {json.dumps(parts, ensure_ascii=False)};
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
    if css_href and not STYLE_RE.search(html):
        html = html.replace(
            "<head>",
            f'<head>\n    <link rel="stylesheet" crossorigin href="{css_href}">',
            1,
        )
    index.write_text(html, encoding="utf-8")
    print(f"split {source.name}: {len(data)} bytes into {len(parts)} parts of <= {args.chunk_bytes} bytes")
    print(f"loader: {loader_path.name} ({loader_path.stat().st_size} bytes)")
    if css_href:
        print(f"stylesheet: {css_href}")
    else:
        print("stylesheet: none found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
