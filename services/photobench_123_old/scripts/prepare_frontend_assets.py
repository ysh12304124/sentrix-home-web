#!/usr/bin/env python3
"""Prepare PhotoBench production frontend assets for constrained HTTP servers.

By default this is a no-op.  Set --chunk-bytes (for example 30000 on Jetson
118) to split oversized Vite assets into byte-exact pieces and generate a
loader that reassembles the original UTF-8 bytes before evaluating them.

This script must not rewrite index.html from a skeleton.  It only replaces the
Vite JS entry src and keeps every other tag (CSS, preload, meta).  Splitting
is performed on bytes, not decoded text, so identifiers and UTF-8 sequences
are never changed at chunk boundaries.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCRIPT_RE = re.compile(r'<script(?P<attrs>[^>]+)src="(?P<src>/assets/[^"]+\.js)"(?P<after>[^>]*)></script>')
STYLE_RE = re.compile(r'<link[^>]+rel="stylesheet"[^>]*href="(?P<href>/assets/[^"]+\.css)"[^>]*>')
LOADER_MARKERS = ("index-loader", "photobench-app.bundle.js")


def split_bytes(data: bytes, dest: Path, stem: str, suffix: str, chunk_bytes: int) -> list[str]:
    parts: list[str] = []
    for offset in range(0, len(data), chunk_bytes):
        name = f"{stem}.part{offset // chunk_bytes:03d}{suffix}"
        (dest / name).write_bytes(data[offset : offset + chunk_bytes])
        parts.append(name)
    return parts


def is_loader(path: Path) -> bool:
    if "loader" in path.name:
        return True
    if ".part" in path.name:
        return True
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:400]
    except OSError:
        return False
    return any(marker in sample for marker in LOADER_MARKERS)


def resolve_js_entry(dist: Path, html: str) -> Path:
    match = SCRIPT_RE.search(html)
    if not match:
        raise SystemExit("Vite JS entry not found in dist/index.html")
    source = dist / match.group("src").lstrip("/")
    if source.is_file() and not is_loader(source):
        return source
    assets = dist / "assets"
    candidates = [
        path for path in assets.glob("index-*.js")
        if path.is_file() and not is_loader(path)
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit(
        "JS entry is a loader/split file and the original Vite bundle is ambiguous. "
        f"candidates={[path.name for path in candidates]}"
    )


def resolve_stylesheet(dist: Path, html: str, explicit: str | None) -> str | None:
    if explicit:
        href = explicit if explicit.startswith("/") else f"/assets/{Path(explicit).name}"
        path = dist / href.lstrip("/")
        if not path.is_file():
            raise SystemExit(f"stylesheet not found: {path}")
        return href
    match = STYLE_RE.search(html)
    if match:
        href = match.group("href")
        if (dist / href.lstrip("/")).is_file():
            return href
        raise SystemExit(f"index.html stylesheet missing on disk: {href}")
    css_files = sorted((dist / "assets").glob("index-*.css"))
    if not css_files:
        return None
    if len(css_files) > 1:
        raise SystemExit(
            "index.html has no stylesheet and multiple CSS files exist; "
            f"pass --css. candidates={[path.name for path in css_files]}"
        )
    return f"/assets/{css_files[0].name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=0)
    parser.add_argument("--loader-name", default="index-loader.js")
    parser.add_argument("--css", default="", help="Stylesheet path or /assets/*.css when index.html lost its link")
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
    source = resolve_js_entry(dist, html)
    css_href = resolve_stylesheet(dist, html, args.css or None)
    assets = dist / "assets"
    js_parts = split_bytes(source.read_bytes(), assets, source.stem, source.suffix, args.chunk_bytes)

    css_parts: list[str] = []
    if css_href:
        css_path = dist / css_href.lstrip("/")
        css_data = css_path.read_bytes()
        if len(css_data) > args.chunk_bytes:
            css_parts = split_bytes(css_data, assets, css_path.stem, css_path.suffix, args.chunk_bytes)

    loader = f"""const jsParts = {json.dumps(js_parts, ensure_ascii=False)};
const cssParts = {json.dumps(css_parts, ensure_ascii=False)};
const cssHref = {json.dumps(css_href, ensure_ascii=False)};
async function fetchBytes(name) {{
  let lastError = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {{
    try {{
      const response = await fetch("/assets/" + name, {{ cache: "no-store" }});
      if (!response.ok) throw new Error(name + ": HTTP " + response.status);
      return new Uint8Array(await response.arrayBuffer());
    }} catch (error) {{
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 250 * (attempt + 1)));
    }}
  }}
  throw lastError;
}}
function concatBytes(chunks) {{
  const total = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const combined = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {{
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }}
  return combined;
}}
(async () => {{
  try {{
    if (cssHref && !document.querySelector('link[rel="stylesheet"][href="' + cssHref + '"]')) {{
      const link = document.createElement("link");
      link.rel = "stylesheet";
      if (cssParts.length) {{
        const cssText = new TextDecoder("utf-8", {{ fatal: true }}).decode(
          concatBytes(await Promise.all(cssParts.map(fetchBytes)))
        );
        link.href = URL.createObjectURL(new Blob([cssText], {{ type: "text/css" }}));
      }} else {{
        link.href = cssHref;
      }}
      document.head.appendChild(link);
    }}
    const source = new TextDecoder("utf-8", {{ fatal: true }}).decode(
      concatBytes(await Promise.all(jsParts.map(fetchBytes)))
    ).replace(/(?:\\n)?\\/\\/# sourceMappingURL=.*$/, "");
    new Function(source + "\\n//# sourceURL=photobench-app.bundle.js")();
  }} catch (error) {{
    console.error("PhotoBench 前端资源加载失败", error);
    document.body.innerHTML = `<pre style="padding:2rem;color:#b42318;white-space:pre-wrap">PhotoBench 前端资源加载失败：${{String(error)}}</pre>`;
  }}
}})();
"""
    loader_path = assets / args.loader_name
    loader_path.write_text(loader, encoding="utf-8")
    match = SCRIPT_RE.search(html)
    if not match:
        raise SystemExit("script tag disappeared while rewriting index.html")
    html = html[: match.start("src")] + f"/assets/{args.loader_name}" + html[match.end("src") :]
    if css_href and not STYLE_RE.search(html):
        html = html.replace(
            "<head>",
            f'<head>\n    <link rel="stylesheet" crossorigin href="{css_href}">',
            1,
        )
    index.write_text(html, encoding="utf-8")
    print(f"split {source.name}: {source.stat().st_size} bytes into {len(js_parts)} JS parts of <= {args.chunk_bytes} bytes")
    if css_href:
        print(f"stylesheet: {css_href}" + (f" ({len(css_parts)} parts)" if css_parts else ""))
    print(f"loader: {loader_path.name} ({loader_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
