#!/usr/bin/env python3
"""Download reproducible public image batches for Sentrix memory evaluation."""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://commons.wikimedia.org/w/api.php"


def fetch_json(params):
    request = Request(f"{API}?{urlencode(params)}", headers={"User-Agent": "SentrixHomeMemory/0.1 (local evaluation)"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def download_category(category, output, limit):
    output.mkdir(parents=True, exist_ok=True)
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "file",
        "gcmlimit": min(limit, 500),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 1600,
        "format": "json",
    }
    payload = fetch_json(params)
    pages = payload.get("query", {}).get("pages", {}).values()
    manifest = []
    for index, page in enumerate(pages):
        image = (page.get("imageinfo") or [{}])[0]
        url = image.get("thumburl") or image.get("url")
        if not url:
            continue
        suffix = Path(url.split("?", 1)[0]).suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        path = output / f"{index:04d}{suffix}"
        try:
            request = Request(url, headers={"User-Agent": "SentrixHomeMemory/0.1 (local evaluation)"})
            with urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
            manifest.append({"file": path.name, "title": page.get("title"), "source_url": url, "license": (image.get("extmetadata") or {}).get("LicenseShortName", {}).get("value")})
            time.sleep(0.05)
        except Exception as error:
            print(f"SKIP {url}: {error}")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"category": category, "downloaded": len(manifest), "output": str(output)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/test-albums/wikimedia"))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--categories", default="Family photographs,Family gatherings,Family portraits,Children")
    args = parser.parse_args()
    for category in [value.strip() for value in args.categories.split(",") if value.strip()]:
        target = args.output / category.lower().replace(" ", "-")
        download_category(category, target, args.limit)
