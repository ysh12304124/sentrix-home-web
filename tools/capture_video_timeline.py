#!/usr/bin/env python3
"""Capture the deployed video timeline through Chrome DevTools Protocol."""

import argparse
import asyncio
import base64
import json
from pathlib import Path

import requests
import websockets


class CDP:
    def __init__(self, socket):
        self.socket = socket
        self.sequence = 0

    async def call(self, method, params=None):
        self.sequence += 1
        request_id = self.sequence
        await self.socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(await self.socket.recv())
            if message.get("id") == request_id:
                if message.get("error"):
                    raise RuntimeError(message["error"])
                return message.get("result") or {}


async def capture(debug_url, app_url, output_dir):
    targets = requests.get(f"{debug_url}/json", timeout=10).json()
    target = next(item for item in targets if item.get("type") == "page")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=50 * 1024 * 1024) as socket:
        cdp = CDP(socket)
        await cdp.call("Page.enable")
        await cdp.call("Runtime.enable")
        await cdp.call("Emulation.setDeviceMetricsOverride", {
            "width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False,
        })

        async def evaluate(expression):
            return await cdp.call("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})

        async def shot(name):
            result = await cdp.call("Page.captureScreenshot", {
                "format": "png", "captureBeyondViewport": True, "fromSurface": True,
            })
            (output / name).write_bytes(base64.b64decode(result["data"]))

        await cdp.call("Page.navigate", {"url": app_url.rstrip("/") + "/#/timeline"})
        await asyncio.sleep(2)
        await evaluate("localStorage.setItem('sentrix.scopeId','video-img3957'); location.reload()")
        await asyncio.sleep(8)
        await shot("03_timeline.png")

        await evaluate("document.querySelector('[data-action=\"open-event\"]')?.click()")
        await asyncio.sleep(4)
        await shot("04_scene_detail.png")

        keyframes = await evaluate("[...document.querySelectorAll('[data-action=\"seek-video\"]')].map(x=>x.dataset.timestampSec)")
        before_seek = await evaluate("(()=>{const p=document.querySelector('#scene-video-player'); return {time:p?.currentTime,duration:p?.duration,ready:p?.readyState,seekable:p?.seekable?.length};})()")
        immediate = await evaluate("(()=>{document.querySelectorAll('[data-action=\"seek-video\"]')[1]?.click(); const p=document.querySelector('#scene-video-player'); return {time:p?.currentTime,ready:p?.readyState};})()")
        await asyncio.sleep(0.75)
        current = await evaluate("(()=>{const p=document.querySelector('#scene-video-player'); if(p)p.pause(); return {time:p?.currentTime,ready:p?.readyState,paused:p?.paused};})()")
        await evaluate("document.querySelector('.scene-keyframe-grid')?.scrollIntoView({block:'center'})")
        seek_evidence = {
            "keyframe_timestamps": keyframes.get("result", {}).get("value"),
            "player_before_click": before_seek.get("result", {}).get("value"),
            "player_immediately_after_click": immediate.get("result", {}).get("value"),
            "player_after_second_keyframe_click": current.get("result", {}).get("value"),
        }
        (output / "05_video_seek.json").write_text(json.dumps(seek_evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        await shot("05_video_seek.png")

        await cdp.call("Page.navigate", {"url": app_url.rstrip("/") + "/#/imports"})
        await asyncio.sleep(6)
        await shot("01_import.png")
        await shot("02_processing.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-url", default="http://127.0.0.1:9223")
    parser.add_argument("--app-url", default="http://127.0.0.1:4174")
    parser.add_argument("--output", default="artifacts/video_timeline")
    args = parser.parse_args()
    asyncio.run(capture(args.debug_url, args.app_url, args.output))


if __name__ == "__main__":
    main()
