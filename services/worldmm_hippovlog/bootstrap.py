#!/usr/bin/env python3
"""Run an official WorldMM script with the local single-GPU model adapter."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

from runtime import configure_visual_sdpa, configure_worldmm, limit_worldmm_generation_threads, worldmm_root


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"build_memory", "eval"}:
        raise SystemExit("usage: bootstrap.py {build_memory|eval} [WorldMM script arguments]")
    name = sys.argv[1]
    target = worldmm_root() / ("preprocess/build_memory.py" if name == "build_memory" else "eval/eval.py")
    configure_worldmm()
    limit_worldmm_generation_threads()
    visual_step = name == "build_memory" and any(
        arg == "visual" or arg == "--step=visual" for arg in sys.argv[2:]
    )
    if visual_step:
        configure_visual_sdpa()
    sys.argv = [str(target), *sys.argv[2:]]
    if visual_step and "--split-id" not in sys.argv:
        # The upstream default forks the raw script, losing our local SDPA and
        # adapter/base-model compatibility patch in the child process.
        sys.argv.extend(["--split-id", "0", "--num-splits", "1"])
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
