"""Guard: runtime code must never embed benchmark queries, GT filenames or
Asset IDs (Phase R R1A).

The three-album benchmark set lives outside the repository (``~/Downloads/
samples``).  This test scans production code and default configs for leaked
benchmark strings.  A leak is a red gate for any Phase R stage.

Scan targets:
- backend/*.py and backend/**/*.py
- configs/retrieval/defaults.json
- scripts/runtime/*.sh
- scripts/maintenance/*.py

If the samples directory is absent the test is skipped (CI with no benchmark
copy must not fail); when present it asserts a clean runtime.
"""

import json
import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_ROOT = os.getenv("SENTRIX_BENCHMARK_SAMPLES", str(Path.home() / "Downloads" / "samples"))


def _load_benchmark_strings():
    queries, files = [], []
    for album in ("album1", "album2", "album3"):
        path = Path(SAMPLES_ROOT) / album / "query.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for case in data:
            query = case.get("query_cn") or ""
            if query.strip():
                queries.append(query.strip())
            files.extend(name for name in (case.get("ground_truth") or []) if name)
    return queries, files


def _scan_targets():
    """Production code + default configs; tests are excluded because the guard
    itself must reference benchmark data to assert against it."""
    targets = []
    backend_dir = REPO_ROOT / "backend"
    for path in backend_dir.rglob("*.py"):
        if "tests" in path.parts:
            continue
        targets.append(path)
    for path in (REPO_ROOT / "scripts" / "runtime").glob("*.sh"):
        targets.append(path)
    for path in (REPO_ROOT / "scripts" / "maintenance").glob("*.py"):
        targets.append(path)
    config_defaults = REPO_ROOT / "configs" / "retrieval" / "defaults.json"
    if config_defaults.is_file():
        targets.append(config_defaults)
    return targets


class NoBenchmarkRuntimeDependencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries, cls.files = _load_benchmark_strings()
        if not (cls.queries or cls.files):
            raise unittest.SkipTest("benchmark samples not present")

    def test_no_benchmark_query_text_in_runtime(self):
        violations = []
        for target in _scan_targets():
            text = target.read_text(encoding="utf-8", errors="ignore")
            for query in self.queries:
                if query in text:
                    violations.append(f"{target}: contains benchmark query {query!r}")
        self.assertEqual([], violations)

    def test_no_benchmark_gt_filename_in_runtime(self):
        violations = []
        for target in _scan_targets():
            text = target.read_text(encoding="utf-8", errors="ignore")
            for file_name in self.files:
                if file_name in text:
                    violations.append(f"{target}: contains GT filename {file_name!r}")
        self.assertEqual([], violations)

    def test_no_samples_path_in_runtime(self):
        forbidden = {"Downloads/samples", "samples/album1", "samples/album2", "samples/album3"}
        violations = []
        for target in _scan_targets():
            text = target.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                if token in text:
                    violations.append(f"{target}: contains benchmark path token {token!r}")
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
