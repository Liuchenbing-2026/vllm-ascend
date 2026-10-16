"""Regression tests for benchmark result classification.

``benchmarks/scripts/convert_json_to_markdown.py`` must classify each result by
its own file name, not by the name of any parent directory. A result folder
called e.g. ``serving_reports`` used to make every ``*.json`` inside it look
like a serving result, which then crashed on the serving-only columns.

The script is a standalone CLI (its logic lives in ``__main__``), so the tests
drive it as a subprocess, exactly like the benchmark workflow does.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("tabulate")

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "convert_json_to_markdown.py"
TEMPLATE = "LATEST:{latency_tests_markdown_table}\nTHROUGHPUT:{throughput_tests_markdown_table}\nSERVING:{serving_tests_markdown_table}\n"

LATENCY_RESULT = {
    "avg_latency": 0.01,
    "percentiles": {str(p): 0.02 for p in (10, 25, 50, 75, 90, 99)},
}
THROUGHPUT_RESULT = {
    "num_requests": 8,
    "total_num_tokens": 1024,
    "elapsed_time": 1.0,
    "requests_per_second": 8.0,
    "tokens_per_second": 1024.0,
}
SERVING_RESULT = {
    "request_rate": 8.0,
    "request_throughput": 8.0,
    "output_throughput": 256.0,
    "median_ttft_ms": 100.0,
    "median_tpot_ms": 10.0,
    "median_itl_ms": 1.0,
}


def _run(root: Path, results_folder: Path) -> tuple[int, str, str]:
    template = root / "template.md"
    template.write_text(TEMPLATE, encoding="utf-8")
    output = root / "output"
    output.mkdir(exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--results_folder",
            str(results_folder),
            "--output_folder",
            str(output),
            "--markdown_template",
            str(template),
        ],
        capture_output=True,
        text=True,
    )
    report = (output / "benchmark_results.md").read_text(encoding="utf-8") if completed.returncode == 0 else ""
    return completed.returncode, completed.stderr, report


def _table(report: str, header: str) -> str:
    body = report.split(header, 1)[1]
    for other in ("LATEST:", "THROUGHPUT:", "SERVING:"):
        body = body.split(other, 1)[0]
    return body


def _write(folder: Path, name: str, payload: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(payload), encoding="utf-8")


def test_latency_result_under_serving_named_parent(tmp_path: Path):
    """A latency result must stay a latency result in a ``serving*`` folder."""
    results = tmp_path / "serving_reports"
    _write(results, "latency_fixture.json", LATENCY_RESULT)

    returncode, stderr, report = _run(tmp_path, results)

    assert returncode == 0, stderr
    assert "latency_fixture" in _table(report, "LATEST:")
    assert "latency_fixture" not in _table(report, "SERVING:")


def test_throughput_result_under_latency_named_parent(tmp_path: Path):
    """A throughput result must stay a throughput result in a ``latency*`` folder."""
    results = tmp_path / "latency_reports"
    _write(results, "throughput_fixture.json", THROUGHPUT_RESULT)

    returncode, stderr, report = _run(tmp_path, results)

    assert returncode == 0, stderr
    assert "throughput_fixture" in _table(report, "THROUGHPUT:")
    assert "throughput_fixture" not in _table(report, "LATEST:")


def test_classification_uses_file_name(tmp_path: Path):
    """Every result type is classified by its own file name in a neutral folder."""
    results = tmp_path / "reports"
    _write(results, "serving_fixture.json", SERVING_RESULT)
    _write(results, "latency_fixture.json", LATENCY_RESULT)
    _write(results, "throughput_fixture.json", THROUGHPUT_RESULT)

    returncode, stderr, report = _run(tmp_path, results)

    assert returncode == 0, stderr
    expected = {
        "LATEST:": "latency_fixture",
        "THROUGHPUT:": "throughput_fixture",
        "SERVING:": "serving_fixture",
    }
    for header, stem in expected.items():
        for name in expected.values():
            assert (name in _table(report, header)) == (name == stem), (header, name)


def test_unrecognized_file_name_is_skipped(tmp_path: Path):
    """An unknown JSON file name is skipped instead of being read as throughput."""
    results = tmp_path / "reports"
    _write(results, "mystery_fixture.json", SERVING_RESULT)
    _write(results, "latency_fixture.json", LATENCY_RESULT)

    returncode, stderr, report = _run(tmp_path, results)

    assert returncode == 0, stderr
    assert "mystery_fixture" not in report
    assert "latency_fixture" in _table(report, "LATEST:")
