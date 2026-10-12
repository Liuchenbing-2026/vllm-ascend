#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Regression tests for ``benchmarks/scripts/convert_json_to_markdown.py``.

Each result must be classified by its own file name. A parent directory whose
name contains another result keyword (``serving_reports``, ``latency_reports``,
``throughput_reports``) used to override the classification and make the script
read keys the result does not have.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
# Reported by benchmarks/requirements-bench.txt; the report builder imports it.
pytest.importorskip("tabulate")

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "convert_json_to_markdown.py"
MARKDOWN_TEMPLATE = (
    "{latency_tests_markdown_table}\n{throughput_tests_markdown_table}\n{serving_tests_markdown_table}\n"
)

LATENCY_RESULT = {
    "avg_latency": 0.01,
    "percentiles": {str(percentile): 0.02 for percentile in (10, 25, 50, 75, 90, 99)},
}
THROUGHPUT_RESULT = {
    "num_requests": 8,
    "total_num_tokens": 1024,
    "elapsed_time": 1.5,
    "requests_per_second": 5.33,
    "tokens_per_second": 682.67,
}
SERVING_RESULT = {
    "request_rate": 1.0,
    "request_throughput": 1.0,
    "output_throughput": 10.0,
    "median_ttft_ms": 100.0,
    "median_tpot_ms": 20.0,
    "median_itl_ms": 21.0,
}


def _run_report(results_dir: Path, output_dir: Path, template: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--results_folder",
            str(results_dir),
            "--output_folder",
            str(output_dir),
            "--markdown_template",
            str(template),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "parent_dir,result_name,result,expected_column,absent_column",
    [
        ("serving_reports", "latency_bench.json", LATENCY_RESULT, "Median latency (ms)", "TTFT (ms)"),
        ("throughput_reports", "latency_bench.json", LATENCY_RESULT, "Median latency (ms)", "Tput (tok/s)"),
        ("latency_reports", "throughput_bench.json", THROUGHPUT_RESULT, "Total num of tokens", "Median latency (ms)"),
        ("serving_reports", "throughput_bench.json", THROUGHPUT_RESULT, "Total num of tokens", "TTFT (ms)"),
        ("latency_reports", "serving_bench.json", SERVING_RESULT, "TTFT (ms)", "Median latency (ms)"),
        ("throughput_reports", "serving_bench.json", SERVING_RESULT, "TTFT (ms)", "Total num of tokens"),
    ],
)
def test_result_type_comes_from_the_file_name(
    tmp_path: Path, parent_dir: str, result_name: str, result: dict, expected_column: str, absent_column: str
):
    """A parent directory keyword must not reclassify a benchmark result."""
    results_dir = tmp_path / parent_dir
    results_dir.mkdir()
    (results_dir / result_name).write_text(json.dumps(result), encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    template = tmp_path / "template.md"
    template.write_text(MARKDOWN_TEMPLATE, encoding="utf-8")

    completed = _run_report(results_dir, output_dir, template)

    assert completed.returncode == 0, completed.stderr
    markdown = (output_dir / "benchmark_results.md").read_text(encoding="utf-8")
    assert expected_column in markdown
    assert absent_column not in markdown


@pytest.mark.parametrize("parent_dir", ["neutral_reports", "serving_reports", "throughput_reports"])
def test_unknown_result_name_is_skipped_regardless_of_parent_directory(tmp_path: Path, parent_dir: str):
    """An unrecognized file name is skipped even under a keyword directory."""
    results_dir = tmp_path / parent_dir
    results_dir.mkdir()
    (results_dir / "mystery_bench.json").write_text(json.dumps(LATENCY_RESULT), encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    template = tmp_path / "template.md"
    template.write_text(MARKDOWN_TEMPLATE, encoding="utf-8")

    completed = _run_report(results_dir, output_dir, template)

    assert completed.returncode == 0, completed.stderr
    assert "Skipping" in completed.stdout
    markdown = (output_dir / "benchmark_results.md").read_text(encoding="utf-8")
    assert "Median latency (ms)" not in markdown
    assert "TTFT (ms)" not in markdown
