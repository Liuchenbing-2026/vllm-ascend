"""Recompute whole-cohort metrics from per-request records, without exclusions."""

import argparse
import json
from pathlib import Path

SCENES = ["20_20_c16", "2k200_c60", "8k1_c256"]
MODES = ["candidate", "bf16", "optimized"]


def summarize(root):
    runs = sorted((root / "results").glob("write_ab_*"))
    assert len(runs) == 1
    rows = []
    total = 0
    payloads = {}
    for scene in SCENES:
        for mode in MODES:
            cohorts = [json.loads(p.read_text()) for p in sorted(runs[0].glob(f"{scene}_{mode}_r[123].json"))]
            if not cohorts:
                continue
            for cohort in cohorts:
                assert cohort["complete_cohort"] and cohort["failed"] == 0
                assert cohort["successful"] == cohort["num_prompts"] == len(cohort["requests"]) == cohort["concurrency"]
                expected = payloads.setdefault(scene, cohort["payload_sha256"])
                assert cohort["payload_sha256"] == expected
                for request in cohort["requests"]:
                    assert request["status"] == "success" and request["http_status"] == 200
                    assert request["usage"]["prompt_tokens"] == cohort["input_len"]
                    assert request["usage"]["completion_tokens"] == cohort["output_len"]
            requests = [q for c in cohorts for q in c["requests"]]
            seconds = sum(c["duration_s"] for c in cohorts)
            output_tps = sum(q["usage"]["completion_tokens"] for q in requests) / seconds
            input_tps = sum(q["usage"]["prompt_tokens"] for q in requests) / seconds
            rows.append({"scene": scene, "mode": mode, "rounds": len(cohorts), "successful": len(requests), "failed": 0,
                         "ttft_s": sum(q["ttft_s"] for q in requests) / len(requests),
                         "tpot_ms": sum(q["tpot_s"] for q in requests) * 1000 / len(requests) if cohorts[0]["output_len"] > 1 else None,
                         "output_tps": output_tps, "input_tps": input_tps})
            total += len(requests)
    for scene in SCENES:
        current = next((x for x in rows if x["scene"] == scene and x["mode"] == "candidate"), None)
        if not current:
            continue
        metric = "input_tps" if scene == "8k1_c256" else "output_tps"
        for mode in ["bf16", "optimized"]:
            reference = next((x for x in rows if x["scene"] == scene and x["mode"] == mode), None)
            if reference and current["rounds"] == reference["rounds"] == 3:
                current["gain_vs_" + mode + "_percent"] = (current[metric] / reference[metric] - 1) * 100
    equality = None
    a, b = [runs[0] / ("precision_model_" + mode + ".json") for mode in ["candidate", "optimized"]]
    if a.exists() and b.exists():
        aa, bb = [json.loads(p.read_text())["records"] for p in [a, b]]
        assert len(aa) == len(bb) == 4
        tokens = 0
        for x, y in zip(aa, bb):
            assert x["payload_sha256"] == y["payload_sha256"]
            first, second = [r["response"]["choices"][0]["logprobs"]["tokens"] for r in [x, y]]
            assert first == second
            tokens += len(first)
        assert tokens == 440
        equality = {"samples": 4, "equal_token_ids": tokens, "scope": "candidate TQ vs prior TQ, not BF16 semantic quality"}
    complete = len(rows) == 9 and all(x["rounds"] == 3 for x in rows)
    if complete:
        assert total == 2988
    return {"run_id": runs[0].name, "matrix_complete": complete, "successful": total, "failed": 0,
            "rows": rows, "token_equality": equality, "no_retries_or_exclusions": True,
            "timing": "whole HTTP cohort, payload generation excluded, warmup/profile excluded"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"run_id": report["run_id"], "matrix_complete": report["matrix_complete"], "successful": report["successful"], "rows": len(report["rows"])}))
