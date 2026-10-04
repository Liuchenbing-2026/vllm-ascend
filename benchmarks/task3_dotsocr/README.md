# Task 3 rejected RoPE experiment

These are the standalone scripts used for the dots.ocr operator experiment,
not an accepted serving optimization or upstream CI coverage.

- Run `test_rope_copy.py` in the original PA image, with this directory on
  `PYTHONPATH`. Its baseline imports the installed original RoPE, so do not run
  its latency comparison in the patched image.
- Run `test_rope_integration.py` in the candidate image. It loads the original
  source from `/work/rope.base.py`; obtain that file from
  `task3-dotsocr-pa-repro-20261004:vllm_ascend/ops/triton/rope.py` with `git show`.
- Run one device-owning container at a time. The measured device was Ascend
  910B4-1, logical device 0, with the fixed Task 3 software stack.

Historical checks: 16 bitwise/dynamic graph cases, 13 wrapper/alias/fallback
cases, and 90 existing RoPE unit tests passed. These do not establish complete
OCR accuracy. FULL serving throughput averaged 7,428 tokens/s versus PA's
7,450, with concurrent-page output changes. The candidate was not adopted.

The Model_test `ocr` Task 3 archive contains the full scripts, image recipe,
profiling summaries, raw benchmark timestamps, and accuracy limitations.
