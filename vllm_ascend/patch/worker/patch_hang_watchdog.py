#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
# ---------------------------------------------------------------------------
# DIAGNOSTIC: auto-capture the INTERMITTENT FDO sample_tokens hang (bug.md ㉙).
#
# ㉙ finding: FDO + LoRA runs the first few requests fine (2-3s each), then a
# later request hangs at sample_tokens -> RPC timeout -> EngineDeadError. So it
# is NOT a first-request hard deadlock; it triggers after N requests (suspected
# TP graph-replay resource/state accumulation). The ㉘ manual capture missed it
# because a fixed `sleep 20s` fired while the worker was already idle.
#
# This watchdog removes the timing guesswork: it wraps NPUWorker.execute_model
# and NPUWorker.sample_tokens; each call arms a one-shot
# faulthandler.dump_traceback_later(THRESHOLD). If the call returns in time the
# timer is cancelled (no output). If ANY call hangs longer than THRESHOLD
# seconds, faulthandler auto-dumps ALL thread stacks of THAT worker process to
# stderr — on every TP rank independently — WITHOUT killing it. So the moment
# the intermittent hang happens, the exact stuck stack of both ranks lands in
# the worker logs, no manual `kill -USR1` / py-spy timing needed.
#
# THRESHOLD = VLLM_ASCEND_WATCHDOG_SEC (default 15s; well above the ~2-3s normal
# decode, and set it below the engine's sample_tokens RPC timeout so the dump
# fires BEFORE the engine is torn down). Set to 0 to disable.
#
# Read the dumps: compare the two ranks' stacks.
#   * both stuck in the same collective (all_reduce/all_gather/broadcast) -> TP
#     deadlock; look at what diverged over the preceding requests.
#   * stuck in an NPU op .wait()/synchronize / graph replay -> op/stream/handle
#     exhaustion.
#   * one rank past the collective, the other waiting -> host-branch divergence.
#
# Remove this file + its import once the hang stack is captured.
# ---------------------------------------------------------------------------
import faulthandler
import functools
import os
import sys
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    _SEC = float(os.environ.get("VLLM_ASCEND_WATCHDOG_SEC", "15") or 0)
except ValueError:
    _SEC = 15.0


def _wrap(fn, name):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        t0 = time.monotonic()
        # One-shot: dump all thread stacks if this call outlives THRESHOLD.
        faulthandler.dump_traceback_later(_SEC, repeat=False, file=sys.stderr)
        try:
            return fn(self, *args, **kwargs)
        finally:
            faulthandler.cancel_dump_traceback_later()
            dt = time.monotonic() - t0
            if dt > _SEC:
                logger.warning(
                    "[hang-watchdog] %s took %.1fs (>%.0fs threshold) on pid=%d "
                    "— all-thread stack was dumped to stderr above.",
                    name, dt, _SEC, os.getpid())
    return wrapped


if _SEC > 0:
    import sys as _sys

    # adapt_patch() (which imports this module) is invoked from inside
    # vllm_ascend.worker.worker, so a top-level `from vllm_ascend.worker.worker
    # import NPUWorker` can hit a partially-initialised module -> circular
    # ImportError (bug.md ㉜). Fetch the already-loaded module object from
    # sys.modules WITHOUT triggering the import machinery; only fall back to a
    # real import if it is genuinely absent.
    _NPUWorker = None
    _mod = _sys.modules.get("vllm_ascend.worker.worker")
    if _mod is not None:
        _NPUWorker = getattr(_mod, "NPUWorker", None)
    if _NPUWorker is None:
        try:
            from vllm_ascend.worker.worker import NPUWorker as _NPUWorker
        except Exception as _e:  # noqa: BLE001
            _NPUWorker = None
            logger.warning(
                "[hang-watchdog] NPUWorker not importable yet (%s); "
                "watchdog NOT armed. Re-run; or use patch_hang_dump SIGUSR1.",
                _e)

    if _NPUWorker is not None:
        _NPUWorker.execute_model = _wrap(_NPUWorker.execute_model,
                                         "execute_model")
        _NPUWorker.sample_tokens = _wrap(_NPUWorker.sample_tokens,
                                         "sample_tokens")
        logger.warning(
            "[hang-watchdog] armed on NPUWorker.execute_model/sample_tokens "
            "(threshold=%.0fs, pid=%d). A hang beyond threshold auto-dumps all "
            "thread stacks; set VLLM_ASCEND_WATCHDOG_SEC=0 to disable.",
            _SEC, os.getpid())
