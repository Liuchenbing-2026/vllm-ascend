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
# DIAGNOSTIC: pinpoint the FULL_DECODE_ONLY "sample_tokens timed out" hang.
#
# The FDO request-time hang (bug.md ⑰ / ⑤y / ㉖) is currently only OBSERVED as
# `RPC call to sample_tokens timed out` + worker unresponsive. Whether it is a
# TP collective deadlock, an NPU op that never completes, or a stream sync that
# never fires is NOT yet established — the decode->sample path has ~20 collective
# / rank-divergent points, so guessing which one to "fix" is a shot in the dark.
#
# This patch arms two non-destructive stack dumpers on every worker process so
# the exact hang site on every TP rank can be captured:
#
#   1. faulthandler.enable() — on a fatal signal, dump all thread stacks.
#   2. faulthandler.register(SIGUSR1, all_threads=True) — while the request is
#      hung, run `kill -USR1 <worker_pid>` for EACH worker; every rank prints its
#      full Python stack (all threads) to stderr WITHOUT dying. Compare the ranks:
#        * all stuck in the same collective (all_reduce / all_gather / broadcast)
#          with mismatched args  -> real TP deadlock (I fix the divergence)
#        * stuck on an NPU op .wait()/synchronize                 -> op/stream issue
#        * one rank elsewhere than the others                     -> rank-divergent
#          host branch before a collective
#   3. optional: set VLLM_ASCEND_HANG_DUMP_SEC=<n> to auto-dump all stacks every
#      <n> seconds (use a large n like 600 so the long graph-compile phase is not
#      spammed; SIGUSR1 on-demand is preferred and cleaner).
#
# The worker PID is printed at arm time so the operator knows which PIDs to
# signal. This patch is inert unless a signal fires / the env var is set.
#
# Alternative with zero code (if py-spy is installed in the container):
#   py-spy dump --pid <worker_pid>     # for each hung worker
#
# Remove this file + its import once the hang site is captured and fixed.
# ---------------------------------------------------------------------------
import faulthandler
import os
import signal
import sys

from vllm.logger import init_logger

logger = init_logger(__name__)

# Dump C-level + Python tracebacks on fatal signals (SIGSEGV/SIGABRT/...).
try:
    faulthandler.enable()
except Exception:  # noqa: BLE001
    pass

# On-demand, non-destructive full-stack dump: `kill -USR1 <pid>`.
_sig = getattr(signal, "SIGUSR1", None)
if _sig is not None:
    try:
        faulthandler.register(_sig, all_threads=True, chain=True)
        logger.warning(
            "[hang-dump] armed: pid=%d — during a sample_tokens hang run "
            "`kill -USR1 %d` (one per worker PID) to dump all thread stacks "
            "to stderr without killing the worker.",
            os.getpid(), os.getpid())
    except Exception as e:  # noqa: BLE001
        logger.warning("[hang-dump] SIGUSR1 register failed: %s", e)

# Optional auto-dump timer (opt-in; keep n large to avoid spamming graph compile).
_sec = os.environ.get("VLLM_ASCEND_HANG_DUMP_SEC")
if _sec:
    try:
        faulthandler.dump_traceback_later(int(_sec), repeat=True, file=sys.stderr)
        logger.warning("[hang-dump] auto-dump every %ss armed (pid=%d).",
                       _sec, os.getpid())
    except Exception as e:  # noqa: BLE001
        logger.warning("[hang-dump] dump_traceback_later failed: %s", e)
