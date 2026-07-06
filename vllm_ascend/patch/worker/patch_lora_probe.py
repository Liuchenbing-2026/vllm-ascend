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
# DIAGNOSTIC PROBE (temporary, not a fix).
#
# Goal: on vLLM v0.22.1rc1 the Qwen3.5-27B dense LoRA does not take effect;
# ⑤o localised it to the punica stacked buffers being zero at inference time
# (||lora_a_stacked|| == ||lora_b_stacked|| == 0). The open question is WHY:
#   H1  set_lora never runs   -> weights are simply never loaded, or
#   H2  reset_lora runs AFTER set_lora on the same (layer, index) -> the
#       just-loaded weights get wiped (the 0.22.1 model_manager alias guard,
#       keyed on id(module), misses the Ascend register_oot-swapped layers).
#
# This patch wraps BaseLinearLayerWithLoRA.{set_lora,reset_lora} and the
# MergedColumnParallelLinearWithLoRA.set_lora override (gate_up_proj) so every
# call prints:  ordinal seq no. + layer class + index + resulting stacked-A
# norm.  Run one LoRA request, then read the log: for each layer look at the
# ORDER of [set]/[reset] and whether a non-zero buffer is later zeroed.
#
# Verdict rule:
#   * a  [set] post=0.0                       -> H1 (weights never copied in)
#   * a  [set] post>0  followed by [reset] on -> H2 (set OK, wiped afterwards)
#     the SAME class+idx with pre>0
#
# Enable  : import is registered in patch/worker/__init__.py (guarded by the
#           VLLM_ASCEND_LORA_PROBE env var so it is a no-op unless asked for).
# Disable : unset VLLM_ASCEND_LORA_PROBE (default).  Delete this file + the
#           import line once the root cause is confirmed.
# ---------------------------------------------------------------------------
import itertools
import os

_ENABLED = os.environ.get("VLLM_ASCEND_LORA_PROBE", "0") not in ("0", "", "false", "False")

if _ENABLED:
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
    from vllm.lora.layers.column_parallel_linear import (
        MergedColumnParallelLinearWithLoRA)

    _seq = itertools.count()

    def _stacked_a_norm(self, index):
        """Norm of the loraA buffer for this slot; -1 if unreadable."""
        try:
            buf = self.lora_a_stacked
            if isinstance(buf, (tuple, list)):
                buf = buf[0]
            return float(buf[index].norm().item())
        except Exception as e:  # noqa: BLE001
            return f"err({type(e).__name__})"

    def _wrap_set(orig, tag):
        def set_lora(self, index, *args, **kwargs):
            ret = orig(self, index, *args, **kwargs)
            print(f"[LORA-PROBE {next(_seq):04d} set  {tag:6}] "
                  f"{type(self).__name__} idx={index} "
                  f"post_normA={_stacked_a_norm(self, index)}", flush=True)
            return ret
        return set_lora

    def _wrap_reset(orig):
        def reset_lora(self, index):
            pre = _stacked_a_norm(self, index)
            print(f"[LORA-PROBE {next(_seq):04d} reset      ] "
                  f"{type(self).__name__} idx={index} "
                  f"pre_normA={pre}", flush=True)
            return orig(self, index)
        return reset_lora

    # reset_lora is defined ONLY on the base class (no subclass override) ->
    # wrapping it here catches every wipe, whatever the concrete layer type.
    BaseLinearLayerWithLoRA.reset_lora = _wrap_reset(
        BaseLinearLayerWithLoRA.reset_lora)

    # set_lora: base covers Column / Row / QKV (inherited); the merged
    # gate_up_proj has its own override, wrap it too.
    BaseLinearLayerWithLoRA.set_lora = _wrap_set(
        BaseLinearLayerWithLoRA.set_lora, "base")
    MergedColumnParallelLinearWithLoRA.set_lora = _wrap_set(
        MergedColumnParallelLinearWithLoRA.set_lora, "merged")

    print("[LORA-PROBE] set_lora/reset_lora instrumentation installed "
          "(VLLM_ASCEND_LORA_PROBE on)", flush=True)
