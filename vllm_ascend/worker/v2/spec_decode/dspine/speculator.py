# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.dspine.speculator import DSpineSpeculator

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator


class AscendDSpineSpeculator(DSpineSpeculator, AscendDFlashSpeculator):
    """MRV2 DSpine with shared core scoring and Ascend attention/graph hooks."""

    _lmhead_tp_sample_draft_supported = False

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        spec = vllm_config.speculative_config
        assert spec is not None
        if spec.draft_sample_method != "greedy":
            raise ValueError("DSpine MRV2 on Ascend currently requires greedy draft sampling")
        if not spec.enforce_eager:
            raise ValueError("DSpine MRV2 on Ascend currently requires draft enforce_eager=True")
        parallel = vllm_config.parallel_config
        for name in (
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel, name) != 1:
                raise ValueError(f"DSpine MRV2 on Ascend currently requires {name}=1")
        super().__init__(vllm_config, device)
        # The upstream constructor explicitly calls DFlashSpeculator.__init__,
        # bypassing AscendDFlashSpeculator.__init__ in this MRO.
        self._lmhead_tp_validate_draft_sampling()
