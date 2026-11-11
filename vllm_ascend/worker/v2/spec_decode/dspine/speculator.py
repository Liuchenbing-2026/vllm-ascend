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
        if vllm_config.model_config.use_fp64_gumbel:
            raise ValueError("DSpine MRV2 on Ascend does not support use_fp64_gumbel=True")
        super().__init__(vllm_config, device)
