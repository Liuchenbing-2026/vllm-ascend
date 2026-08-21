# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)


class AscendDFlash2Speculator(DFlash2Speculator, AscendDFlashSpeculator):
    """Run the upstream DFlash2 selector through the Ascend DFlash path.

    The cooperative MRO keeps DFlash2's candidate generation and path walk,
    while reusing Ascend's attention metadata, input preparation, and ACLGraph
    integration from ``AscendDFlashSpeculator``.
    """
