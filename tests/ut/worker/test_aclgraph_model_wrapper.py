# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext


class _DFlash2DraftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Identity()

    def compute_candidates(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states + 1, hidden_states - 1


def test_model_with_context_forwards_dflash2_model_interface() -> None:
    draft_model = _DFlash2DraftModel()
    wrapped_model = ModelWithContext(draft_model, is_draft_model=True)
    hidden_states = torch.tensor([[1.0, 2.0]])

    candidate_ids, unary_logits = wrapped_model.compute_candidates(hidden_states)

    assert wrapped_model.model is draft_model.model
    torch.testing.assert_close(candidate_ids, hidden_states + 1)
    torch.testing.assert_close(unary_logits, hidden_states - 1)
