#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import pytest
import torch

from vllm_ascend.patch.worker import patch_draft_quarot
from vllm_ascend.patch.worker.patch_draft_quarot import (
    make_qwen3_dspark_load_weights,
    rotate_concatenated_fc_weight,
)


def test_rotate_concatenated_fc_weight_rotates_each_hidden_block():
    weight = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    actual = rotate_concatenated_fc_weight(weight, rotation)
    expected = torch.cat(
        [weight[:, start : start + 2] @ rotation for start in range(0, 6, 2)],
        dim=1,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("weight_shape", "rotation_shape"),
    [((3, 6), (2, 3)), ((3, 5), (2, 2))],
)
def test_rotate_concatenated_fc_weight_rejects_invalid_shapes(
    weight_shape, rotation_shape
):
    with pytest.raises(ValueError):
        rotate_concatenated_fc_weight(
            torch.empty(weight_shape), torch.empty(rotation_shape)
        )


def test_qwen3_dspark_loader_rotates_only_fc_weight(monkeypatch):
    rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    monkeypatch.setattr(patch_draft_quarot, "get_rotataion_matrix", lambda _: rotation)
    captured = {}

    def original_load_weights(_, weights):
        captured.update(weights)
        return {"loaded"}

    fake_model = type(
        "FakeDSparkModel",
        (),
        {"model": type("Inner", (), {"fc": type("FC", (), {"weight": torch.empty(1)})()})()},
    )()
    weight = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    other = torch.ones(2)
    load_weights = make_qwen3_dspark_load_weights(original_load_weights, "rotation")

    result = load_weights(fake_model, [("fc.weight", weight), ("other", other)])

    expected = torch.cat(
        [weight[:, start : start + 2] @ rotation for start in range(0, 6, 2)],
        dim=1,
    )
    assert result == {"loaded"}
    torch.testing.assert_close(captured["fc.weight"], expected)
    assert captured["other"] is other
