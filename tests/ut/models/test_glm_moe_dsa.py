# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_ascend.models import deepseek_mtp
from vllm_ascend.models.deepseek_mtp import AscendGlmMoeDsaForCausalLM


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("pp_size,local_layers", [(1, 75), (2, 39), (2, 36), (2, 0), (3, 24)])
def test_eplb_layer_count_matches_local_weights_only_for_v2_pp(use_v2, pp_size, local_layers):
    config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        speculative_config=None,
    )
    original_count = 75 if local_layers else 0

    def init(self, *, vllm_config, prefix):
        nn.Module.__init__(self)
        assert vllm_config is config
        assert prefix == "target"
        self.num_moe_layers = original_count
        self.moe_layers = [object() for _ in range(local_layers)]
        self.model = nn.Module()
        self.model.embed_tokens = nn.Identity()

    with patch.object(GlmMoeDsaForCausalLM, "__init__", init):
        model = AscendGlmMoeDsaForCausalLM(vllm_config=config, prefix="target")

    expected = local_layers if use_v2 and pp_size > 1 else original_count
    assert model.num_moe_layers == expected
    assert len(model.moe_layers) == local_layers


@pytest.mark.parametrize(
    "needs_shared_embedding,initial_embedding,expect_created",
    [
        (True, PPMissingLayer(), True),
        (False, PPMissingLayer(), False),
        (True, nn.Identity(), False),
    ],
)
def test_dspark_provisions_embedding_on_last_target_stage(
    monkeypatch,
    needs_shared_embedding,
    initial_embedding,
    expect_created,
):
    config = SimpleNamespace(
        use_v2_model_runner=False,
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        quant_config=object(),
    )
    created = []

    def init(self, *, vllm_config, prefix):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(vocab_size=32, hidden_size=8)
        self.model = nn.Module()
        self.model.embed_tokens = initial_embedding

    def make_embedding(*args, **kwargs):
        embedding = nn.Embedding(args[0], args[1])
        created.append((embedding, kwargs))
        return embedding

    monkeypatch.setattr(GlmMoeDsaForCausalLM, "__init__", init)
    monkeypatch.setattr(deepseek_mtp, "VocabParallelEmbedding", make_embedding)
    monkeypatch.setattr(
        deepseek_mtp,
        "spec_decode_needs_target_embed",
        lambda received: received is config and needs_shared_embedding,
    )

    model = AscendGlmMoeDsaForCausalLM(vllm_config=config, prefix="target")

    assert bool(created) is expect_created
    if expect_created:
        embedding, kwargs = created[0]
        assert model.model.embed_tokens is embedding
        assert kwargs == {
            "quant_config": config.quant_config,
            "prefix": "target.model.embed_tokens",
        }
