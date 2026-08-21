from vllm_ascend.patch.platform.patch_speculative_config import _update_dflash


def _dflash_config() -> dict:
    return {
        "aux_hidden_state_layer_ids": [1],
        "draft_vocab_size": 32,
        "mask_token_id": 31,
        "target_hidden_size": 16,
    }


def test_dflash2_architecture_is_preserved() -> None:
    pretrained_config = {"architectures": ["DFlash2DraftModel"]}

    _update_dflash(_dflash_config(), pretrained_config)

    assert pretrained_config["architectures"] == ["DFlash2DraftModel"]
    assert pretrained_config["dflash_config"]["mask_token_id"] == 31


def test_plain_dflash_keeps_upstream_default() -> None:
    pretrained_config = {"architectures": ["Qwen3ForCausalLM"]}

    _update_dflash(_dflash_config(), pretrained_config)

    assert pretrained_config["architectures"] == ["DFlashDraftModel"]
