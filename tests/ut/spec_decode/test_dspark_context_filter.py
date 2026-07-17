from types import SimpleNamespace

import torch

from vllm_ascend.spec_decode.dspark_proposer import AscendDsparkProposer


class _ContextRecorder:
    def precompute_and_store_context_kv(self, hidden_states, positions, slot_mapping):
        self.hidden_states = hidden_states
        self.positions = positions
        self.slot_mapping = slot_mapping


def test_dspark_context_kv_filters_rejected_rows():
    proposer = SimpleNamespace(
        _dflash_num_context=4,
        _dflash_hidden_states=torch.arange(16, dtype=torch.float32).view(4, 4),
        _context_positions_buffer=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        _context_slot_mapping_buffer=torch.tensor([20, -1, 22, -1], dtype=torch.int32),
        input_ids=torch.zeros(8, dtype=torch.int32),
        positions=torch.zeros(8, dtype=torch.int32),
        model=_ContextRecorder(),
    )

    model_inputs = AscendDsparkProposer.build_model_inputs_first_pass(
        proposer,
        num_input_tokens=8,
    )

    assert proposer.model.hidden_states.tolist() == [[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]]
    assert proposer.model.positions.tolist() == [10, 12]
    assert proposer.model.slot_mapping.tolist() == [20, 22]
    assert model_inputs["input_ids"].shape[0] == 8
