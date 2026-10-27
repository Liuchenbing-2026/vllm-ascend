# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UNO tree GQA attention using paged BSH FIA-v2 with an explicit mask."""

import torch_npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def forward_uno_tree_attention(impl, query, key, value, metadata, output):
    key, value, block_size, table, seq_lens = impl._get_fia_params(key, value, metadata)
    rows = metadata.num_actual_tokens
    if len(seq_lens) != 1:
        raise ValueError("UNO tree attention currently requires a single request.")
    query_bsh = query[:rows].reshape(1, rows, -1)
    output_bsh = output[:rows].view(1, rows, -1)
    if table is None:
        key = key.reshape(1, rows, -1)
        value = value.reshape(1, rows, -1)
    kwargs = dict(
        query=query_bsh,
        key=key,
        value=value,
        block_table=table,
        atten_mask=metadata.uno_tree_mask,
        input_layout="BSH",
        block_size=block_size,
        actual_seq_qlen=[rows],
        actual_seq_kvlen=seq_lens,
        num_query_heads=impl.num_heads,
        num_key_value_heads=impl.num_kv_heads,
        softmax_scale=impl.scale,
        sparse_mode=0,
    )
    if not _EXTRA_CTX.capturing:
        result, _ = torch_npu.npu_fused_infer_attention_score_v2(**kwargs)
        output_bsh.copy_(result)
        return output

    raise NotImplementedError(
        "UNO tree attention requires enforce_eager: the Ascend attention graph path "
        "rebuilds per-layer parameters through UpdatableGraph, which cannot replay the "
        "task-group entry this kernel registers."
    )
