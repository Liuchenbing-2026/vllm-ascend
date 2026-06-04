/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#ifndef FUSED_KV_NORM_ROPE_SWA_CACHE_CUSTOM_TORCH_ADPT_H
#define FUSED_KV_NORM_ROPE_SWA_CACHE_CUSTOM_TORCH_ADPT_H

namespace vllm_ascend {

// DSV4-DSA SWA-kv prep fusion (replaces dsa_v1.py:1907-1920 kv trio: kv_norm + partial
// interleave RoPE on [nope:head_dim] + scatter-nd write into swa_kv_cache).
//   kv_in        [nt, head_dim]            fp16/bf16 (post-wkv, pre-norm)
//   gamma        [head_dim]                fp16/bf16 (kv_norm.weight)
//   cos / sin    [nt, rope_dim]            fp32, pre-gathered + pair-repeated
//   slot_mapping [nt, 2]                   int32 (block_idx, block_offset)
//   kv_cache     [num_blocks, bs, head_dim] fp16/bf16, mutated in place
// Returns kv_cache; the normalized+roped value is written only to cache.
at::Tensor npu_fused_kv_norm_rope_swa_cache(
    const at::Tensor& kv_in,
    const at::Tensor& gamma,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& slot_mapping,
    at::Tensor& kv_cache,
    double epsilon)
{
    // aclnn arg order = inputs..., attr..., outputs... (kv_cache is both input[5] and output[0], in place)
    EXEC_NPU_CMD(aclnnFusedKvNormRopeSwaCacheCustom,
                 kv_in, gamma, cos, sin, slot_mapping, kv_cache,
                 epsilon,
                 kv_cache);
    return kv_cache;
}
}  // namespace vllm_ascend
#endif
