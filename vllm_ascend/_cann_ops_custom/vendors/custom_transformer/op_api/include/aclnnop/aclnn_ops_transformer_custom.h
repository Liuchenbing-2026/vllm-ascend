/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_OPS_TRANSFORMER_CUSTOM_H_
#define ACLNN_OPS_TRANSFORMER_CUSTOM_H_

#include "aclnn_add_rms_norm_bias.h"
#include "aclnn_causal_conv1d.h"
#include "aclnn_copy_and_expand_eagle_inputs.h"
#include "aclnn_dequant_swiglu_quant.h"
#include "aclnn_dequant_swiglu_quant_v2.h"
#include "aclnn_hamming_dist_top_k.h"
#include "aclnn_hc_post.h"
#include "aclnn_hc_pre.h"
#include "aclnn_hc_pre_inv_rms.h"
#include "aclnn_hc_pre_sinkhorn.h"
#include "aclnn_moe_gating_top_k.h"
#include "aclnn_moe_gating_top_k_hash.h"
#include "aclnn_moe_init_routing_custom.h"
#include "aclnn_transpose_kv_cache_by_block.h"
#include "aclnn_compressor.h"
#include "aclnn_compressor_metadata.h"
#include "aclnn_inplace_partial_rotary_mul.h"
#include "aclnn_lightning_indexer_quant.h"
#include "aclnn_rms_norm_dynamic_quant.h"
#include "aclnn_sparse_attn_sharedkv.h"
#include "aclnn_store_kv_block.h"
#include "aclnn_vllm_quant_lightning_indexer.h"
#include "aclnnInner_matmul_allreduce_add_rmsnorm.h"
#include "aclnnInner_moe_grouped_matmul.h"
#include "aclnnInner_lightning_indexer.h"
#include "aclnnInner_ngram_spec_decode.h"
#include "aclnnInner_sparse_flash_attention.h"
#include "aclnn_grouped_matmul_swiglu_quant.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz.h"
#include "aclnn_grouped_matmul_swiglu_quant_v2.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_tensor_list.h"
#include "aclnn_matmul_allreduce_add_rmsnorm.h"
#include "aclnn_apply_top_k_top_p_custom.h"
#include "aclnn_chunk_fwd_o.h"
#include "aclnn_chunk_gated_delta_rule_fwd_h.h"
#include "aclnn_moe_grouped_matmul.h"
#include "aclnn_moe_grouped_matmul_weight_nz.h"
#include "aclnn_scatter_nd_update_v2.h"
#include "aclnn_fused_gdn_gating.h"
#include "aclnn_lightning_indexer.h"
#include "aclnn_ngram_spec_decode.h"
#include "aclnn_recurrent_gated_delta_rule.h"
#include "aclnn_sparse_attn_sharedkv_metadata.h"
#include "aclnn_sparse_flash_attention.h"
#include "aclnn_store_kv_block_metadata.h"
#include "aclnn_vllm_quant_lightning_indexer_metadata.h"

#endif // ACLNN_OPS_TRANSFORMER_CUSTOM_H_
