/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_H
#define OP_API_ACLNN_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief FusedSigmoidGatingDeltaRuleUpdate 鐨勭涓€娈垫帴鍙ｏ紝鏍规嵁鍏蜂綋鐨勮绠楁祦绋嬶紝璁＄畻workspace澶у皬銆?
 * @param [in] query: 鏁版嵁绫诲瀷鏀寔锛歜float16銆?
 * @param [in] key: 鏁版嵁绫诲瀷鏀寔锛歜float16銆?
 * @param [in] value: 鏁版嵁绫诲瀷鏀寔锛歜float16銆?
 * @param [in] beta: 鏁版嵁绫诲瀷鏀寔锛歜float16銆?
 * @param [in] state: 鏁版嵁绫诲瀷鏀寔锛歜float16銆?
 * @param [in] actualSeqLengths: 鏁版嵁绫诲瀷鏀寔锛歩nt32銆?
 * @param [in] ssmStateIndices: 鏁版嵁绫诲瀷鏀寔锛歩nt32銆?
 * @param [in] g: 鏁版嵁绫诲瀷鏀寔锛歠loat32銆?
 * @param [in] gk: 鏁版嵁绫诲瀷鏀寔锛歠loat32銆?
 * @param [in] numAcceptedTokens: 鏁版嵁绫诲瀷鏀寔锛歩nt32銆?
 * @param [in] scaleValue: 鏁版嵁绫诲瀷鏀寔锛歠loat32銆?
 * @param [out] out: 鏁版嵁绫诲瀷鏀寔锛歜float16銆?
 * @param [out] 杩斿洖闇€瑕佸湪npu device渚х敵璇风殑workspace澶у皬銆?
 * @param [out] executor: 杩斿洖op鎵ц鍣紝鍖呭惈浜嗙畻瀛愯绠楁祦绋嬨€?
 * @return aclnnStatus: 杩斿洖鐘舵€佺爜
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedSigmoidGatingDeltaRuleUpdateGetWorkspaceSize(
    const aclTensor *aLog, const aclTensor *a, const aclTensor *b, const aclTensor *dtBias,
    const aclTensor *query, const aclTensor *key, const aclTensor *value, aclTensor *stateRef,
    const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices, const aclTensor *numAcceptedTokens,
    float scaleValue, float softplusBeta, float softplusThreshold, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief 
 * @param [in] workspace: 鍦╪pu device渚х敵璇风殑workspace鍐呭瓨璧峰潃銆?
 * @param [in] workspace_size: 鍦╪pu
 * device渚х敵璇风殑workspace澶у皬锛岀敱绗竴娈垫帴鍙clnnFusedSigmoidGatingDeltaRuleUpdateGetWorkspaceSize鑾峰彇銆?
 * @param [in] executor: op鎵ц鍣紝鍖呭惈浜嗙畻瀛愯绠楁祦绋嬨€?
 * @param [in] stream: acl stream娴併€?
 * @return aclnnStatus: 杩斿洖鐘舵€佺爜
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedSigmoidGatingDeltaRuleUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_FUSED_SIGMOID_GATING_DELTA_RULE_UPDATE_H
