
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_VLLM_QUANT_LIGHTNING_INDEXER_H_
#define ACLNN_VLLM_QUANT_LIGHTNING_INDEXER_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnVllmQuantLightningIndexerGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * weights : required
 * queryDequantScale : required
 * keyDequantScale : required
 * actualSeqLengthsQueryOptional : optional
 * actualSeqLengthsKeyOptional : optional
 * blockTableOptional : optional
 * metadataOptional : optional
 * queryQuantMode : required
 * keyQuantMode : required
 * layoutQueryOptional : optional
 * layoutKeyOptional : optional
 * sparseCount : optional
 * sparseMode : optional
 * preTokens : optional
 * nextTokens : optional
 * cmpRatio : optional
 * returnValues : optional
 * stride : optional
 * scaleStride : optional
 * sparseIndicesOut : required
 * sparseValuesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVllmQuantLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *metadataOptional,
    int64_t queryQuantMode,
    int64_t keyQuantMode,
    char *layoutQueryOptional,
    char *layoutKeyOptional,
    int64_t sparseCount,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t cmpRatio,
    bool returnValues,
    int64_t stride,
    int64_t scaleStride,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnVllmQuantLightningIndexer
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVllmQuantLightningIndexer(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
