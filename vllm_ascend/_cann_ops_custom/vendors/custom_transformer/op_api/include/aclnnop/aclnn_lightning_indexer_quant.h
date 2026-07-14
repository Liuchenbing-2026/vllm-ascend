
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LIGHTNING_INDEXER_QUANT_H_
#define ACLNN_LIGHTNING_INDEXER_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLightningIndexerQuantGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * weights : required
 * queryDequantScale : required
 * keyDequantScale : required
 * actualSeqLengthsQueryOptional : optional
 * actualSeqLengthsKeyOptional : optional
 * blockTableOptional : optional
 * queryQuantMode : required
 * keyQuantMode : required
 * layoutQueryOptional : optional
 * layoutKeyOptional : optional
 * sparseCount : optional
 * sparseMode : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLightningIndexerQuantGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *queryDequantScale,
    const aclTensor *keyDequantScale,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    int64_t queryQuantMode,
    int64_t keyQuantMode,
    char *layoutQueryOptional,
    char *layoutKeyOptional,
    int64_t sparseCount,
    int64_t sparseMode,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLightningIndexerQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLightningIndexerQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
