
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_LIGHTNING_INDEXER_H_
#define ACLNN_INNER_LIGHTNING_INDEXER_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerLightningIndexerGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * weights : required
 * actualSeqLengthsQueryOptional : optional
 * actualSeqLengthsKeyOptional : optional
 * blockTableOptional : optional
 * layoutQueryOptional : optional
 * layoutKeyOptional : optional
 * sparseCount : optional
 * sparseMode : optional
 * preTokens : optional
 * nextTokens : optional
 * returnValues : optional
 * sparseIndicesOut : required
 * sparseValuesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerLightningIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *weights,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    char *layoutQueryOptional,
    char *layoutKeyOptional,
    int64_t sparseCount,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    bool returnValues,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerLightningIndexer
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerLightningIndexer(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
