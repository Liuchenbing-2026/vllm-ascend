
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_TRANSPOSE_KV_CACHE_BY_BLOCK_H_
#define ACLNN_TRANSPOSE_KV_CACHE_BY_BLOCK_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnTransposeKvCacheByBlockGetWorkspaceSize
 * parameters :
 * kCache : dynamic
 * vCache : dynamic
 * blockIDs : required
 * blockSize : required
 * headNum : required
 * headDim : required
 * splitNum : required
 * layerNum : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTransposeKvCacheByBlockGetWorkspaceSize(
    const aclTensorList *kCache,
    const aclTensorList *vCache,
    const aclTensor *blockIDs,
    int64_t blockSize,
    int64_t headNum,
    int64_t headDim,
    int64_t splitNum,
    int64_t layerNum,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnTransposeKvCacheByBlock
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTransposeKvCacheByBlock(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
