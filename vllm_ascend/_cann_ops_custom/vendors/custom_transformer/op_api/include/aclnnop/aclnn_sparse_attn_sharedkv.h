
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SPARSE_ATTN_SHAREDKV_H_
#define ACLNN_SPARSE_ATTN_SHAREDKV_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSparseAttnSharedkvGetWorkspaceSize
 * parameters :
 * q : required
 * oriKvOptional : optional
 * cmpKvOptional : optional
 * oriSparseIndicesOptional : optional
 * cmpSparseIndicesOptional : optional
 * oriBlockTableOptional : optional
 * cmpBlockTableOptional : optional
 * cuSeqlensQOptional : optional
 * cuSeqlensOriKvOptional : optional
 * cuSeqlensCmpKvOptional : optional
 * sequsedQOptional : optional
 * sequsedKvOptional : optional
 * sinksOptional : optional
 * metadataOptional : optional
 * softmaxScale : required
 * cmpRatio : required
 * oriMaskMode : required
 * cmpMaskMode : required
 * oriKvStride : required
 * cmpKvStride : required
 * oriWinLeft : optional
 * oriWinRight : optional
 * layoutQOptional : optional
 * layoutKvOptional : optional
 * returnSoftmaxLse : optional
 * attnOutOut : required
 * softmaxLseOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseAttnSharedkvGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *oriKvOptional,
    const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional,
    const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional,
    const aclTensor *cmpBlockTableOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    double softmaxScale,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriKvStride,
    int64_t cmpKvStride,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    char *layoutQOptional,
    char *layoutKvOptional,
    bool returnSoftmaxLse,
    const aclTensor *attnOutOut,
    const aclTensor *softmaxLseOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSparseAttnSharedkv
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseAttnSharedkv(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
