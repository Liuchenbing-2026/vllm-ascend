
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_HAMMING_DIST_TOP_K_H_
#define ACLNN_HAMMING_DIST_TOP_K_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnHammingDistTopKGetWorkspaceSize
 * parameters :
 * query : required
 * keyCompressed : required
 * k : required
 * seqLen : required
 * chunkSize : required
 * keyBlockTable : required
 * indicesIn : required
 * keyCompressedRopeOptional : optional
 * maskOptional : optional
 * maxSeqLen : optional
 * sink : optional
 * recent : optional
 * supportOffload : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHammingDistTopKGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *keyCompressed,
    const aclTensor *k,
    const aclTensor *seqLen,
    const aclTensor *chunkSize,
    const aclTensor *keyBlockTable,
    const aclTensor *indicesIn,
    const aclTensor *keyCompressedRopeOptional,
    const aclTensor *maskOptional,
    int64_t maxSeqLen,
    int64_t sink,
    int64_t recent,
    int64_t supportOffload,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnHammingDistTopK
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHammingDistTopK(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
