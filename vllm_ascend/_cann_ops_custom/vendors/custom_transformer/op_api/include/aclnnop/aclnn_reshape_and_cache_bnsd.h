
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RESHAPE_AND_CACHE_BNSD_H_
#define ACLNN_RESHAPE_AND_CACHE_BNSD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnReshapeAndCacheBnsdGetWorkspaceSize
 * parameters :
 * keyIn : required
 * keyCacheIn : required
 * slotMapping : required
 * seqLen : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnReshapeAndCacheBnsdGetWorkspaceSize(
    const aclTensor *keyIn,
    const aclTensor *keyCacheIn,
    const aclTensor *slotMapping,
    const aclTensor *seqLen,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnReshapeAndCacheBnsd
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnReshapeAndCacheBnsd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
