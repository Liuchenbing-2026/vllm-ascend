
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_STORE_KVBLOCK_H_
#define ACLNN_STORE_KVBLOCK_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnStoreKVBlockGetWorkspaceSize
 * parameters :
 * keyIn : required
 * keyCacheIn : required
 * groupLen : required
 * groupKeyIdx : required
 * groupKeyCacheIdx : required
 * blockSize : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnStoreKVBlockGetWorkspaceSize(
    const aclTensor *keyIn,
    const aclTensor *keyCacheIn,
    const aclTensor *groupLen,
    const aclTensor *groupKeyIdx,
    const aclTensor *groupKeyCacheIdx,
    int64_t blockSize,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnStoreKVBlock
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnStoreKVBlock(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
