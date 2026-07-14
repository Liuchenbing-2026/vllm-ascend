
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INPLACE_PARTIAL_ROTARY_MUL_H_
#define ACLNN_INPLACE_PARTIAL_ROTARY_MUL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInplacePartialRotaryMulGetWorkspaceSize
 * parameters :
 * xRef : required
 * cos : required
 * sin : required
 * mode : optional
 * partialSliceOptional : optional
 * xRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInplacePartialRotaryMulGetWorkspaceSize(
    aclTensor *xRef,
    const aclTensor *cos,
    const aclTensor *sin,
    int64_t mode,
    const aclIntArray *partialSliceOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInplacePartialRotaryMul
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInplacePartialRotaryMul(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
