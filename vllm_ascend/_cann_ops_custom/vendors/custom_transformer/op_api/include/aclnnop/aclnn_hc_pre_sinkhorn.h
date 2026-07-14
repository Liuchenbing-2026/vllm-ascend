
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_HC_PRE_SINKHORN_H_
#define ACLNN_HC_PRE_SINKHORN_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnHcPreSinkhornGetWorkspaceSize
 * parameters :
 * mixes : required
 * rsqrt : required
 * hcScale : required
 * hcBase : required
 * x : required
 * hcMult : optional
 * hcSinkhornIters : optional
 * hcEps : optional
 * yOut : required
 * postOut : required
 * combFragOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHcPreSinkhornGetWorkspaceSize(
    const aclTensor *mixes,
    const aclTensor *rsqrt,
    const aclTensor *hcScale,
    const aclTensor *hcBase,
    const aclTensor *x,
    int64_t hcMult,
    int64_t hcSinkhornIters,
    double hcEps,
    const aclTensor *yOut,
    const aclTensor *postOut,
    const aclTensor *combFragOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnHcPreSinkhorn
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHcPreSinkhorn(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
