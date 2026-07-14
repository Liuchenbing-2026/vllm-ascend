
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_HC_PRE_INV_RMS_H_
#define ACLNN_HC_PRE_INV_RMS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnHcPreInvRmsGetWorkspaceSize
 * parameters :
 * x : required
 * epsilon : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHcPreInvRmsGetWorkspaceSize(
    const aclTensor *x,
    double epsilon,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnHcPreInvRms
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHcPreInvRms(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
