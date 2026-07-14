
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_HC_POST_H_
#define ACLNN_HC_POST_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnHcPostGetWorkspaceSize
 * parameters :
 * x : required
 * residual : required
 * post : required
 * comb : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHcPostGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *residual,
    const aclTensor *post,
    const aclTensor *comb,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnHcPost
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHcPost(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
