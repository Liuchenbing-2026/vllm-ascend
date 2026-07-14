
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MOE_GROUPED_MATMUL_H_
#define ACLNN_INNER_MOE_GROUPED_MATMUL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMoeGroupedMatmulGetWorkspaceSize
 * parameters :
 * x : dynamic
 * weight : dynamic
 * groupList : required
 * transposeWeight : optional
 * out : dynamic
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeGroupedMatmulGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensorList *weight,
    const aclTensor *groupList,
    bool transposeWeight,
    const aclTensorList *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMoeGroupedMatmul
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMoeGroupedMatmul(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
