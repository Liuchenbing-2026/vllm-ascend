
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_MATMUL_ALLREDUCE_ADD_RMSNORM_H_
#define ACLNN_INNER_MATMUL_ALLREDUCE_ADD_RMSNORM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerMatmulAllreduceAddRmsnormGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * residual : required
 * gamma : required
 * groupTp : required
 * tpRankSize : required
 * tpRankId : required
 * epsilon : optional
 * isTransB : optional
 * isGatherAddOut : optional
 * yOut : required
 * addOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMatmulAllreduceAddRmsnormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *residual,
    const aclTensor *gamma,
    char *groupTp,
    int64_t tpRankSize,
    int64_t tpRankId,
    double epsilon,
    bool isTransB,
    bool isGatherAddOut,
    const aclTensor *yOut,
    const aclTensor *addOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerMatmulAllreduceAddRmsnorm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerMatmulAllreduceAddRmsnorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
