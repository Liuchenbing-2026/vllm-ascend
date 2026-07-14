
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_INIT_ROUTING_CUSTOM_H_
#define ACLNN_MOE_INIT_ROUTING_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeInitRoutingCustomGetWorkspaceSize
 * parameters :
 * x : required
 * expertIdx : required
 * scaleOptional : optional
 * offsetOptional : optional
 * activeNum : optional
 * expertCapacity : optional
 * expertNum : optional
 * dropPadMode : optional
 * expertTokensNumType : optional
 * expertTokensNumFlag : optional
 * quantMode : optional
 * activeExpertRangeOptional : optional
 * rowIdxType : optional
 * expandedXOut : required
 * expandedRowIdxOut : required
 * expertTokensCountOrCumsumOut : required
 * expandedScaleOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeInitRoutingCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *expertIdx,
    const aclTensor *scaleOptional,
    const aclTensor *offsetOptional,
    int64_t activeNum,
    int64_t expertCapacity,
    int64_t expertNum,
    int64_t dropPadMode,
    int64_t expertTokensNumType,
    bool expertTokensNumFlag,
    int64_t quantMode,
    const aclIntArray *activeExpertRangeOptional,
    int64_t rowIdxType,
    const aclTensor *expandedXOut,
    const aclTensor *expandedRowIdxOut,
    const aclTensor *expertTokensCountOrCumsumOut,
    const aclTensor *expandedScaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeInitRoutingCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeInitRoutingCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
