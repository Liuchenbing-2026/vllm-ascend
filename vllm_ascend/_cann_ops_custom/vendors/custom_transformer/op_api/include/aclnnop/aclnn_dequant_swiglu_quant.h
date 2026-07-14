
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DEQUANT_SWIGLU_QUANT_H_
#define ACLNN_DEQUANT_SWIGLU_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDequantSwigluQuantGetWorkspaceSize
 * parameters :
 * x : required
 * weightScaleOptional : optional
 * activationScaleOptional : optional
 * biasOptional : optional
 * quantScaleOptional : optional
 * quantOffsetOptional : optional
 * groupIndexOptional : optional
 * activateLeft : optional
 * quantModeOptional : optional
 * yOut : required
 * scaleOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDequantSwigluQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightScaleOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *biasOptional,
    const aclTensor *quantScaleOptional,
    const aclTensor *quantOffsetOptional,
    const aclTensor *groupIndexOptional,
    bool activateLeft,
    char *quantModeOptional,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDequantSwigluQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDequantSwigluQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
