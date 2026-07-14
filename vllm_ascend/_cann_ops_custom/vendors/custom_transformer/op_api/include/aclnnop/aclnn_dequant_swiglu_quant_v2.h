
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DEQUANT_SWIGLU_QUANT_V2_H_
#define ACLNN_DEQUANT_SWIGLU_QUANT_V2_H_
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDequantSwigluQuantV2GetWorkspaceSize
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
 * dstType : optional
 * roundModeOptional : optional
 * activateDim : optional
 * swigluMode : optional
 * clampLimit : optional
 * gluAlpha : optional
 * gluBias : optional
 * yOut : required
 * scaleOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDequantSwigluQuantV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightScaleOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *biasOptional,
    const aclTensor *quantScaleOptional,
    const aclTensor *quantOffsetOptional,
    const aclTensor *groupIndexOptional,
    bool activateLeft,
    char *quantModeOptional,
    int64_t dstType,
    char *roundModeOptional,
    int64_t activateDim,
    int64_t swigluMode,
    double clampLimit,
    double gluAlpha,
    double gluBias,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDequantSwigluQuantV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDequantSwigluQuantV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
