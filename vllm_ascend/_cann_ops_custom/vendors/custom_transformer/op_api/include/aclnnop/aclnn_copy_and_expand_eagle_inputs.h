
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_COPY_AND_EXPAND_EAGLE_INPUTS_H_
#define ACLNN_COPY_AND_EXPAND_EAGLE_INPUTS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCopyAndExpandEagleInputsGetWorkspaceSize
 * parameters :
 * targetTokenIds : required
 * targetPositions : required
 * nextTokenIds : required
 * queryStartLoc : required
 * queryEndLoc : required
 * paddingTokenId : required
 * parallelDraftingTokenId : required
 * numPaddingSlotsPerRequest : required
 * shiftInputIds : required
 * totalInputTokens : required
 * outInputIdsOut : required
 * outPositionsOut : required
 * outIsRejectedTokenMaskOut : required
 * outIsMaskedTokenMaskOut : required
 * outNewTokenIndicesOut : required
 * outHiddenStateMappingOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCopyAndExpandEagleInputsGetWorkspaceSize(
    const aclTensor *targetTokenIds,
    const aclTensor *targetPositions,
    const aclTensor *nextTokenIds,
    const aclTensor *queryStartLoc,
    const aclTensor *queryEndLoc,
    int64_t paddingTokenId,
    int64_t parallelDraftingTokenId,
    int64_t numPaddingSlotsPerRequest,
    bool shiftInputIds,
    int64_t totalInputTokens,
    const aclTensor *outInputIdsOut,
    const aclTensor *outPositionsOut,
    const aclTensor *outIsRejectedTokenMaskOut,
    const aclTensor *outIsMaskedTokenMaskOut,
    const aclTensor *outNewTokenIndicesOut,
    const aclTensor *outHiddenStateMappingOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCopyAndExpandEagleInputs
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCopyAndExpandEagleInputs(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
