
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_COMPRESSOR_H_
#define ACLNN_COMPRESSOR_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCompressorGetWorkspaceSize
 * parameters :
 * x : required
 * wkv : required
 * wgate : required
 * stateCacheRef : required
 * ape : required
 * normWeight : required
 * ropeSin : required
 * ropeCos : required
 * stateBlockTableOptional : optional
 * cuSeqlensOptional : optional
 * sequsedOptional : optional
 * startPosOptional : optional
 * ropeHeadDim : required
 * cmpRatio : required
 * coff : optional
 * normEps : optional
 * rotaryMode : optional
 * cacheMode : optional
 * stateCacheStrideDim0 : optional
 * cmpKvOut : required
 * stateCacheRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCompressorGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *wkv,
    const aclTensor *wgate,
    aclTensor *stateCacheRef,
    const aclTensor *ape,
    const aclTensor *normWeight,
    const aclTensor *ropeSin,
    const aclTensor *ropeCos,
    const aclTensor *stateBlockTableOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    const aclTensor *startPosOptional,
    int64_t ropeHeadDim,
    int64_t cmpRatio,
    int64_t coff,
    double normEps,
    int64_t rotaryMode,
    int64_t cacheMode,
    int64_t stateCacheStrideDim0,
    const aclTensor *cmpKvOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCompressor
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCompressor(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
