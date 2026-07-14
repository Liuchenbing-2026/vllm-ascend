
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_COMPRESSOR_METADATA_H_
#define ACLNN_COMPRESSOR_METADATA_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCompressorMetadataGetWorkspaceSize
 * parameters :
 * ropeCos : required
 * ropeSin : required
 * cuSeqlens : required
 * startPos : required
 * kvBlockTable : required
 * kvBlockSize : required
 * slotMappingFormat : required
 * cmpRatio : required
 * actualNumReqs : required
 * compressCosOut : required
 * compressSinOut : required
 * slotMappingOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCompressorMetadataGetWorkspaceSize(
    const aclTensor *ropeCos,
    const aclTensor *ropeSin,
    const aclTensor *cuSeqlens,
    const aclTensor *startPos,
    const aclTensor *kvBlockTable,
    int64_t kvBlockSize,
    int64_t slotMappingFormat,
    int64_t cmpRatio,
    int64_t actualNumReqs,
    const aclTensor *compressCosOut,
    const aclTensor *compressSinOut,
    const aclTensor *slotMappingOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCompressorMetadata
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCompressorMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
