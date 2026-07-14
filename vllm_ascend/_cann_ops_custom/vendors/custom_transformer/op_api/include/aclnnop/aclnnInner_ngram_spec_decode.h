
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_NGRAM_SPEC_DECODE_H_
#define ACLNN_INNER_NGRAM_SPEC_DECODE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnInnerNgramSpecDecodeGetWorkspaceSize
 * parameters :
 * tokenIds : required
 * numTokensNoSpec : required
 * sampledTokenIds : required
 * discardRequestMask : required
 * vocabSize : required
 * minN : required
 * maxN : required
 * k : required
 * nextTokenIdsOut : required
 * draftTokenIdsOut : required
 * numValidDraftTokensOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerNgramSpecDecodeGetWorkspaceSize(
    const aclTensor *tokenIds,
    const aclTensor *numTokensNoSpec,
    const aclTensor *sampledTokenIds,
    const aclTensor *discardRequestMask,
    int64_t vocabSize,
    int64_t minN,
    int64_t maxN,
    int64_t k,
    const aclTensor *nextTokenIdsOut,
    const aclTensor *draftTokenIdsOut,
    const aclTensor *numValidDraftTokensOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerNgramSpecDecode
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerNgramSpecDecode(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
