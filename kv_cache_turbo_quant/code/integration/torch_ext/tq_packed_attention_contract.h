#ifndef TQ_PACKED_ATTENTION_CONTRACT_H
#define TQ_PACKED_ATTENTION_CONTRACT_H
#include <ATen/core/Tensor.h>
#include <string>
#include <tuple>

namespace tq_packed_binding {
using Outputs = std::tuple<at::Tensor, at::Tensor>;

Outputs Allocate(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                 const at::Tensor &table, const at::Tensor &lengths, const at::Tensor &centroids,
                 double scale, int64_t apiScratchBytes, const std::string &opApiLibrary);
}  // namespace tq_packed_binding
#endif
