// Real Torch schema/Meta registration, separately buildable without CANN.
// No CPU arithmetic or replacement device implementation is registered.
#include "tq_packed_attention_contract.h"
#include <ATen/ops/empty.h>
#include <torch/library.h>
#include <cmath>
#include <limits>

namespace tq_packed_binding {
Outputs Allocate(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                 const at::Tensor &table, const at::Tensor &lengths, const at::Tensor &centroids,
                 double scale, int64_t apiScratchBytes, const std::string &opApiLibrary)
{
    TORCH_CHECK(query.dim() == 3 && query.size(2) == 128 && query.scalar_type() == at::kBFloat16,
                "TQ packed query must be BF16 [B,Hq,128]");
    TORCH_CHECK(key.dim() == 4 && key.size(3) == 66 && key.scalar_type() == at::kByte,
                "TQ packed key must be uint8 [P,S,Hkv,66]");
    TORCH_CHECK(value.sizes() == key.sizes() && value.scalar_type() == at::kByte,
                "TQ packed K/V must have identical uint8 shapes");
    TORCH_CHECK(table.dim() == 2 && table.size(0) == query.size(0) && table.scalar_type() == at::kInt,
                "TQ packed table must be int32 [B,W]");
    TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) == query.size(0) && lengths.scalar_type() == at::kInt,
                "TQ packed lengths must be int32 [B]");
    TORCH_CHECK(centroids.dim() == 1 && centroids.numel() == 16 && centroids.scalar_type() == at::kBFloat16,
                "TQ packed centroids must be BF16 [16]");
    for (const auto &tensor : {query, key, value, table, lengths, centroids}) {
        TORCH_CHECK(tensor.device() == query.device(), "TQ packed tensors must share a device");
        TORCH_CHECK(tensor.is_contiguous(), "TQ packed ABI requires contiguous ND tensors");
        for (int64_t dim : tensor.sizes()) {
            TORCH_CHECK(dim >= 0 && static_cast<uint64_t>(dim) <= UINT32_MAX,
                        "TQ packed dimensions must fit uint32");
        }
    }
    TORCH_CHECK(query.size(1) > 0 && key.size(0) > 0 && key.size(1) > 0 && key.size(2) > 0,
                "TQ packed head counts and physical cache dimensions must be positive");
    TORCH_CHECK(query.size(1) % key.size(2) == 0, "TQ packed query heads must be divisible by KV heads");
    TORCH_CHECK(query.size(0) == 0 || table.size(1) > 0, "TQ packed nonempty batch needs a page table");
    TORCH_CHECK(static_cast<uint64_t>(query.size(0)) * key.size(2) <= UINT32_MAX,
                "TQ packed task count exceeds uint32");
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)),
                "TQ packed scale must be representable as a finite float attribute");
    TORCH_CHECK(apiScratchBytes >= 0 && static_cast<uint64_t>(apiScratchBytes) <= UINT32_MAX,
                "TQ packed scratch reservation must fit uint32");
    TORCH_CHECK(!opApiLibrary.empty() && opApiLibrary.front() == '/',
                "TQ packed requires an explicit absolute OPP library path");
    // The library path configures PTA loading only, never a device/GE input.
    // Meta does not open files or read tensor contents, including lengths.
    return {at::empty(query.sizes(), query.options()),
            at::empty({query.size(0), key.size(2)}, query.options().dtype(at::kInt))};
}
}  // namespace tq_packed_binding

TORCH_LIBRARY_FRAGMENT(turboquant, m)
{
    m.def("tq_packed_attention(Tensor query_rot, Tensor key, Tensor value, Tensor table, "
          "Tensor lengths, Tensor centroids, float scale, int api_scratch_bytes, "
          "str op_api_library) -> (Tensor attention_rot, Tensor status)");
}
TORCH_LIBRARY_IMPL(turboquant, Meta, m)
{
    m.impl("tq_packed_attention", &tq_packed_binding::Allocate);
}
