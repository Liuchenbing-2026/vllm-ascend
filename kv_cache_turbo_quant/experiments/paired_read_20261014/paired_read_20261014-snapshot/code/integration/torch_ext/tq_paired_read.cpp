#include <dlfcn.h>
#include <vector>
#include <tuple>
#include <torch/extension.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {
constexpr const char *OP_API = "/ws/paired_read_20261014/opp/vendors/tqreadpair/op_api/lib/libcust_opapi.so";
using WorkspaceFn = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
using RunFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

struct Api {
    WorkspaceFn workspace;
    RunFn run;
};
const Api &GetApi()
{
    static const Api api = [] {
        void *handle = dlopen(OP_API, RTLD_NOW | RTLD_GLOBAL);
        TORCH_CHECK(handle != nullptr, "native read dlopen failed: ", dlerror());
        Api result{reinterpret_cast<WorkspaceFn>(dlsym(handle, "aclnnTqPagedReadPairGetWorkspaceSize")),
            reinterpret_cast<RunFn>(dlsym(handle, "aclnnTqPagedReadPair"))};
        TORCH_CHECK(result.workspace != nullptr && result.run != nullptr, "native read symbols missing");
        return result;
    }();
    return api;
}

aclTensor *Wrap(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.is_contiguous(), "native read requires contiguous inputs");
    aclDataType dtype;
    switch (tensor.scalar_type()) {
        case at::kByte: dtype = ACL_UINT8; break;
        case at::kInt: dtype = ACL_INT32; break;
        case at::kBFloat16: dtype = ACL_BF16; break;
        default: TORCH_CHECK(false, "native read unsupported dtype");
    }
    const auto sizes = tensor.sizes();
    return aclCreateTensor(sizes.data(), sizes.size(), dtype, nullptr, 0, ACL_FORMAT_ND,
        sizes.data(), sizes.size(), tensor.data_ptr());
}

std::tuple<at::Tensor, at::Tensor> Read(const at::Tensor &cache, const at::Tensor &value, const at::Tensor &table, const at::Tensor &jobs,
    const at::Tensor &lookup, int64_t tokens)
{
    TORCH_CHECK(cache.dim() == 4 && cache.size(3) == 66 && cache.scalar_type() == at::kByte,
        "native read requires original 66-byte cache");
    TORCH_CHECK(value.sizes() == cache.sizes() && value.scalar_type() == cache.scalar_type() && value.device() == cache.device(), "paired value cache mismatch");
    TORCH_CHECK(table.dim() == 2 && jobs.dim() == 2 && jobs.size(1) == 4,
        "native read metadata shape mismatch");
    TORCH_CHECK(table.scalar_type() == at::kInt && jobs.scalar_type() == at::kInt &&
        lookup.scalar_type() == at::kBFloat16 && lookup.numel() == 5696, "native read metadata dtype mismatch");
    TORCH_CHECK(tokens >= 0, "negative native read token count");
    TORCH_CHECK(cache.device() == table.device() && cache.device() == jobs.device() &&
        cache.device() == lookup.device(), "native read device mismatch");
    auto output = at::empty({tokens, cache.size(2), 128}, cache.options().dtype(at::kBFloat16));
    auto valueOutput = at::empty_like(output);
    if (tokens == 0) {
        return std::make_tuple(output, valueOutput);
    }
    std::vector<aclTensor *> tensors{Wrap(cache), Wrap(value), Wrap(table), Wrap(jobs), Wrap(lookup), Wrap(output), Wrap(valueOutput)};
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    const Api &api = GetApi();
    aclnnStatus status = api.workspace(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tokens,
        tensors[5], tensors[6], &bytes, &executor);
    if (status != ACL_SUCCESS) {
        for (auto *tensor : tensors) aclDestroyTensor(tensor);
        TORCH_CHECK(false, "native read workspace failed: ", static_cast<int>(status));
    }
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, cache.options());
    status = api.run(bytes ? workspace.data_ptr() : nullptr, bytes, executor,
        c10_npu::getCurrentNPUStream().stream());
    for (auto *tensor : tensors) aclDestroyTensor(tensor);
    TORCH_CHECK(status == ACL_SUCCESS, "native read execute failed: ", static_cast<int>(status));
    return std::make_tuple(output, valueOutput);
}

std::tuple<at::Tensor, at::Tensor> ReadMeta(const at::Tensor &cache, const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, int64_t tokens)
{
    auto output = at::empty({tokens, cache.size(2), 128}, cache.options().dtype(at::kBFloat16));
    return std::make_tuple(output, at::empty_like(output));
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
TORCH_LIBRARY_FRAGMENT(turboquant, m)
{
    m.def("tq_paged_read_pair(Tensor cache, Tensor value, Tensor table, Tensor jobs, Tensor lookup, int tokens) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(turboquant, PrivateUse1, m)
{
    m.impl("tq_paged_read_pair", &Read);
}
TORCH_LIBRARY_IMPL(turboquant, Meta, m)
{
    m.impl("tq_paged_read_pair", &ReadMeta);
}
