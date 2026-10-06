/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Torch glue that exposes the aclnn KvCacheTurboQuant custom op as
// torch.ops.turboquant.kv_cache_turbo_quant on Ascend NPU (PrivateUse1).
#include <dlfcn.h>

#include <tuple>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ops/empty.h>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

constexpr const char *kOpApiSoPath =
    "/usr/local/Ascend/cann-9.1.0/opp/vendors/customize/op_api/lib/libcust_opapi.so";

using GetWorkspaceSizeFn = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
using RunFn = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

struct OpApi {
    GetWorkspaceSizeFn getWorkspaceSize = nullptr;
    RunFn run = nullptr;
};

const OpApi &GetOpApi()
{
    static const OpApi api = [] {
        void *handle = dlopen(kOpApiSoPath, RTLD_NOW | RTLD_GLOBAL);
        TORCH_CHECK(handle != nullptr, "dlopen ", kOpApiSoPath, " failed: ", dlerror());
        OpApi resolved;
        resolved.getWorkspaceSize = reinterpret_cast<GetWorkspaceSizeFn>(
            dlsym(handle, "aclnnKvCacheTurboQuantGetWorkspaceSize"));
        resolved.run = reinterpret_cast<RunFn>(dlsym(handle, "aclnnKvCacheTurboQuant"));
        TORCH_CHECK(resolved.getWorkspaceSize != nullptr && resolved.run != nullptr,
            "dlsym aclnnKvCacheTurboQuant failed: ", dlerror());
        return resolved;
    }();
    return api;
}

aclDataType ToAclType(const at::ScalarType dtype)
{
    switch (dtype) {
        case at::kBFloat16: return ACL_BF16;
        case at::kFloat: return ACL_FLOAT;
        case at::kByte: return ACL_UINT8;
        default:
            TORCH_CHECK(false, "unsupported dtype for kv_cache_turbo_quant: ", dtype);
    }
}

aclTensor *WrapTensor(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.is_contiguous(), "kv_cache_turbo_quant requires contiguous tensors");
    const auto sizes = tensor.sizes();
    return aclCreateTensor(sizes.data(), sizes.size(), ToAclType(tensor.scalar_type()),
        nullptr, 0, ACL_FORMAT_ND, sizes.data(), sizes.size(), tensor.data_ptr());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> KvCacheTurboQuant(const at::Tensor &kv, const at::Tensor &rotation,
    const at::Tensor &qjl, int64_t mseBits)
{
    const OpApi &api = GetOpApi();
    TORCH_CHECK(kv.dim() == 3, "kv must be [num_tokens, num_kv_heads, head_dim]");
    TORCH_CHECK(rotation.dim() == 2 && qjl.dim() == 2, "rotation/qjl must be 2-D");
    TORCH_CHECK(mseBits == 2 || mseBits == 3 || mseBits == 4, "mse_bits must be 2, 3 or 4");
    const int64_t numTokens = kv.size(0);
    const int64_t numKvHeads = kv.size(1);
    const int64_t headDim = kv.size(2);
    TORCH_CHECK(rotation.size(0) == headDim && rotation.size(1) == headDim,
        "rotation must be [head_dim, head_dim]");
    const int64_t idxBytes = headDim * mseBits / 8;
    const int64_t qjlBytes = qjl.size(0) / 8;

    const auto u8Options = kv.options().dtype(at::kByte);
    at::Tensor quantIdx = at::empty({numTokens, numKvHeads, idxBytes}, u8Options);
    at::Tensor quantQjl = at::empty({numTokens, numKvHeads, qjlBytes}, u8Options);
    at::Tensor quantNorm = at::empty({numTokens, numKvHeads}, kv.options());
    at::Tensor quantGamma = at::empty({numTokens, numKvHeads}, kv.options());

    aclTensor *kvTensor = WrapTensor(kv);
    aclTensor *rotTensor = WrapTensor(rotation);
    aclTensor *qjlTensor = WrapTensor(qjl);
    aclTensor *idxTensor = WrapTensor(quantIdx);
    aclTensor *qjlOutTensor = WrapTensor(quantQjl);
    aclTensor *normTensor = WrapTensor(quantNorm);
    aclTensor *gammaTensor = WrapTensor(quantGamma);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = api.getWorkspaceSize(kvTensor, rotTensor, qjlTensor, mseBits,
        idxTensor, qjlOutTensor, normTensor, gammaTensor, &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS,
        "aclnnKvCacheTurboQuantGetWorkspaceSize failed, ret=", static_cast<int>(status));

    at::Tensor workspace;
    void *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)}, u8Options);
        workspacePtr = workspace.data_ptr();
    }
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    status = api.run(workspacePtr, workspaceSize, executor, stream);

    aclDestroyTensor(kvTensor);
    aclDestroyTensor(rotTensor);
    aclDestroyTensor(qjlTensor);
    aclDestroyTensor(idxTensor);
    aclDestroyTensor(qjlOutTensor);
    aclDestroyTensor(normTensor);
    aclDestroyTensor(gammaTensor);
    TORCH_CHECK(status == ACL_SUCCESS,
        "aclnnKvCacheTurboQuant failed, ret=", static_cast<int>(status));
    return std::make_tuple(quantIdx, quantQjl, quantNorm, quantGamma);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(turboquant, m)
{
    m.def("kv_cache_turbo_quant(Tensor kv, Tensor rotation, Tensor qjl, int mse_bits) "
          "-> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(turboquant, PrivateUse1, m)
{
    m.impl("kv_cache_turbo_quant", &KvCacheTurboQuant);
}