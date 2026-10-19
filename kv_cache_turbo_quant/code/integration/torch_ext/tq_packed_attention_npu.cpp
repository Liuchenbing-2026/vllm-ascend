// Target-dependent PTA candidate. It MUST compile against the actual generated
// OPP header: no handwritten ACLNN workspace/run function-pointer signatures.
#include "tq_packed_attention_contract.h"
#include "aclnn_tq_packed_attention.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include <ATen/ops/empty.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <dlfcn.h>
#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tq_packed_binding {
namespace {
class Api {
public:
    // Decltypes are checked against the generated target header at build time.
    using Workspace = decltype(&aclnnTqPackedAttentionGetWorkspaceSize);
    using Run = decltype(&aclnnTqPackedAttention);
    explicit Api(const std::string &path)
    {
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        TORCH_CHECK(handle_ != nullptr, "TQ packed OPP dlopen failed: ", dlerror());
        workspace = reinterpret_cast<Workspace>(dlsym(handle_, "aclnnTqPackedAttentionGetWorkspaceSize"));
        run = reinterpret_cast<Run>(dlsym(handle_, "aclnnTqPackedAttention"));
        if (!workspace || !run) {
            dlclose(handle_);
            handle_ = nullptr;
            TORCH_CHECK(false, "TQ packed OPP is missing generated ACLNN symbols");
        }
    }
    ~Api() { if (handle_) dlclose(handle_); }
    Api(const Api &) = delete;
    Api &operator=(const Api &) = delete;
    Workspace workspace;
    Run run;
private:
    void *handle_ = nullptr;
};

std::shared_ptr<Api> GetApi(const std::string &path)
{
    // Immutable, isolated OPP paths are required. No environment lookup or
    // bare libcust_opapi.so lookup that can pick another session's vendor.
    static std::mutex mutex;
    static std::unordered_map<std::string, std::shared_ptr<Api>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(path);
    if (found == cache.end()) found = cache.emplace(path, std::make_shared<Api>(path)).first;
    return found->second;
}

struct Views {
    std::array<at::Tensor, 8> tensors;  // Keep every tensor alive until queued call completes.
    std::vector<aclTensor *> handles;
    explicit Views(std::array<at::Tensor, 8> inputs) : tensors(std::move(inputs))
    {
        handles.reserve(tensors.size());
        for (const auto &tensor : tensors) {
            aclDataType dtype;
            switch (tensor.scalar_type()) {
                case at::kBFloat16: dtype = ACL_BF16; break;
                case at::kByte: dtype = ACL_UINT8; break;
                case at::kInt: dtype = ACL_INT32; break;
                default: TORCH_CHECK(false, "TQ packed unsupported ACL dtype");
            }
            const auto sizes = tensor.sizes();
            const auto strides = tensor.strides();
            // Validated dense ND; data_ptr is the logical view start. An
            // explicit zero ACL offset avoids applying a storage offset twice.
            auto *view = aclCreateTensor(sizes.data(), sizes.size(), dtype, strides.data(), 0,
                ACL_FORMAT_ND, sizes.data(), sizes.size(), tensor.data_ptr());
            if (!view) {
                for (auto *old : handles) aclDestroyTensor(old);
                handles.clear();
                TORCH_CHECK(false, "TQ packed aclCreateTensor failed");
            }
            handles.push_back(view);
        }
    }
    ~Views() { for (auto *view : handles) aclDestroyTensor(view); }
    Views(const Views &) = delete;
    Views &operator=(const Views &) = delete;
};

Outputs Run(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
            const at::Tensor &table, const at::Tensor &lengths, const at::Tensor &centroids,
            double scale, int64_t apiScratchBytes, const std::string &opApiLibrary)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "TQ packed requires NPU tensors");
    c10::DeviceGuard guard(query.device());
    auto outputs = Allocate(query, key, value, table, lengths, centroids, scale, apiScratchBytes, opApiLibrary);
    if (query.size(0) == 0) return outputs;
    auto api = GetApi(opApiLibrary);
    auto views = std::make_shared<Views>(std::array<at::Tensor, 8>{
        query, key, value, table, lengths, centroids, std::get<0>(outputs), std::get<1>(outputs)});
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    const auto &t = views->handles;
    auto result = api->workspace(t[0], t[1], t[2], t[3], t[4], t[5], scale, apiScratchBytes,
                                 t[6], t[7], &bytes, &executor);
    TORCH_CHECK(result == ACL_SUCCESS, "TQ packed ACLNN workspace query failed: ", static_cast<int>(result));
    TORCH_CHECK(bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "TQ packed ACLNN workspace exceeds int64");
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, query.options().dtype(at::kByte));
    const auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // Preserve framework queue ordering. No direct kernel launch, stream(true)
    // flush, forced synchronization, or legacy reader fallback is introduced.
    auto call = [api, views, workspace, executor, stream, bytes]() -> int {
        (void)views;  // Deliberately retain descriptor/input/output ownership.
        const auto status = api->run(bytes ? workspace.data_ptr() : nullptr, bytes, executor, stream);
        TORCH_CHECK(status == ACL_SUCCESS, "TQ packed ACLNN execute failed: ", static_cast<int>(status));
        return static_cast<int>(status);
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnTqPackedAttention", call);
    return outputs;
}
}  // namespace
}  // namespace tq_packed_binding

TORCH_LIBRARY_IMPL(turboquant, PrivateUse1, m)
{
    m.impl("tq_packed_attention", &tq_packed_binding::Run);
}
