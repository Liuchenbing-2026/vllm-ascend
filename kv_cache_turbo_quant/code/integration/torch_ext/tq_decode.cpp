// Host orchestration only: preserve the existing FP32 rotations and BF16 FIA.
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <unordered_map>

namespace {
struct AttentionCall {
    c10::OperatorHandle op;
    c10::Stack defaults;
    std::unordered_map<std::string, size_t> positions;

    AttentionCall()
        : op(c10::Dispatcher::singleton().findSchemaOrThrow(
              "npu::npu_fused_infer_attention_score", ""))
    {
        for (const auto &arg : op.schema().arguments()) {
            positions.emplace(arg.name(), defaults.size());
            defaults.push_back(arg.default_value().value_or(c10::IValue()));
        }
    }
};

void Decode(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
            const at::Tensor &rotation, const at::Tensor &rotationT,
            at::IntArrayRef qLengths, at::IntArrayRef kvLengths,
            int64_t heads, int64_t kvHeads, int64_t blockSize, double scale,
            at::Tensor output)
{
    TORCH_CHECK(query.dim() == 3 && query.size(1) == heads && query.size(2) == 128,
                "TQ decode requires [batch, heads, 128] query");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && output.sizes() == query.sizes()
                    && output.scalar_type() == query.scalar_type(),
                "TQ decode query/output layout mismatch");
    TORCH_CHECK(key.dim() == 3 && key.size(1) == kvHeads && key.size(2) == 128
                    && value.sizes() == key.sizes(), "TQ decode KV shape mismatch");
    TORCH_CHECK(key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16,
                "TQ decode requires BF16 materialized KV");
    TORCH_CHECK(rotation.scalar_type() == at::kFloat && rotationT.scalar_type() == at::kFloat
                    && rotation.sizes() == at::IntArrayRef({128,128})
                    && rotationT.sizes() == rotation.sizes(), "TQ rotation shape/dtype mismatch");
    for (const auto &tensor : {key,value,rotation,rotationT,output}) {
        TORCH_CHECK(tensor.device() == query.device(), "TQ decode device mismatch");
    }
    TORCH_CHECK(qLengths.size() == static_cast<size_t>(query.size(0))
                    && qLengths.size() == kvLengths.size(), "TQ decode lengths mismatch");
    if (query.size(0) == 0) return;
    TORCH_CHECK(qLengths.back() == query.size(0) && kvLengths.back() == key.size(0),
                "TQ decode cumulative lengths mismatch");
    int64_t previous=0;
    for (size_t i=0; i<qLengths.size(); ++i) {
        TORCH_CHECK(qLengths[i] == static_cast<int64_t>(i)+1 && kvLengths[i]>previous,
                    "TQ decode requires one query and positive KV length per sequence");
        previous=kvLengths[i];
    }
    auto rotatedQuery = at::matmul(query.to(at::kFloat), rotationT).to(at::kBFloat16);
    static const AttentionCall attention;
    auto stack = attention.defaults;
    auto set = [&](const char *name, c10::IValue value) {
        stack.at(attention.positions.at(name)) = std::move(value);
    };
    set("query", rotatedQuery); set("key",key); set("value",value);
    set("actual_seq_lengths", qLengths.vec());
    set("actual_seq_lengths_kv", kvLengths.vec());
    set("num_heads",heads); set("num_key_value_heads",kvHeads);
    set("block_size",blockSize); set("scale",scale);
    set("input_layout",std::string("TND")); set("sparse_mode",int64_t(0));
    attention.op.callBoxed(&stack);
    auto attentionOutput=stack.at(0).toTensor().view({query.size(0),heads,128});
    output.copy_(at::matmul(attentionOutput.to(at::kFloat),rotation));
}

void DecodeMeta(const at::Tensor &, const at::Tensor &, const at::Tensor &,
                const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef,
                int64_t, int64_t, int64_t, double, at::Tensor) {}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
TORCH_LIBRARY_FRAGMENT(turboquant, m)
{
    m.def("tq_decode(Tensor query, Tensor key, Tensor value, Tensor rotation, "
          "Tensor rotation_t, int[] q_lengths, int[] kv_lengths, int heads, "
          "int kv_heads, int block_size, float scale, Tensor(a!) output) -> ()");
}
TORCH_LIBRARY_IMPL(turboquant, PrivateUse1, m) { m.impl("tq_decode", &Decode); }
TORCH_LIBRARY_IMPL(turboquant, Meta, m) { m.impl("tq_decode", &DecodeMeta); }
