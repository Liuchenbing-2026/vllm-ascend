"""Prepare a separate queued quantizer while leaving the deployed reader run frozen."""

from pathlib import Path

root = Path(__file__).resolve().parent
integration = root.parent.parent / "source/tq-queued-codec-20261014/kv_cache_turbo_quant/code/integration"
baseline = root.parent.parent / "source/tq-queued-read-20261014/kv_cache_turbo_quant/code/integration"
extension = integration / "torch_ext"
text = (baseline / "torch_ext/turboquant_torch.cpp").read_text()
text = text.replace("#include <vector>", '#include <vector>\n#include <memory>\n#include "torch_npu/csrc/framework/OpCommand.h"')
start = text.index("    at::Tensor workspace;\n")
stop = text.index("    return std::make_tuple", start)
text = text[:start] + '''    // Metadata and Tensor storage stay alive until PTA submits the kernel.
    auto descriptors = std::shared_ptr<std::vector<aclTensor *>>(
        new std::vector<aclTensor *>{kvTensor, rotTensor, qjlTensor, idxTensor,
            qjlOutTensor, normTensor, gammaTensor},
        [](std::vector<aclTensor *> *values) {
            for (auto *value : *values) aclDestroyTensor(value);
            delete values;
        });
    at::Tensor workspace;
    if (workspaceSize > 0) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)}, u8Options);
    }
    std::vector<at::Tensor> keepAlive{kv, rotation, qjl, quantIdx, quantQjl,
        quantNorm, quantGamma, workspace};
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    const RunFn run = api.run;
    auto handler = [descriptors, keepAlive, workspaceSize, executor, stream, run]() -> int {
        auto status = run(workspaceSize ? keepAlive.back().data_ptr() : nullptr,
            workspaceSize, executor, stream);
        TORCH_CHECK(status == ACL_SUCCESS, "queued quantizer failed, ret=", static_cast<int>(status));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnKvCacheTurboQuant", handler);
''' + text[stop:]
text = text.replace("TORCH_LIBRARY(turboquant, m)", "TORCH_LIBRARY_FRAGMENT(turboquant, m)")
text = text.replace("kv_cache_turbo_quant(", "kv_cache_turbo_quant_queued(")
text = text.replace('m.impl("kv_cache_turbo_quant",', 'm.impl("kv_cache_turbo_quant_queued",')
position = text.index("} // namespace")
text = text[:position] + '''std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> QuantMeta(
    const at::Tensor &kv, const at::Tensor &, const at::Tensor &qjl, int64_t mseBits)
{
    const auto u8 = kv.options().dtype(at::kByte);
    return std::make_tuple(at::empty({kv.size(0), kv.size(1), kv.size(2) * mseBits / 8}, u8),
        at::empty({kv.size(0), kv.size(1), qjl.size(0) / 8}, u8),
        at::empty({kv.size(0), kv.size(1)}, kv.options()),
        at::empty({kv.size(0), kv.size(1)}, kv.options()));
}

''' + text[position:]
text += '''
TORCH_LIBRARY_IMPL(turboquant, Meta, m)
{
    m.impl("kv_cache_turbo_quant_queued", &QuantMeta);
}
'''
start = text.index("    // Metadata and Tensor storage")
stop = text.index("    at::Tensor workspace;", start)
descriptors = text[start:stop]
text = text[:start] + text[stop:]
position = text.index("    uint64_t workspaceSize = 0;")
text = text[:position] + descriptors + text[position:]
(extension / "tq_queued_quant.cpp").write_text(text)
builder = (baseline / "torch_ext/build_queued_read.py").read_text().replace("tq_queued_read", "tq_queued_quant")
(extension / "build_queued_quant.py").write_text(builder)
(integration / "kvtq_queued_quant.py").write_text('''"""Queue-aware submission of the original, unchanged Ascend C quantizer."""

import torch
import torch_npu  # noqa: F401

NATIVE_EXTENSION = "/ws/queued_codec_20261014/torch-extensions/tq_queued_quant/tq_queued_quant.so"


class QueuedQuantizer:
    def __init__(self):
        if not hasattr(torch.ops.turboquant, "kv_cache_turbo_quant_queued"):
            torch.ops.load_library(NATIVE_EXTENSION)

    def __call__(self, vectors, rotation, qjl, bits):
        return torch.ops.turboquant.kv_cache_turbo_quant_queued(vectors, rotation, qjl, bits)
''')
store = integration / "kvtq_store.py"
text = (baseline / "kvtq_store.py").read_text()
needle = '''    idx, _qjlb, norm, _gamma = torch.ops.turboquant.kv_cache_turbo_quant(
        vectors, rot, qjl, _MSE_BITS)'''
assert text.count(needle) == 1
text = text.replace(needle, '''    from kvtq_queued_quant import QueuedQuantizer

    quantizer = getattr(self, '_kvtq_queued_quantizer', None)
    if quantizer is None:
        quantizer = QueuedQuantizer()
        self._kvtq_queued_quantizer = quantizer
    idx, _qjlb, norm, _gamma = quantizer(vectors, rot, qjl, _MSE_BITS)''')
store.write_text(text)
print("Prepared isolated candidate, original quantizer/device code untouched")
