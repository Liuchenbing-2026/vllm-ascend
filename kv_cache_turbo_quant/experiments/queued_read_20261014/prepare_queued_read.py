"""Enqueue the already verified ACLNN reader through the installed PTA API."""

from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
source = root.parent.parent / "source/tq-queued-read-20261014/kv_cache_turbo_quant"
ext = source / "code/integration/torch_ext"
code = (ext / "tq_native_read.cpp").read_text()
code = code.replace("#include <vector>", "#include <vector>\n#include <memory>\n#include \"torch_npu/csrc/framework/OpCommand.h\"")
code = code.replace('tq_paged_read', 'tq_paged_read_queued')
start = code.index("    std::vector<aclTensor *> tensors{")
stop = code.index("    return output;", start)
code = code[:start] + """    // ACL descriptors and Tensor storage must survive until the queued
    // handler has submitted the physical kernel, even if the caller releases
    // its temporary inputs immediately after this function returns.
    auto tensors = std::shared_ptr<std::vector<aclTensor *>>(
        new std::vector<aclTensor *>{Wrap(cache), Wrap(table), Wrap(jobs), Wrap(lookup), Wrap(output)},
        [](std::vector<aclTensor *> *values) {
            for (auto *value : *values) aclDestroyTensor(value);
            delete values;
        });
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    const Api &api = GetApi();
    aclnnStatus status = api.workspace((*tensors)[0], (*tensors)[1], (*tensors)[2], (*tensors)[3], tokens,
        (*tensors)[4], &bytes, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "queued read workspace failed: ", static_cast<int>(status));
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, cache.options());
    std::vector<at::Tensor> keepAlive{cache, table, jobs, lookup, output, workspace};
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    const RunFn run = api.run;
    auto aclCall = [tensors, keepAlive, bytes, executor, stream, run]() -> int {
        const auto result = run(bytes ? keepAlive.back().data_ptr() : nullptr, bytes, executor, stream);
        TORCH_CHECK(result == ACL_SUCCESS, "queued read failed: ", static_cast<int>(result));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnTqPagedRead", aclCall);
""" + code[stop:]
(ext / "tq_queued_read.cpp").write_text(code)
build = (ext / "build_native_read.py").read_text().replace("tq_native_read", "tq_queued_read")
(ext / "build_queued_read.py").write_text(build)
wrapper = source / "code/integration/kvtq_native_read.py"
wrapper.write_text(wrapper.read_text().replace("native_read_20261013/torch-extensions/tq_native_read/tq_native_read.so", "queued_read_20261014/torch-extensions/tq_queued_read/tq_queued_read.so").replace("tq_paged_read", "tq_paged_read_queued"))
experiment = source / "experiments/queued_read_20261014"
experiment.mkdir(parents=True, exist_ok=True)
shutil.copy2(Path(__file__), experiment / Path(__file__).name)
print("queued reader uses original TqPagedRead ACLNN/kernel and the installed RunOpApiV2 queue")
