# -*- coding: utf-8 -*-
"""Preserve the real service failure and fixed inference-mode regression evidence."""

import json
from pathlib import Path
import shutil


def copy(source, destination):
    destination = Path(destination)
    if destination.exists() and destination.read_bytes() == Path(source).read_bytes():return str(destination)
    if destination.exists():destination.chmod(destination.stat().st_mode | 0o200)
    return shutil.copy2(source, destination)


def main():
    root = Path(__file__).resolve().parent
    experiment = root.parent.parent / "source/tq-paired-read-20261014/kv_cache_turbo_quant/experiments/paired_read_20261014"
    previous = root.parent.parent / "knowledge-opt-20261013/项目交付/任务列表/任务23 Qwen3-30B-A3B-TurboQuant-128K双卡性能对比/详细任务/20261014_Decode定位与入队优化/published-source-paired.json"
    record = json.loads(previous.read_text())
    assert record["sha"] == "a7a072ec381e9eedd5e1f1a91e713fc29c19f1a1"
    copy(previous, root / "published-source-paired-before-inference.json")
    for name in ["paired_read_inference_20261014", "paired_model_inference_20261014", "paired_model_attempt2_20261014"]:
        source = root / (name + "-snapshot")
        shutil.copytree(source, experiment / source.name, dirs_exist_ok=True, copy_function=copy)
    for name in ["fix_paired_inference.py", "deploy_paired_inference.py", "host_paired_inference.py", "verify_inference_lookup.py", "prepare_paired_model_inference.py", "published-source-paired-before-inference.json", "package_inference_stage.py"]:
        copy(root / name, experiment / name)
    shutil.copytree(root / "paired-model-inference-runtime/scripts", experiment / "actual-inference-model-scripts",
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"), copy_function=copy)
    memory = json.loads((root / "paired_read_inference_20261014-snapshot/memory/memory-review.json").read_text())
    assert memory["native_invocations"] == memory["invocations_without_access_error"] == 40
    cpu = json.loads((root / "paired_read_inference_20261014-snapshot/cpu-regression.json").read_text())
    assert cpu["untracked_mutations_passed"] == 3
    readme = experiment / "README.md";text = readme.read_text();marker = "## 推理模式兼容修复与重新验收"
    if marker in text:text = text[:text.index(marker)]
    text += "\n" + marker + """

首轮整网8K冒烟失败：Inference tensors do not track version counter。错误在新LUT缓存的版本检查，失败run `write_ab_20261014T004515Z`全部保留，未进入正式cohort。

修复将生产常量LUT创建为普通、可追踪版本的tensor；对任意inference LUT每次重建前缀以正确处理未追踪的原位修改，固定索引表仍按设备复用。CPU原版明确失败/修复通过，原15组逐位+20异步+2LUT更新保留，另增4组真实NPU inference-mode/原位更新，40次访存正覆盖通过。复用原72c838...设备kernel和Torch二进制，未更改舍入或容差。

修复后两路读取微测：320token161.433→114.375us（-29.15%），120000token718.718→684.862us（-4.71%），均事件区间含Host间隙，不能作为TTFT/TPOT或整网收益。新run `write_ab_20261014T005839Z`已重新启动；实际新脚本在actual-inference-model-scripts，完整BF16对照待完成。原2MiB退出泄漏及量化器告警仍未关闭。

修复依据：[PyTorch inference_mode文档](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)及本机torch2.10.0 CPU/NPU反例；没有通过捕获异常复用旧LUT。
"""
    readme.write_text(text)


if __name__ == "__main__":
    main()
