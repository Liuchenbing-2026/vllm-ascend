# Retained experiment, 2026-10-14

Rejected: 15 strict codec cases and 14 clean access checks; long microbench regressed 14%/23%, no model adoption. Known 2MiB global exit leak remains.

Branch perf/tq-read-tiles-20261014; base c894ba84c9bb1325dd6b689b1cf41a9ce3f26f6c. Delivery SHA is recorded in the task archive after push. This is an experiment archive, not a merged or accepted runtime.

Actual image vllm-ascend:dspark-a2-028; image ID sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14; RepoDigests empty. Dedicated tq-opt-20261011-model container, physical 2/5, 910B4-1, CANN9.1.0, Python3.12.13, torch2.10.0+cpu, torch_npu2.10.0.post4. Framework sources unchanged: vLLM 2cf0a6915ce544dc493a0990f2ea38d81601128a and Ascend 96df623103f921df1a4488170d106f36689acb59. Full dependencies, actual service arguments and requests are in Model_test/main task23.

The retained scripts and evidence/command.json files contain actual build, precision and kernel-filtered memory commands. All prior failed attempts are retained. No semantic quality dataset, CI or clean-image rebuild was run. No performance acceptance is claimed for this experiment.
