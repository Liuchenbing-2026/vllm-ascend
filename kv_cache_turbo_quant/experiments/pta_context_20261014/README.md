# Retained PTA diagnostics, 2026-10-14

Not adopted; model gates remain closed. Final candidate compiles and passes 15 exact four-output codec cases, 20 async ordering cases and 90 full-cache cases, but 54 filtered native calls report 135 GM bounds warnings and 534 register warnings. Original standalone control also reports a GM bounds warning; baseline reproduction is not root-cause closure. Known 2MiB exit leak remains.

For stream-resource attempts, attempt1 failed loading unsafe_empty_workspace and attempt2 failed loading UseStreamResInCurrentThread; headers declare symbols absent from the installed shared library. Attempt3 uses exported CANN stream-resource API and an owned Tensor workspace. Standard PTA variant uses the installed EXEC_NPU_CMD_V1_EXT macro and its exported context/workspace helpers. Original native quantizer and vendor unchanged.

Base 9e188464533435b6b29035d2bee71324cb47e6e5; branch perf/tq-pta-context-20261014. Exact delivery SHA is in task23 after remote verification. Actual image vllm-ascend:dspark-a2-028; image ID sha256:3d74258bb4aba4d7ba1d972f886b78d25b7459346bcce4933f27ce8e6bdd4e14; RepoDigests empty. Dedicated tq-opt-20261011-model, physical2/5, CANN9.1.0, Python3.12.13, torch2.10.0+cpu, torch_npu2.10.0.post4. Framework sources unchanged: vLLM 2cf0a6915ce544dc493a0990f2ea38d81601128a / Ascend 96df623103f921df1a4488170d106f36689acb59.

Scripts and retained command.json files contain actual compilation, precision and mssanitizer commands. Full service/dependencies and fixed model recipes are in Model_test/main task23. No CI, clean-image rebuild or new semantic quality evaluation. No end-to-end gains claimed; model run was not released.
