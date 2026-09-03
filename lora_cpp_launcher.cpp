/* Minimal C++ launcher for triton-compiled Ascend kernels.
 *
 * Replaces the triton Python launch path (measured ~88us/launch) with a
 * direct rtKernelLaunch (~6us/launch, probe-verified bit-identical output).
 *
 * Layout (matches triton's generated launcher for our kernels, verified via
 * isolate2 4-layout experiment: 3-slot wins, 0/2-slot leave y unchanged):
 *   [ffts ctrl addr 8B][syncBlockLock NULL 8B][workspace NULL 8B][kernel args aligned][gridX gridY gridZ 3x4B]
 * ffts = rtGetC2cCtrlAddr value (fetched once per device; kernel doesn't read
 * it, but layout must match). force_simt_only=False, lock_num=0, workspace=0.
 *
 * Built with the same toolchain as triton's stubs:
 *   _build_npu_ext("lora_cpp_launcher", src, kernel_launcher="torch")
 */
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

#include "runtime/runtime/rt.h"

namespace {

struct StubHolder {
  uint64_t storage = 0;
};

std::unordered_map<std::string, std::unique_ptr<StubHolder>>& stubs() {
  static std::unordered_map<std::string, std::unique_ptr<StubHolder>> m;
  return m;
}
std::unordered_map<std::string, int>& name_counts() {
  static std::unordered_map<std::string, int> m;
  return m;
}

}  // namespace

extern "C" {

/* Register a kernel binary once; returns the rtFunctionRegister stub handle
 * (what rtKernelLaunch takes as `func`), or 0 on failure. */
uint64_t lora_register_kernel(const char* name, const void* data,
                              uint64_t data_size, const char* mode_str,
                              int device) {
  rtDevBinary_t devbin;
  devbin.data = const_cast<void*>(data);
  devbin.length = data_size;
  devbin.magic = (std::string(mode_str) == "aiv")
                     ? RT_DEV_BINARY_MAGIC_ELF_AIVEC
                     : RT_DEV_BINARY_MAGIC_ELF;
  devbin.version = 0;

  if (rtSetDevice(device) != RT_ERROR_NONE) return 0;

  void* devbinHandle = nullptr;
  if (rtDevBinaryRegister(&devbin, &devbinHandle) != RT_ERROR_NONE) return 0;

  // NOTE: stub name MUST NOT collide with triton's own registration names
  // (npu_utils.cpp uses `name + "_" + counter` with its own counter — the
  // same scheme we used before; duplicates silently clobber each other in
  // the runtime and the clobbered stub no-ops on launch). "lora_cpp_" prefix
  // makes ours disjoint.
  std::string stub_name = "lora_cpp_" + std::string(name) + "_" +
                          std::to_string(name_counts()[name]++);
  auto it = stubs().emplace(stub_name, std::make_unique<StubHolder>()).first;
  void* stub_handle = &it->second->storage;
  rtError_t ret = rtFunctionRegister(devbinHandle, stub_handle,
                                     stub_name.c_str(),
                                     const_cast<char*>(name), 0);
  if (ret != RT_ERROR_NONE) {
    printf("[lora_cpp] rtFunctionRegister FAILED name=%s stub=%s ret=0x%x\n",
           name, stub_name.c_str(), ret);
    return 0;
  }
  printf("[lora_cpp] registered name=%s stub=%s storage=%p\n",
         name, stub_name.c_str(), stub_handle);
  // First launch after a fresh registration can no-op if the runtime hasn't
  // finished filling the stub (observed: new kernel registered after other
  // kernels were already launched executes nothing, ret=0). Sync once per
  // registration — cost is per kernel case, not per launch.
  rtDeviceSynchronize();
  return reinterpret_cast<uint64_t>(stub_handle);
}

/* Minimal launch: pack args and fire rtKernelLaunch on the given stream.
 * arg_ptrs[i] points at the i-th arg value, arg_sizes[i] its byte size.
 * Returns the rtKernelLaunch error code. */
uint64_t lora_launch_kernel(uint64_t func, uint64_t stream, int32_t grid_x,
                            const void* const* arg_ptrs,
                            const uint64_t* arg_sizes, int32_t num_args) {
  alignas(8) unsigned char buf[512];

  // leading slots: [ffts][syncBlockLock=NULL][workspace=NULL]
  void* ffts_addr = nullptr;
  uint32_t ffts_len = 0;
  rtGetC2cCtrlAddr(reinterpret_cast<uint64_t*>(&ffts_addr), &ffts_len);
  std::memcpy(buf, &ffts_addr, 8);
  std::memset(buf + 8, 0, 16);

  uint64_t off = 24;
  for (int32_t i = 0; i < num_args; ++i) {
    uint64_t align = arg_sizes[i] >= 8 ? 8 : (arg_sizes[i] >= 4 ? 4 : 1);
    off = (off + align - 1) & ~(align - 1);
    std::memcpy(buf + off, arg_ptrs[i], arg_sizes[i]);
    off += arg_sizes[i];
  }
  off = (off + 3) & ~3ULL;
  int32_t grid[3] = {grid_x, 1, 1};
  std::memcpy(buf + off, grid, 12);
  off += 12;

  return static_cast<uint64_t>(
      rtKernelLaunch(reinterpret_cast<void*>(func),
                     static_cast<uint32_t>(grid_x), buf, off, nullptr,
                     reinterpret_cast<rtStream_t>(stream)));
}

/* Zero-copy variant: caller pre-packs the flat launch buffer
 * ([24B slots][args][grid]) in Python; C++ only fires the launch.
 * Returns the rtKernelLaunch error code. */
uint64_t lora_launch_flat(uint64_t func, uint64_t stream, int32_t grid_x,
                          const void* flat_args, uint64_t flat_size) {
  return static_cast<uint64_t>(
      rtKernelLaunch(reinterpret_cast<void*>(func),
                     static_cast<uint32_t>(grid_x), const_cast<void*>(flat_args),
                     static_cast<uint32_t>(flat_size), nullptr,
                     reinterpret_cast<rtStream_t>(stream)));
}

/* Read back the runtime-filled stub storage for a registered func handle
 * (diagnostic: 0 means the registration never stuck). */
uint64_t lora_peek_stub(uint64_t func) {
  return *reinterpret_cast<uint64_t*>(func);
}

/* Fetch the C2C control address for `device` (cache in Python, pack into
 * slot 0 of the flat buffer once per registration). 0 on failure. */
uint64_t lora_get_ffts_addr(int device) {
  if (rtSetDevice(device) != RT_ERROR_NONE) return 0;
  void* ffts_addr = nullptr;
  uint32_t ffts_len = 0;
  if (rtGetC2cCtrlAddr(reinterpret_cast<uint64_t*>(&ffts_addr), &ffts_len) !=
      RT_ERROR_NONE)
    return 0;
  return reinterpret_cast<uint64_t>(ffts_addr);
}

}  // extern "C"
