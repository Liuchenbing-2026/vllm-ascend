/* Native (kPrivateUse1) torch impls for the vllm_ascend_triton custom ops.
 *
 * Why: the python custom_op impls are NOT captured into the aclgraph that
 * vllm-ascend replays for decode (torch_npu records torch-native ops, not
 * python impls).  So every decode step re-executes the 15 LoRA op impls in
 * python outside the graph (~240us each + interplay overhead), measured
 * 35.4 vs 31.1 ms/char (+13.7%) against the native AscendC ops.  Registering
 * the same schemas on the PrivateUse1 dispatch key (exactly how
 * vllm_ascend_C.so registers its AscendC ops) makes them torch-native:
 * aclgraph captures their kernel launches, decode replays them with no
 * python on the step path.
 *
 * Case registry: the python module compiles the triton kernels during warmup
 * and binds key -> rtFunctionRegister handle via lora_native_bind_case.  On a
 * miss the impl calls back into python (lora_native_set_handler) which builds
 * the case (compile + verify) and binds it; if that fails or is skipped the
 * impl falls back to the AscendC native op via the dispatcher so results stay
 * correct.  The key string format must match lora_ops_triton._cpp_key_str.
 *
 * Built with torch.utils.cpp_extension.load (needs torch headers + libtorch;
 * the plain _build_npu_ext stub builder does not link torch).
 */
#include <Python.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <c10/core/DispatchKey.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/runtime/rt.h"

namespace {

bool env_flag_off(const char* name) {
  const char* v = getenv(name);
  return v != nullptr && std::string(v) == "0";
}

std::string fmt_dtype(at::ScalarType t) {
  switch (t) {
    case at::kHalf: return "torch.float16";
    case at::kBFloat16: return "torch.bfloat16";
    case at::kInt: return "torch.int32";
    case at::kFloat: return "torch.float32";
    case at::kLong: return "torch.int64";
    default: return "torch.unknown";
  }
}

bool triton_dtype(const at::Tensor& t) {
  return t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16;
}

// canonical case key: "name|k=v,k=v|dtype,dtype" -- MUST match
// lora_ops_triton._cpp_key_str (ints as "%lld", floats as "%.1f").
struct Kv { std::string k, v; };

std::string build_key(const std::string& name, std::vector<Kv> kvs,
                      const std::vector<std::string>& dtypes) {
  std::sort(kvs.begin(), kvs.end(),
            [](const Kv& a, const Kv& b) { return a.k < b.k; });
  std::string s = name + "|";
  for (size_t i = 0; i < kvs.size(); ++i) {
    if (i) s += ",";
    s += kvs[i].k + "=" + kvs[i].v;
  }
  s += "|";
  for (size_t i = 0; i < dtypes.size(); ++i) {
    if (i) s += ",";
    s += dtypes[i];
  }
  return s;
}

std::unordered_map<std::string, uint64_t>& cases() {
  static std::unordered_map<std::string, uint64_t> m;
  return m;
}

// per-tag int32 scratch for idx/seq casts (mirror python _to_int32)
std::unordered_map<int64_t, at::Tensor>& scratch(int tag) {
  static std::unordered_map<int64_t, at::Tensor> s1, s2;
  return tag == 1 ? s1 : s2;
}

at::Tensor cast_i32(const at::Tensor& t, int tag) {
  auto& s = scratch(tag)[t.numel()];
  if (!s.defined() || s.device() != t.device()) {
    s = at::empty({t.numel()}, t.options().dtype(at::kInt));
  }
  s.copy_(t);  // dtype-converting device copy, stream-ordered
  return s;
}

PyObject* g_handler = nullptr;        // key -> bool (case built + bound)
PyObject* g_stream_getter = nullptr;  // -> int npu stream

}  // namespace

extern "C" {

uint64_t lora_native_bind_case(const char* key, uint64_t func) {
  cases()[key] = func;
  return 1;
}

void lora_native_set_handler(PyObject* fn) {
  Py_XINCREF(fn);
  Py_XDECREF(g_handler);
  g_handler = fn;
}

void lora_native_set_stream_getter(PyObject* fn) {
  Py_XINCREF(fn);
  Py_XDECREF(g_stream_getter);
  g_stream_getter = fn;
}

}  // extern "C"

namespace {

// Impls always run with the GIL held (invoked from the python dispatcher),
// so PyGILState_Ensure is a no-op here -- kept for safety.
bool call_handler(const std::string& key) {
  if (!g_handler) return false;
  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* r = PyObject_CallFunction(g_handler, "s", key.c_str());
  bool ok = r == Py_True;
  if (r == nullptr) {
    PyErr_Print();
    PyErr_Clear();
  }
  Py_XDECREF(r);
  PyGILState_Release(st);
  return ok;
}

uint64_t current_stream() {
  if (!g_stream_getter) return 0;
  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* r = PyObject_CallNoArgs(g_stream_getter);
  uint64_t v = 0;
  if (r != nullptr) {
    // same value as the old python launcher passed (may be negative for the
    // default stream on some torch_npu versions; preserve the raw bits)
    v = static_cast<uint64_t>(PyLong_AsLongLong(r));
    Py_DECREF(r);
  } else {
    PyErr_Clear();
  }
  PyGILState_Release(st);
  return v;
}

// flat-pack launch, same layout as the python _cpp_launch buffer
// [ffts][syncBlockLock][workspace][ptrs...][floats...][grid x3]
void cntlog();  // defined below (debug launch counter)
uint64_t launch(uint64_t func, int32_t grid_x,
                const std::vector<const void*>& ptrs,
                const std::vector<float>& floats) {
  alignas(8) unsigned char buf[512];
  void* ffts = nullptr;
  uint32_t fl = 0;
  rtGetC2cCtrlAddr(reinterpret_cast<uint64_t*>(&ffts), &fl);
  std::memcpy(buf, &ffts, 8);
  std::memset(buf + 8, 0, 16);
  uint64_t off = 24;
  for (const void* p : ptrs) {
    std::memcpy(buf + off, &p, 8);
    off += 8;
  }
  for (float f : floats) {
    std::memcpy(buf + off, &f, 4);
    off += 4;
  }
  off = (off + 3) & ~3ULL;
  int32_t grid[3] = {grid_x, 1, 1};
  std::memcpy(buf + off, grid, 12);
  off += 12;
  uint64_t ret = static_cast<uint64_t>(
      rtKernelLaunch(reinterpret_cast<void*>(func),
                     static_cast<uint32_t>(grid_x), buf, off, nullptr,
                     reinterpret_cast<rtStream_t>(current_stream())));
  cntlog();
  return ret;
}

// observability: when TRITON_LORA_CPP_DEBUG is set, append a running launch
// counter so we can tell pure aclgraph replay (counter frozen) from per-step
// host dispatch (counter advancing) during serve.
void cntlog() {
  if (getenv("TRITON_LORA_CPP_DEBUG") == nullptr) return;
  static uint64_t n = 0;
  if (++n % 2000 == 0 || n <= 3) {
    FILE* f = fopen("/tmp/lora_native_calls.log", "a");
    if (f) {
      fprintf(f, "launch-count %llu\n", static_cast<unsigned long long>(n));
      fclose(f);
    }
  }
}

uint64_t lookup_or_build(const std::string& key) {
  auto it = cases().find(key);
  if (it != cases().end()) return it->second;
  bool built = call_handler(key);
  if (built) {
    it = cases().find(key);
    if (it != cases().end()) return it->second;
  }
  if (getenv("TRITON_LORA_CPP_DEBUG") != nullptr)
    printf("[lora_native] MISS key=%s -> AscendC fallback\n", key.c_str());
  return 0;
}

// ---- debug: unconditional first-call traces (diagnose serve dispatch) ----
// Append to a file: the EngineCore's stdout may drop raw printf bytes, a file
// cannot be lost.  Removed by the python module at import (fresh per boot).
void dbg(const std::string& tag, const std::string& msg) {
  static std::set<std::string> seen;
  static int n = 0;
  if (!(seen.insert(tag).second || n++ < 6)) return;
  FILE* f = fopen("/tmp/lora_native_calls.log", "a");
  if (f) {
    fprintf(f, "%s %s\n", tag.c_str(), msg.c_str());
    fclose(f);
  }
  printf("[lora_native] %s %s\n", tag.c_str(), msg.c_str());
}

std::string sh(const at::Tensor& t) {
  char b[128];
  snprintf(b, sizeof(b), "[%lld,%lld]%s",
           static_cast<long long>(t.size(0)),
           static_cast<long long>(t.size(-1)), fmt_dtype(t.scalar_type()).c_str());
  return b;
}

// The python dispatcher unwraps FunctionalTensorWrapper before it invokes a
// python-target impl, but a C++ python-key impl receives the raw wrapper, and
// torch ops (reshape / copy_) on it outside a FunctionalTensorMode throw
// "Attempting to use FunctionalTensor on its own".  Mirror the unwrap so all
// torch ops below operate on the underlying real tensor -- the same thing the
// native AscendC ops effectively run on (their PrivateUse1 impls get the raw
// wrapper too and just use its storage/address).
at::Tensor uw(const at::Tensor& t) {
  if (!at::functionalization::impl::isFunctionalTensor(t)) return t;
  return at::functionalization::impl::from_functional_tensor(t, false);
}

// Dynamo's fake pass runs FX nodes on FunctionalTensorWrapper(FakeTensor) to
// validate the compiled graph; on the raw wrapper has_storage() is true, so
// the only reliable discriminator is the UNWRAPPED value: a fake/meta tensor
// carries no real storage.  Skip the node then (no-op, exactly how the old
// pyobj bail passed the fake pass).  Real aclgraph-capture wrappers unwrap to
// real NPU tensors and reach the kernels.
bool fake_like(const at::Tensor& t) {
  const auto* impl = t.unsafeGetTensorImpl();
  if (!impl->has_storage()) return true;
  if (t.device().is_meta()) return true;
  return impl->storage().data_ptr() == nullptr;
}

// ---- AscendC fallbacks (boxed dispatcher calls; their impls are C++) ----

bool boxed_call(const char* opname, c10::Stack& stk) {
  std::optional<c10::OperatorHandle> op;
  try {
    op = c10::Dispatcher::singleton().findSchemaOrThrow(opname, "");
  } catch (const c10::Error&) {
    printf("[lora_native] %s not registered, y left unchanged\n", opname);
    return false;
  }
  op->callBoxed(&stk);
  return true;
}

void fb_bgmv_shrink(const at::Tensor& x, const at::Tensor& w,
                    const at::Tensor& idx, at::Tensor& y, double scale) {
  c10::Stack stk;
  stk.reserve(5);
  stk.emplace_back(x);
  stk.emplace_back(w);
  stk.emplace_back(idx);
  stk.emplace_back(y);
  stk.emplace_back(scale);
  boxed_call("_C_ascend::bgmv_shrink", stk);
}

void fb_bgmv_expand(const at::Tensor& x, const at::Tensor& w,
                    const at::Tensor& idx, at::Tensor& y,
                    int64_t slice_offset, int64_t slice_size) {
  c10::Stack stk;
  stk.reserve(6);
  stk.emplace_back(x);
  stk.emplace_back(w);
  stk.emplace_back(idx);
  stk.emplace_back(y);
  stk.emplace_back(slice_offset);
  stk.emplace_back(slice_size);
  boxed_call("_C_ascend::bgmv_expand", stk);
}

void fb_sgmv_shrink(const at::Tensor& x, const at::Tensor& w,
                    const at::Tensor& idx, const at::Tensor& seq,
                    at::Tensor& y, double scale) {
  c10::Stack stk;
  stk.reserve(6);
  stk.emplace_back(x);
  stk.emplace_back(w);
  stk.emplace_back(idx);
  stk.emplace_back(seq);
  stk.emplace_back(y);
  stk.emplace_back(scale);
  boxed_call("_C_ascend::sgmv_shrink", stk);
}

void fb_sgmv_expand(const at::Tensor& x, const at::Tensor& w,
                    const at::Tensor& idx, const at::Tensor& seq,
                    at::Tensor& y, int64_t slice_offset, int64_t slice_size) {
  c10::Stack stk;
  stk.reserve(7);
  stk.emplace_back(x);
  stk.emplace_back(w);
  stk.emplace_back(idx);
  stk.emplace_back(seq);
  stk.emplace_back(y);
  stk.emplace_back(slice_offset);
  stk.emplace_back(slice_size);
  boxed_call("_C_ascend::sgmv_expand", stk);
}

void cpp_bgmv_shrink(const at::Tensor& inputs_raw,
                     const at::Tensor& lora_a_weights_raw,
                     at::Tensor output_tensor_raw,
                     const at::Tensor& lora_indices_tensor_raw,
                     double scaling) {
  const auto inputs = uw(inputs_raw);
  const auto lora_a_weights = uw(lora_a_weights_raw);
  auto output_tensor = uw(output_tensor_raw);
  const auto lora_indices_tensor = uw(lora_indices_tensor_raw);
  if (fake_like(inputs) || fake_like(lora_a_weights) ||
      fake_like(output_tensor) || fake_like(lora_indices_tensor)) return;
  dbg("bgmv_shrink",
      "x=" + sh(inputs) + " w=" + sh(lora_a_weights) +
      " idx=" + sh(lora_indices_tensor));
  if (env_flag_off("TRITON_LORA_CPP") ||
      !triton_dtype(inputs) || !triton_dtype(lora_a_weights) ||
      lora_a_weights.size(-1) != inputs.size(-1)) {
    fb_bgmv_shrink(inputs, lora_a_weights, lora_indices_tensor,
                   output_tensor, scaling);
    return;
  }
  int64_t B = inputs.size(0), H = inputs.size(1);
  auto w = lora_a_weights.reshape(
      {lora_a_weights.size(0), -1, lora_a_weights.size(-1)});
  int64_t L = w.size(0), R = w.size(1);
  auto idx32 = cast_i32(lora_indices_tensor, 1);
  std::string key = build_key(
      "bgmv_shrink",
      {{"H", std::to_string(H)}, {"R", std::to_string(R)},
       {"L", std::to_string(L)}},
      {fmt_dtype(inputs.scalar_type()), fmt_dtype(w.scalar_type()),
       "torch.int32", fmt_dtype(output_tensor.scalar_type())});
  uint64_t func = lookup_or_build(key);
  if (!func) {
    fb_bgmv_shrink(inputs, lora_a_weights, lora_indices_tensor,
                   output_tensor, scaling);
    return;
  }
  launch(func, static_cast<int32_t>(B),
         {inputs.data_ptr(), w.data_ptr(), idx32.data_ptr(),
          output_tensor.data_ptr()},
         {static_cast<float>(scaling)});
}

void cpp_bgmv_expand_slice(const at::Tensor& inputs_raw,
                           const at::Tensor& lora_b_weights_raw,
                           at::Tensor output_tensor_raw,
                           const at::Tensor& lora_indices_tensor_raw,
                           int64_t slice_offset, int64_t slice_size) {
  const auto inputs = uw(inputs_raw);
  const auto lora_b_weights = uw(lora_b_weights_raw);
  auto output_tensor = uw(output_tensor_raw);
  const auto lora_indices_tensor = uw(lora_indices_tensor_raw);
  if (fake_like(inputs) || fake_like(lora_b_weights) ||
      fake_like(output_tensor) || fake_like(lora_indices_tensor)) return;
  dbg("bgmv_expand",
      "x=" + sh(inputs) + " w=" + sh(lora_b_weights) +
      " off=" + std::to_string(slice_offset));
  if (env_flag_off("TRITON_LORA_CPP") ||
      !triton_dtype(lora_b_weights) ||
      lora_b_weights.size(-1) != inputs.size(-1)) {
    fb_bgmv_expand(inputs, lora_b_weights, lora_indices_tensor,
                   output_tensor, slice_offset, slice_size);
    return;
  }
  int64_t B = inputs.size(0), R = inputs.size(1);
  auto w = lora_b_weights.reshape(
      {lora_b_weights.size(0), -1, lora_b_weights.size(-1)});
  int64_t L = w.size(0), Ho = w.size(1);
  int64_t blk = (R * 256 * 4 > 64 * 1024) ? 128 : 256;
  int64_t Y_HO = output_tensor.size(1);
  auto idx32 = cast_i32(lora_indices_tensor, 1);
  std::string key = build_key(
      "bgmv_expand",
      {{"R", std::to_string(R)}, {"Ho", std::to_string(Ho)},
       {"L", std::to_string(L)}, {"BLOCK_HO", std::to_string(blk)},
       {"Y_HO", std::to_string(Y_HO)},
       {"SLICE_OFF", std::to_string(slice_offset)}},
      {fmt_dtype(inputs.scalar_type()), fmt_dtype(w.scalar_type()),
       "torch.int32", fmt_dtype(output_tensor.scalar_type()),
       fmt_dtype(output_tensor.scalar_type())});
  uint64_t func = lookup_or_build(key);
  if (!func) {
    fb_bgmv_expand(inputs, lora_b_weights, lora_indices_tensor,
                   output_tensor, slice_offset, slice_size);
    return;
  }
  launch(func, static_cast<int32_t>(B),
         {inputs.data_ptr(), w.data_ptr(), idx32.data_ptr(),
          output_tensor.data_ptr(), output_tensor.data_ptr()},
         {});
}

void cpp_sgmv_shrink(const at::Tensor& inputs_raw,
                     const at::Tensor& lora_a_weights_raw,
                     at::Tensor output_tensor_raw,
                     const at::Tensor& seq_len_tensor_raw,
                     const at::Tensor& lora_indices_tensor_raw,
                     double scaling) {
  const auto inputs = uw(inputs_raw);
  const auto lora_a_weights = uw(lora_a_weights_raw);
  auto output_tensor = uw(output_tensor_raw);
  const auto seq_len_tensor = uw(seq_len_tensor_raw);
  const auto lora_indices_tensor = uw(lora_indices_tensor_raw);
  if (fake_like(inputs) || fake_like(lora_a_weights) ||
      fake_like(output_tensor) || fake_like(seq_len_tensor) ||
      fake_like(lora_indices_tensor)) return;
  dbg("sgmv_shrink",
      "x=" + sh(inputs) + " w=" + sh(lora_a_weights) +
      " idx=" + sh(lora_indices_tensor) +
      " seq=" + sh(seq_len_tensor));
  if (env_flag_off("TRITON_LORA_CPP") ||
      !triton_dtype(inputs) || !triton_dtype(lora_a_weights) ||
      lora_a_weights.size(-1) != inputs.size(-1) ||
      seq_len_tensor.numel() > 16) {
    fb_sgmv_shrink(inputs, lora_a_weights, lora_indices_tensor,
                   seq_len_tensor, output_tensor, scaling);
    return;
  }
  int64_t B = inputs.size(0), H = inputs.size(1);
  auto w = lora_a_weights.reshape(
      {lora_a_weights.size(0), -1, lora_a_weights.size(-1)});
  int64_t L = w.size(0), R = w.size(1), NR = seq_len_tensor.numel();
  auto idx32 = cast_i32(lora_indices_tensor, 1);
  auto seq32 = cast_i32(seq_len_tensor, 2);
  char sbuf[32];
  snprintf(sbuf, sizeof(sbuf), "%.1f", scaling);
  std::string key = build_key(
      "sgmv_shrink_kernel",
      {{"H", std::to_string(H)}, {"R", std::to_string(R)},
       {"L", std::to_string(L)}, {"NR", std::to_string(NR)},
       {"scale", sbuf}},
      {fmt_dtype(inputs.scalar_type()), fmt_dtype(w.scalar_type()),
       "torch.int32", "torch.int32",
       fmt_dtype(output_tensor.scalar_type())});
  uint64_t func = lookup_or_build(key);
  if (!func) {
    fb_sgmv_shrink(inputs, lora_a_weights, lora_indices_tensor,
                   seq_len_tensor, output_tensor, scaling);
    return;
  }
  launch(func, static_cast<int32_t>(B),
         {inputs.data_ptr(), w.data_ptr(), idx32.data_ptr(),
          seq32.data_ptr(), output_tensor.data_ptr()},
         {});
}

void cpp_sgmv_expand_slice(const at::Tensor& inputs_raw,
                           const at::Tensor& lora_b_weights_raw,
                           at::Tensor output_tensor_raw,
                           const at::Tensor& seq_len_tensor_raw,
                           const at::Tensor& lora_indices_tensor_raw,
                           int64_t slice_offset, int64_t slice_size) {
  const auto inputs = uw(inputs_raw);
  const auto lora_b_weights = uw(lora_b_weights_raw);
  auto output_tensor = uw(output_tensor_raw);
  const auto seq_len_tensor = uw(seq_len_tensor_raw);
  const auto lora_indices_tensor = uw(lora_indices_tensor_raw);
  if (fake_like(inputs) || fake_like(lora_b_weights) ||
      fake_like(output_tensor) || fake_like(seq_len_tensor) ||
      fake_like(lora_indices_tensor)) return;
  dbg("sgmv_expand",
      "x=" + sh(inputs) + " w=" + sh(lora_b_weights) +
      " seq=" + sh(seq_len_tensor) +
      " off=" + std::to_string(slice_offset));
  if (env_flag_off("TRITON_LORA_CPP") ||
      !triton_dtype(lora_b_weights) ||
      lora_b_weights.size(-1) != inputs.size(-1) ||
      seq_len_tensor.numel() > 16) {
    fb_sgmv_expand(inputs, lora_b_weights, lora_indices_tensor,
                   seq_len_tensor, output_tensor, slice_offset, slice_size);
    return;
  }
  int64_t B = inputs.size(0), R = inputs.size(1);
  auto w = lora_b_weights.reshape(
      {lora_b_weights.size(0), -1, lora_b_weights.size(-1)});
  int64_t L = w.size(0), Ho = w.size(1), NR = seq_len_tensor.numel();
  int64_t blk = (R * 256 * 4 > 64 * 1024) ? 128 : 256;
  int64_t Y_HO = output_tensor.size(1);
  auto idx32 = cast_i32(lora_indices_tensor, 1);
  auto seq32 = cast_i32(seq_len_tensor, 2);
  std::string key = build_key(
      "sgmv_expand",
      {{"R", std::to_string(R)}, {"Ho", std::to_string(Ho)},
       {"L", std::to_string(L)}, {"NR", std::to_string(NR)},
       {"BLOCK_HO", std::to_string(blk)},
       {"Y_HO", std::to_string(Y_HO)},
       {"SLICE_OFF", std::to_string(slice_offset)}},
      {fmt_dtype(inputs.scalar_type()), fmt_dtype(w.scalar_type()),
       "torch.int32", "torch.int32",
       fmt_dtype(output_tensor.scalar_type()),
       fmt_dtype(output_tensor.scalar_type())});
  uint64_t func = lookup_or_build(key);
  if (!func) {
    fb_sgmv_expand(inputs, lora_b_weights, lora_indices_tensor,
                   seq_len_tensor, output_tensor, slice_offset, slice_size);
    return;
  }
  launch(func, static_cast<int32_t>(B),
         {inputs.data_ptr(), w.data_ptr(), idx32.data_ptr(),
          seq32.data_ptr(), output_tensor.data_ptr(),
          output_tensor.data_ptr()},
         {});
}

TORCH_LIBRARY_IMPL(vllm_ascend_triton, PrivateUse1, m) {
  m.impl("bgmv_shrink", &cpp_bgmv_shrink);
  m.impl("bgmv_expand_slice", &cpp_bgmv_expand_slice);
  m.impl("sgmv_shrink", &cpp_sgmv_shrink);
  m.impl("sgmv_expand_slice", &cpp_sgmv_expand_slice);
}

// Same kernels on the Python key: the PrivateUse1 key is not in the dispatch
// key set of every tensor the serving path dispatches with (dynamo graph
// execution / capture), so those calls never reach the kernels above and
// LoRA silently no-ops.  The Python key outranks PrivateUse1 and matches
// every real tensor, so dispatch always lands here; FakeTensor-mode tracing
// still hits the Fake key (register_fake) which outranks Python.
TORCH_LIBRARY_IMPL(vllm_ascend_triton, Python, m) {
  m.impl("bgmv_shrink", &cpp_bgmv_shrink);
  m.impl("bgmv_expand_slice", &cpp_bgmv_expand_slice);
  m.impl("sgmv_shrink", &cpp_sgmv_shrink);
  m.impl("sgmv_expand_slice", &cpp_sgmv_expand_slice);
}

}  // namespace
