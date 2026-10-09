import gc

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd, l2norm_packed_qk
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize("tokens", [1, 63, 256, 1025])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_packed_qk_matches_separate_npu_normalization(tokens, dtype):
    init_device_properties_triton()
    packed = torch.randn(2 * tokens * 4 + 1, 128, device="npu", dtype=dtype)
    q = packed[1 : 1 + tokens * 4].view(1, tokens, 4, 128)
    k = packed[1 + tokens * 4 :].view_as(q)
    actual_q, actual_k = l2norm_packed_qk(q, k)
    torch.testing.assert_close(actual_q, l2norm_fwd(q), rtol=0, atol=0)
    torch.testing.assert_close(actual_k, l2norm_fwd(k), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 60, torch.float),
            (2, 500, 4, 64, torch.float),
            (2, 1000, 2, 100, torch.float),
            (3, 1024, 4, 128, torch.float),
        ]
    ],
)
def test_l2norm(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    x = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    x = x * 0.5 + 0.3

    ref = F.normalize(x, dim=-1, p=2)
    tri = l2norm_fwd(x)

    assert torch.allclose(tri, ref, rtol=rtol, atol=atol)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
