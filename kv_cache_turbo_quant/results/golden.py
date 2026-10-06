"""Torch NPU golden for KvCacheTurboQuant."""

import numpy as np
import torch
import torch_npu  # noqa: F401
from ml_dtypes import bfloat16


CENTROIDS = {
    2: [-0.1335033178, -0.04002048075, 0.04002048075, 0.1335033178],
    3: [-0.19020693, -0.1187859178, -0.06682205945, -0.02166347019,
        0.02166347019, 0.06682205945, 0.1187859178, 0.19020693],
    4: [-0.2414890379, -0.1828317791, -0.1429702938, -0.1109927073,
        -0.08325428516, -0.05802082643, -0.03428063914, -0.01134236995,
        0.01134236995, 0.03428063914, 0.05802082643, 0.08325428516,
        0.1109927073, 0.1429702938, 0.1828317791, 0.2414890379],
}


def _npu_fp32(value):
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).npu().contiguous()


def _numpy(value):
    return value.detach().cpu().numpy()


def _bf16_numpy(value):
    return np.asarray(value.detach().float().cpu().numpy(), dtype=bfloat16)


def _pack_bits(values, bit_width):
    values = values.to(torch.int64)
    groups = (values.shape[-1] + 7) // 8

    if values.shape[-1] % 8:
        values = torch.nn.functional.pad(values, (0, 8 - values.shape[-1] % 8))

    values = values.reshape(*values.shape[:-1], groups, 8)
    word = values[..., 0].clone()

    for lane in range(1, 8):
        word |= values[..., lane] << (lane * bit_width)

    if bit_width == 1:
        return word.to(torch.uint8)

    packed = torch.stack(
        [(word >> (8 * byte)).to(torch.uint8) for byte in range(bit_width)], -1
    )
    return packed.reshape(*packed.shape[:-2], groups * bit_width)


def calc_expect_func(kv_vectors, rotation_matrix, qjl_matrix, mse_bits):
    bits = int(mse_bits)
    if bits not in (2, 3, 4):
        raise ValueError("mse_bits must be 2, 3, or 4")

    x = _npu_fp32(kv_vectors)
    rotation = _npu_fp32(rotation_matrix)
    qjl = _npu_fp32(qjl_matrix)

    norm = torch.linalg.vector_norm(x, dim=-1)
    unit = torch.where(
        (norm > 0).unsqueeze(-1),
        x / norm.clamp_min(1e-30).unsqueeze(-1),
        torch.zeros_like(x),
    )

    rotated = unit @ rotation.T

    centroids = torch.tensor(CENTROIDS[bits], dtype=torch.float32, device=x.device)
    boundaries = (centroids[:-1] + centroids[1:]) * 0.5
    indices = (rotated.unsqueeze(-1) > boundaries).sum(dim=-1).to(torch.uint8)

    primary = centroids[indices.long()]
    residual = rotated - primary

    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    gamma = norm * residual_norm

    residual_unit = torch.where(
        (residual_norm > 0).unsqueeze(-1),
        residual / residual_norm.clamp_min(1e-30).unsqueeze(-1),
        torch.zeros_like(residual),
    )

    projected = residual_unit @ qjl.T

    quant_idx = _pack_bits(indices, bits)
    quant_qjl = _pack_bits((projected >= 0).to(torch.uint8), 1)

    torch.npu.synchronize()

    return [
        _numpy(quant_idx),
        _numpy(quant_qjl),
        _bf16_numpy(norm),
        _bf16_numpy(gamma),
    ]