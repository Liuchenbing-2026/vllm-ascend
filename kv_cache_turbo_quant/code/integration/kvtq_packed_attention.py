"""Explicit standalone packed-attention adapter; serving dispatch is unchanged."""

import hashlib
import pathlib

import torch


def checked_library(path, expected_sha256):
    path = pathlib.Path(path).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("library SHA256 mismatch: " + str(path))
    return path


class NativePackedAttention:
    def __init__(self, binding, binding_sha256, op_api, op_api_sha256, *, api_scratch_bytes):
        self.binding = checked_library(binding, binding_sha256)
        self.op_api = checked_library(op_api, op_api_sha256)
        if isinstance(api_scratch_bytes, bool) or not isinstance(api_scratch_bytes, int):
            raise ValueError("scratch reservation must be an explicit integer")
        if not 0 <= api_scratch_bytes <= 2 ** 32 - 1:
            raise ValueError("scratch reservation is outside uint32")
        self.api_scratch_bytes = api_scratch_bytes
        torch.ops.load_library(str(self.binding))
        if not torch._C._dispatch_has_kernel_for_dispatch_key("turboquant::tq_packed_attention", "PrivateUse1"):
            raise RuntimeError("binding has no NPU implementation; CPU-only Meta library is not a device binding")
        self.operator = torch.ops.turboquant.tq_packed_attention.default

    def rotated(self, query_rot, key, value, table, lengths, centroids, *, scale):
        """Raw core output and status; caller MUST inspect status before use."""
        return self.operator(query_rot, key, value, table, lengths, centroids,
                             scale, self.api_scratch_bytes, str(self.op_api))

    @staticmethod
    def require_success(status):
        # Status handling/synchronization is part of complete attention cost.
        host_status = status.cpu()
        if torch.count_nonzero(host_status).item():
            raise RuntimeError("TQ packed device status failure: " + repr(host_status.tolist()))

    def __call__(self, query, key, value, table, lengths, centroids, rotation, *, scale):
        if rotation.shape != (128, 128) or rotation.dtype != torch.float32 or rotation.device != query.device:
            raise ValueError("rotation must be same-device FP32 [128,128]")
        query_rot = (query.float() @ rotation.T).to(torch.bfloat16)
        output_rot, status = self.rotated(query_rot, key, value, table, lengths, centroids, scale=scale)
        self.require_success(status)
        return (output_rot.float() @ rotation).to(torch.bfloat16)
