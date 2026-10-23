# SPDX-License-Identifier: Apache-2.0
"""Conservative dispatch policy shared by config resolution and selection."""

# These preserve the measured slow-shape protection. Passing the guard is
# eligibility for further validation, not a performance guarantee.
_A2_BF16_MIN_EP_SIZE = 16
_A2_BF16_MAX_EXPERTS_PER_RANK = 24


def a2_bf16_megamoe_shape_rejection(ep_size: int, num_experts: int) -> str | None:
    if ep_size < 1 or num_experts < 1 or num_experts % ep_size:
        return "experts must be evenly divisible across a nonempty EP group"
    experts_per_rank = num_experts // ep_size
    if ep_size < _A2_BF16_MIN_EP_SIZE or experts_per_rank > _A2_BF16_MAX_EXPERTS_PER_RANK:
        return (
            f"ep={ep_size}, experts_per_rank={experts_per_rank}; "
            f"the conservative BF16 policy requires ep>={_A2_BF16_MIN_EP_SIZE} "
            f"and experts_per_rank<={_A2_BF16_MAX_EXPERTS_PER_RANK}"
        )
    return None
