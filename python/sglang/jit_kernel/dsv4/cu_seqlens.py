"""GPU-resident helper to build DeepGEMM `cu_seqlens` from `extend_start_loc`.

DSV4 prefill already computes `extend_start_loc`: a `[bs]` int32 tensor giving
each request's exclusive-prefix-sum start offset into the (padded) query rows.
DeepGEMM's varlen paged MQA logits kernels take `cu_seqlens`: the same prefix
sum but with a trailing total appended, i.e. shape `[bs + 1]`.

This is a single short append -- no segment identification, no per-token
work -- so a tiny one-program Triton kernel is enough, and it stays entirely
on GPU (no `.item()`/`.tolist()`), which is required for CUDA-graph capture
and to avoid a host sync on the prefill hot path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _append_cu_seqlens_kernel(
    extend_start_loc_ptr,
    out_ptr,
    batch_size,
    total,
    BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch_size
    start_loc = tl.load(extend_start_loc_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, start_loc, mask=mask)
    if pid == 0:
        tl.store(out_ptr + batch_size, total)


def triton_append_cu_seqlens(
    extend_start_loc: torch.Tensor,
    *,
    query_rows: int,
    block: int = 1024,
) -> torch.Tensor:
    """Build DeepGEMM-ABI `cu_seqlens` = `extend_start_loc` + trailing total.

    Args:
        extend_start_loc: `[bs]` int32 GPU tensor, exclusive prefix sum of
            per-request query lengths (`forward_batch.extend_start_loc`).
        query_rows: total number of query rows (the trailing/total entry).
        block: Triton block size for copying `extend_start_loc`.

    Returns:
        `[bs + 1]` int32 GPU tensor: `cu_seqlens` per DeepGEMM's ABI.
    """
    batch_size = extend_start_loc.shape[0]
    out = torch.empty(
        (batch_size + 1,), device=extend_start_loc.device, dtype=torch.int32
    )
    grid = (triton.cdiv(batch_size, block),)
    _append_cu_seqlens_kernel[grid](
        extend_start_loc,
        out,
        batch_size,
        query_rows,
        BLOCK=block,
    )
    return out
