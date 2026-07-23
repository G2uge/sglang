"""MoH (Mixture of Heads) fully-fused head selection — gate GEMM folded in.

The default MoH path (`fp8_paged_mqa_moh_select_heads_torch` in `indexer.py`)
runs the gate GEMM on cuBLAS, then a chain of launch-bound elementwise / top-k /
gather ops. This module folds *all* of it — including the gate GEMM via
`tl.dot` — into a single Triton kernel:

    tl.dot GEMM (hidden @ gate_wT) -> sqrt(softplus)+bias -> 8-pass argmax top-k
    -> vectorized gather of the selected heads' q.

Fusing the GEMM lets the scheduler overlap the Tensor-Core matmul of some rows
with the memory-bound gather of others, giving ~2x over the split-GEMM path.
Trade-off: `tl.dot`'s bf16 accumulation rounds differently from cuBLAS, flipping
~0.5% of boundary head slots — so this path breaks train/inference bit-parity
and is gated behind `SGLANG_OPT_DSV4_MOH_FUSED_GEMM` (off by default).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _moh_fused_gemm_kernel(
    hidden_ptr,        # [B, HIDDEN] bf16
    weight_ptr,        # [NH, HIDDEN] bf16   (Linear weight, row = head)
    bias_ptr,          # [NH] fp32
    w_gate_ptr,        # [B, NH] fp32        (per-row per-head weights)
    q_i32_ptr,         # [B, NH, HDI32] int32
    q_sel_i32_ptr,     # [B, TOPK, HDI32] int32 (out)
    w_sel_ptr,         # [B, TOPK] fp32 (out)
    idx_ptr,           # [B, TOPK] int32 (out)
    B,
    HIDDEN: tl.constexpr,
    NH: tl.constexpr,
    TOPK: tl.constexpr,
    HDI32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)              # [BM]
    m_mask = offs_m < B
    offs_n = tl.arange(0, NH)                                   # [NH]

    # ── (a) GEMM: scores[BM,NH] = hidden[BM,K] @ weightT[K,NH] ────────────────
    acc = tl.zeros((BLOCK_M, NH), dtype=tl.float32)
    for k0 in range(0, HIDDEN, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)                    # [BK]
        h_tile = tl.load(
            hidden_ptr + offs_m[:, None] * HIDDEN + offs_k[None, :],
            mask=m_mask[:, None], other=0.0)                   # [BM,BK] bf16
        # weight[n,k] -> need [BK,NH]: w_tile[k,n] = weight_ptr[n*HIDDEN + k]
        w_tile = tl.load(
            weight_ptr + offs_n[None, :] * HIDDEN + offs_k[:, None])  # [BK,NH] bf16
        acc = tl.dot(h_tile, w_tile, acc)                      # [BM,NH]

    # ── (b) act + (c) bias ────────────────────────────────────────────────────
    sp = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(acc)))
    gates_act = tl.sqrt(sp + 1e-20)                            # [BM,NH]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n).to(tl.float32)       # [NH]
        scores = gates_act + bias[None, :]
    else:
        scores = gates_act
    w_vec = tl.load(w_gate_ptr + offs_m[:, None] * NH + offs_n[None, :],
                    mask=m_mask[:, None], other=0.0)           # [BM,NH]

    # ── (d) 8-pass argmax along head dim (batched over BM rows) ───────────────
    d = tl.arange(0, HDI32)                                    # [HDI32]
    NEG_INF = -float("inf")
    cur = scores
    for i in range(TOPK):
        idx = tl.argmax(cur, axis=1).to(tl.int32)              # [BM] i-th largest head
        sel_mask = offs_n[None, :] == idx[:, None]             # [BM,NH]
        cur = tl.where(sel_mask, NEG_INF, cur)
        gate_at = tl.sum(tl.where(sel_mask, gates_act, 0.0), axis=1)  # [BM]
        w_at = tl.sum(tl.where(sel_mask, w_vec, 0.0), axis=1)         # [BM]
        tl.store(w_sel_ptr + offs_m * TOPK + i, w_at * gate_at, mask=m_mask)
        tl.store(idx_ptr + offs_m * TOPK + i, idx, mask=m_mask)
        # (e) vectorized gather q: [BM, HDI32] moves 128B/head in one pass
        q_off = offs_m[:, None] * NH * HDI32 + idx[:, None] * HDI32 + d[None, :]
        q_vec = tl.load(q_i32_ptr + q_off, mask=m_mask[:, None], other=0)  # [BM,HDI32]
        out_off = offs_m[:, None] * TOPK * HDI32 + i * HDI32 + d[None, :]
        tl.store(q_sel_i32_ptr + out_off, q_vec, mask=m_mask[:, None])


def fp8_paged_mqa_moh_select_heads_fused_gemm(
    q_fp8: torch.Tensor,
    weight: torch.Tensor,
    moh_gate: nn.Linear,
    topk_heads: int,
    hidden_state: torch.Tensor,
    indexer_bias: Optional[torch.Tensor] = None,
    BLOCK_M: int = 32,
    BLOCK_K: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully-fused MoH head selection: a single Triton kernel produces the
    selected q / weights directly from `hidden_state`.

    Drop-in replacement for `fp8_paged_mqa_moh_select_heads_torch` — same
    inputs, same `(q_selected, weight_selected)` return — but folds the gate
    GEMM into the kernel. See the module docstring for the precision caveat.
    """
    batch_size, _, num_heads, head_dim = q_fp8.shape
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert weight.shape == (batch_size, num_heads)
    assert hidden_state.shape[0] == batch_size
    topk_heads = min(topk_heads, num_heads)

    hidden_size = hidden_state.shape[-1]
    head_dim_i32 = head_dim // 4

    gate_w = moh_gate.weight.contiguous()                      # [NH, HIDDEN] bf16
    hs = hidden_state.contiguous()                             # [B, HIDDEN] bf16
    q_i32 = q_fp8[:, 0].contiguous().view(torch.int32)         # [B, NH, HDI32]
    w_f32 = weight.contiguous().float()                        # [B, NH]

    q_sel_i32 = torch.empty(
        batch_size, topk_heads, head_dim_i32, device=q_fp8.device, dtype=torch.int32
    )
    w_sel = torch.empty(
        batch_size, topk_heads, device=q_fp8.device, dtype=torch.float32
    )
    idx = torch.empty(
        batch_size, topk_heads, device=q_fp8.device, dtype=torch.int32
    )

    has_bias = indexer_bias is not None
    bias_arg = indexer_bias.contiguous().float() if has_bias else w_f32

    grid = (triton.cdiv(batch_size, BLOCK_M),)
    _moh_fused_gemm_kernel[grid](
        hs, gate_w, bias_arg, w_f32, q_i32, q_sel_i32, w_sel, idx,
        batch_size,
        HIDDEN=hidden_size, NH=num_heads, TOPK=topk_heads, HDI32=head_dim_i32,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, HAS_BIAS=has_bias,
    )
    q_selected = q_sel_i32.view(torch.uint8).view(q_fp8.dtype).unsqueeze(1)
    return q_selected, w_sel
