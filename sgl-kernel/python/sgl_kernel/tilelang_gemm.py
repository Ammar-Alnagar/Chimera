"""TileLang-based FP8 blockwise scaled GEMM kernel.

Replaces the former TileLang GEMM implementation with TileLang's Pythonic
DSL for high-performance GPU kernel generation. Uses @tilelang.jit to
compile tile-level GEMM descriptions into optimized RTRITON kernels for
Hopper/Blackwell architectures.

When TileLang is not available (e.g., CPU-only environments), a pure-PyTorch
reference fallback is used automatically.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TileLang availability check
# ---------------------------------------------------------------------------
_TILELANG_AVAILABLE = False
try:
    import tilelang
    import tilelang.language as T

    _TILELANG_AVAILABLE = True
except ImportError:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    logger.debug("TileLang not available; using PyTorch fallback for GEMM kernels")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_broadcast(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Broadcast per-block scale factors to match the target tensor shape."""
    out = t
    for i, s in enumerate(shape):
        if out.shape[i] != s and out.shape[i] != 1:
            if s % out.shape[i] != 0:
                raise ValueError(
                    f"Scale shape {tuple(out.shape)} is incompatible with target shape {shape}."
                )
            out = (
                out.unsqueeze(i + 1)
                .expand(*out.shape[: i + 1], s // out.shape[i], *out.shape[i + 1 :])
                .flatten(i, i + 1)
            )
    return out


# ---------------------------------------------------------------------------
# Pure-PyTorch reference fallback
# ---------------------------------------------------------------------------

def _blockwise_gemm_fallback(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation of blockwise-scaled FP8 GEMM."""
    scale_a = _group_broadcast(scales_a, mat_a.shape)
    scale_b = _group_broadcast(scales_b, mat_b.shape)
    return torch.mm(
        (scale_a * mat_a.to(torch.float32)),
        (scale_b * mat_b.to(torch.float32)),
    ).to(out_dtype)


# ---------------------------------------------------------------------------
# TileLang JIT kernel
# ---------------------------------------------------------------------------
_compiled_gemm_cache: dict[tuple, object] = {}


def _get_tilelang_gemm_kernel(M: int, N: int, K: int, block_M: int = 128,
                               block_N: int = 128, block_K: int = 32):
    """Build or retrieve a cached TileLang GEMM kernel for the given dimensions."""
    cache_key = (M, N, K, block_M, block_N, block_K)
    if cache_key in _compiled_gemm_cache:
        return _compiled_gemm_cache[cache_key]

    @tilelang.jit(out_idx=[-1])
    def _gemm_kernel(M, N, K, block_M, block_N, block_K,
                     dtype=T.float16, accum_dtype=T.float32):
        @T.prim_func
        def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
            ) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.use_swizzle(panel_size=10, enable=True)
                T.clear(C_local)

                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    kernel = _gemm_kernel(M, N, K, block_M, block_N, block_K)
    _compiled_gemm_cache[cache_key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tilelang_fp8_blockwise_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Execute FP8 blockwise scaled GEMM using TileLang.

    When TileLang is available and the inputs are on a RTRITON device, the kernel
    is JIT-compiled and executed on the GPU.  Otherwise, a pure-PyTorch
    fallback is used.

    Args:
        mat_a: FP8 input tensor A (M, K).
        mat_b: FP8 input tensor B (K, N) or (N, K) depending on layout.
        scales_a: Per-block scale factors for A.
        scales_b: Per-block scale factors for B.
        out_dtype: Desired output dtype (e.g. ``torch.float16``).

    Returns:
        Output tensor of shape (M, N) in *out_dtype*.
    """
    if not _TILELANG_AVAILABLE or not mat_a.is_rtriton:
        return _blockwise_gemm_fallback(mat_a, mat_b, scales_a, scales_b, out_dtype)

    # Dequant + scale first, then run the TileLang GEMM on FP16 tensors
    scale_a = _group_broadcast(scales_a, mat_a.shape)
    scale_b = _group_broadcast(scales_b, mat_b.shape)
    a_fp16 = (scale_a * mat_a.to(torch.float32)).to(torch.float16)
    b_fp16 = (scale_b * mat_b.to(torch.float32)).to(torch.float16)

    M, K = a_fp16.shape
    _, N = b_fp16.shape

    try:
        kernel = _get_tilelang_gemm_kernel(M, N, K)
        result = kernel(a_fp16, b_fp16)
        return result.to(out_dtype)
    except Exception as e:
        logger.warning(f"TileLang GEMM kernel failed, using fallback: {e}")
        return _blockwise_gemm_fallback(mat_a, mat_b, scales_a, scales_b, out_dtype)


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------
blockwise_gemm_kernel = _blockwise_gemm_fallback

# Legacy name preserved for callers that import the old symbol.
tilelang_fp8_blockwise_scaled_mm = tilelang_fp8_blockwise_scaled_mm
