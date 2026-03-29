import torch


def blockwise_gemm_kernel(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.fp8_blockwise_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
    )


def cutedsl_fp8_blockwise_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return blockwise_gemm_kernel(mat_a, mat_b, scales_a, scales_b, out_dtype)
