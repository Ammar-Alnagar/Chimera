import torch


def _group_broadcast(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
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


def blockwise_gemm_kernel(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    # Python implementation of blockwise-scaled FP8 GEMM.
    scale_a = _group_broadcast(scales_a, mat_a.shape)
    scale_b = _group_broadcast(scales_b, mat_b.shape)
    return torch.mm(
        (scale_a * mat_a.to(torch.float32)),
        (scale_b * mat_b.to(torch.float32)),
    ).to(out_dtype)


def cutedsl_fp8_blockwise_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return blockwise_gemm_kernel(mat_a, mat_b, scales_a, scales_b, out_dtype)
