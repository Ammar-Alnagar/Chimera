import torch


def expert_specialization_kernel(
    output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    stride_a: torch.Tensor,
    stride_b: torch.Tensor,
    stride_d: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    torch.ops.sgl_kernel.es_fp8_blockwise_scaled_grouped_mm.default(
        output,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_d,
        problem_sizes,
        expert_offsets,
        workspace,
    )
    return output


def cutedsl_es_fp8_blockwise_scaled_grouped_mm(
    output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    stride_a: torch.Tensor,
    stride_b: torch.Tensor,
    stride_d: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    return expert_specialization_kernel(
        output,
        a,
        b,
        scales_a,
        scales_b,
        stride_a,
        stride_b,
        stride_d,
        problem_sizes,
        expert_offsets,
        workspace,
    )


def cutedsl_es_sm100_mxfp8_blockscaled_grouped_mm(
    output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
) -> torch.Tensor:
    torch.ops.sgl_kernel.es_sm100_mxfp8_blockscaled_grouped_mm.default(
        a, b, sfa, sfb, output, problem_sizes, expert_offsets, blockscale_offsets
    )
    return output
