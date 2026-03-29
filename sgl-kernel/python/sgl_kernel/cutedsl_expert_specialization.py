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
    num_experts = int(problem_sizes.shape[0])
    for g in range(num_experts):
        m_g = int(problem_sizes[g, 0].item())
        n_g = int(problem_sizes[g, 1].item())
        k_g = int(problem_sizes[g, 2].item())
        start = int(expert_offsets[g].item())
        end = start + m_g

        a_g = a[start:end, :k_g]
        b_g = b[g, :k_g, :n_g]
        sa_g = scales_a[start:end]
        sb_g = scales_b[g]
        sa_full = _group_broadcast(sa_g, a_g.shape)
        sb_full = _group_broadcast(sb_g, b_g.shape)
        output[start:end, :n_g] = torch.mm(
            (sa_full * a_g.to(torch.float32)),
            (sb_full * b_g.to(torch.float32)),
        ).to(output.dtype)
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
    # Fallback reference implementation for SM100 grouped MM.
    # NOTE: scale-factor decoding from blockscaled uint8 tensors is not yet implemented.
    num_experts = int(problem_sizes.shape[0])
    for g in range(num_experts):
        m_g = int(problem_sizes[g, 0].item())
        n_g = int(problem_sizes[g, 1].item())
        k_g = int(problem_sizes[g, 2].item())
        start = int(expert_offsets[g].item())
        end = start + m_g

        a_g = a[start:end, :k_g].to(torch.float32)
        b_g = b[g, :k_g, :n_g].to(torch.float32)
        output[start:end, :n_g] = torch.mm(a_g, b_g).to(output.dtype)
    return output
