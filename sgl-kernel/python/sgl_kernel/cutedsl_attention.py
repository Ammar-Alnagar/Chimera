import torch
import torch.nn.functional as F


def mla_decode_kernel(
    out: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int,
) -> torch.Tensor:
    del workspace, num_kv_splits

    q = torch.cat((q_nope, q_pe), dim=-1)
    for i in range(q.shape[0]):
        seq_len = int(seq_lens[i].item())
        kv = kv_c_and_k_pe_cache[page_table[i]].reshape(-1, kv_c_and_k_pe_cache.shape[-1])
        kv = kv[:seq_len].unsqueeze(0)
        v = kv[:, :, : out.shape[-1]]
        qi = q[i].unsqueeze(1)
        oi = F.scaled_dot_product_attention(qi, kv, v, scale=sm_scale, enable_gqa=True)
        out[i] = oi.squeeze(1).to(out.dtype)
    return out


def cutedsl_mla_decode(
    out: torch.Tensor,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int,
) -> torch.Tensor:
    return mla_decode_kernel(
        out,
        q_nope,
        q_pe,
        kv_c_and_k_pe_cache,
        seq_lens,
        page_table,
        workspace,
        sm_scale,
        num_kv_splits,
    )
