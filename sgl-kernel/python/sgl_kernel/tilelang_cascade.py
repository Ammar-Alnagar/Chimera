import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def _merge_state_kernel(
    v_a: T.Buffer,
    s_a: T.Buffer,
    v_b: T.Buffer,
    s_b: T.Buffer,
    v_merged: T.Buffer,
    s_merged: T.Buffer,
):
    seq_len, num_heads, head_dim = v_a.shape
    
    for i, j in T.Parallel(seq_len, num_heads):
        s_a_val = s_a[i, j]
        s_b_val = s_b[i, j]
        
        s_max = T.max(s_a_val, s_b_val)
        s_merged_val = s_max + T.log(T.exp(s_a_val - s_max) + T.exp(s_b_val - s_max))
        s_merged[i, j] = s_merged_val
        
        p_a = T.exp(s_a_val - s_merged_val)
        p_b = T.exp(s_b_val - s_merged_val)
        
        for k in T.serial(head_dim):
            v_merged[i, j, k] = p_a * v_a[i, j, k] + p_b * v_b[i, j, k]

def tilelang_merge_state(v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor, v_merged: torch.Tensor, s_merged: torch.Tensor):
    _merge_state_kernel(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged
