import torch
import triton
import triton.language as tl


@triton.jit
def gather_kvcache_kernel(
    k_cache_ptr,
    v_cache_ptr,
    src_slot_ptr,
    dst_slot_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    src_slot = tl.load(src_slot_ptr + idx)
    if src_slot == -1:
        return
    dst_slot = tl.load(dst_slot_ptr + idx)
    src_cache_offsets = src_slot * D + tl.arange(0, D)
    dst_cache_offsets = dst_slot * D + tl.arange(0, D)
    key = tl.load(k_cache_ptr + src_cache_offsets)
    value = tl.load(v_cache_ptr + src_cache_offsets)
    tl.store(k_cache_ptr + dst_cache_offsets, key)
    tl.store(v_cache_ptr + dst_cache_offsets, value)


def gather_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor, src_slot_ptr: torch.Tensor, dst_slot_ptr: torch.Tensor):
    _, page_size, num_heads, head_dim = k_cache.shape
    D = tl.constexpr(num_heads * head_dim)
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    N = src_slot_ptr.numel()
    assert src_slot_ptr.numel() == dst_slot_ptr.numel()
    gather_kvcache_kernel[(N,)](k_cache, v_cache, src_slot_ptr, dst_slot_ptr, D)
