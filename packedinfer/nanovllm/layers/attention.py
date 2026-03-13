import math
import os

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from packedinfer.ops.attn import packed_attention
from torch import nn

from packedinfer.ops.attn import packed_attention
from packedinfer.ops.gather import gather_kvcache

from ..utils.context import get_context

_ATTN_EVENTS = []


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # attention_mask is only used in prepacking baseline
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        profile_timing = os.getenv("PROFILE_TIMING", "0") == "1"
        if profile_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        if k_cache.numel() and v_cache.numel():
            if attention_mask is None:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            else:
                k_flat = k.reshape(-1, self.num_kv_heads, self.head_dim)
                v_flat = v.reshape(-1, self.num_kv_heads, self.head_dim)
                store_kvcache(k_flat, v_flat, k_cache, v_cache, context.slot_mapping)

        if context.block_tables is not None:  # prefix cache
            k, v = k_cache, v_cache
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=q.dtype)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            o = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=self.scale,
                is_causal=False,
            )
            if profile_timing:
                end_event.record()
                _ATTN_EVENTS.append(
                    {
                        "start": start_event,
                        "end": end_event,
                        "run_id": context.run_id,
                        "is_prefill": context.is_prefill,
                        "backend": "SDPA",
                        "batch_lens": context.seq_lens,
                    }
                )
            return o.transpose(1, 2).contiguous()
        elif os.getenv("ATTN_BACKEND") == "FLASH_ATTN":
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        elif os.getenv("ATTN_BACKEND") == "PACKED_ATTN":
            o = torch.empty_like(q)
            # gather_kvcache(k_cache, v_cache, context.packed_src_slots, context.packed_dst_slots)
            packed_attention(
                q,
                k,
                v,
                o,
                context.packed_cu_seqlens_q,
                context.packed_cu_seqlens_k,
                context.packed_max_seqlen,
                context.leftpad,
                rightlim=context.rightlim,
                prefix_pad=context.prefix_pad,
                block_table=context.packed_block_tables,
            )
        else:
            raise ValueError(f"Unknown ATTN_BACKEND: {os.getenv('ATTN_BACKEND')}")

        if profile_timing:
            end_event.record()
            _ATTN_EVENTS.append(
                {
                    "start": start_event,
                    "end": end_event,
                    "run_id": context.run_id,
                    "is_prefill": context.is_prefill,
                    "backend": os.getenv("ATTN_BACKEND", "PACKED_ATTN"),
                    "batch_lens": context.seq_lens,
                }
            )
        return o
