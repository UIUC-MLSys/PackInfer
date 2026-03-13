from dataclasses import dataclass

import torch


@dataclass
class Context:
    is_prefill: bool = False
    seq_lens: list[int] | None = None
    run_id: int = -1
    pack_group_lens: list[int] | None = None
    pack_group_ratios: list[float] | None = None
    pack_group_max: int = 0
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # Packed Attn
    leftpad: torch.Tensor | None = None
    packed_cu_seqlens_q: torch.Tensor | None = None
    packed_cu_seqlens_k: torch.Tensor | None = None
    packed_max_seqlen: int = 0
    packed_block_tables: torch.Tensor | None = None
    packed_src_slots: torch.Tensor | None = None
    packed_dst_slots: torch.Tensor | None = None
    prefix_pad: torch.Tensor | None = None
    rightlim: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill,
    seq_lens=None,
    run_id=-1,
    pack_group_lens=None,
    pack_group_ratios=None,
    pack_group_max=0,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    leftpad=None,
    packed_cu_seqlens_q=None,
    packed_cu_seqlens_k=None,
    packed_max_seqlen=0,
    packed_block_tables=None,
    packed_src_slots=None,
    packed_dst_slots=None,
    prefix_pad=None,
    rightlim=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        seq_lens,
        run_id,
        pack_group_lens,
        pack_group_ratios,
        pack_group_max,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        leftpad,
        packed_cu_seqlens_q,
        packed_cu_seqlens_k,
        packed_max_seqlen,
        packed_block_tables,
        packed_src_slots,
        packed_dst_slots,
        prefix_pad,
        rightlim,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
