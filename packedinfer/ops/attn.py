from packed_attn_gpu import packed_attn  # ty:ignore[unresolved-import]


def packed_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    leftpad,
    rightlim=None,
    prefix_pad=None,
    lse=None,
    block_table=None,
):
    packed_attn(
        q,
        k,
        v,
        out,
        lse,
        prefix_pad,
        cu_seqlens_q,
        cu_seqlens_k,
        leftpad,
        rightlim,
        block_table,
        max_seqlen_q,
        1 / (q.size(-1) ** 0.5),
    )
