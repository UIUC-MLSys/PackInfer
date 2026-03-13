#pragma once

#include <cute/tensor.hpp>

#include "copy.hpp"
#include "gemm.hpp"
#include "mask.hpp"
#include "params.hpp"
#include "softmax.hpp"

using namespace cute;

template <typename Config>
__global__ void
__launch_bounds__(Config::kNThreads) packed_attn_kernel(__grid_constant__ const Params params) {
    static constexpr int kNQOHeads = Config::kNQOHeads;
    static constexpr int kNKVHeads = Config::kNKVHeads;
    static constexpr int kNHeadsRatio = kNQOHeads / kNKVHeads;
    static constexpr int kPageSize = Config::kPageSize;
    static constexpr int kBlockQ = Config::kBlockQ;
    static constexpr int kBlockKV = Config::kBlockKV;
    static constexpr int kHeadDim = Config::kHeadDim;
    static constexpr int kNWarps = Config::kNWarps;
    static constexpr bool kPagedKV = Config::kPagedKV;
    static constexpr bool kQKVPacked = Config::kQKVPacked;
    static constexpr int kNThreads = Config::kNThreads;
    static constexpr bool kReturnLSE = Config::kReturnLSE;
    static constexpr bool kQKSquare = Config::kQKSquare;
    static constexpr bool kPrefix = Config::kPrefix;

    static_assert(!kPagedKV || kPageSize >= kBlockKV);
    static_assert(!kQKSquare || !kPrefix);

    const int tid = threadIdx.x;

    const int bid_qo_block = blockIdx.x;
    const int bid_batch = blockIdx.y;
    const int bid_qo_head = blockIdx.z;

    extern __shared__ char smem[];

    const int row_offset_qo = params.cu_seqlens_q[bid_batch];
    const int cur_seqlen_qo = params.cu_seqlens_q[bid_batch + 1] - row_offset_qo;
    if (bid_qo_block * kBlockQ >= cur_seqlen_qo) {
        return;
    }

    static constexpr int kRowStrideQ = kQKVPacked ? kHeadDim * (kNQOHeads + kNKVHeads * 2) : kHeadDim * kNQOHeads;
    static constexpr int kRowStrideKV = kQKVPacked ? kHeadDim * (kNQOHeads + kNKVHeads * 2) : kHeadDim * kNKVHeads;

    const int offset_q = row_offset_qo * kRowStrideQ;
    const int offset_o = row_offset_qo * kHeadDim * kNQOHeads;
    const int offset_lse = row_offset_qo * kNQOHeads;

    const int row_offset_kv = params.cu_seqlens_k[bid_batch];
    const int cur_seqlen_kv = params.cu_seqlens_k[bid_batch + 1] - row_offset_kv;
    const int cur_blocklen_qo = cute::min(kBlockQ, cur_seqlen_qo - bid_qo_block * kBlockQ);

    __shared__ int smem_leftpad[kBlockQ];
    for (int i = tid; i < cur_blocklen_qo; i += kNThreads) {
        smem_leftpad[i] = params.leftpad[row_offset_qo + bid_qo_block * kBlockQ + i];
    }

    __shared__ int smem_rightlim[kBlockQ];
    if constexpr (!kQKSquare) {
        for (int i = tid; i < cur_blocklen_qo; i += kNThreads) {
            smem_rightlim[i] = params.rightlim[row_offset_qo + bid_qo_block * kBlockQ + i];
        }
    }
    __syncthreads();

    // Prefix Pad
    const int prefix_pad = kPrefix ? params.prefix_pad[bid_batch] : 0;

    // mQ: (total_qo, num_qo_heads, head_dim)
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.q_ptr) + offset_q),
                            make_shape(cur_seqlen_qo, Int<kNQOHeads>{}, Int<kHeadDim>{}),
                            make_stride(Int<kRowStrideQ>{}, Int<kHeadDim>{}, _1{}));
    Tensor gQ = local_tile(mQ(_, bid_qo_head, _), Shape<Int<kBlockQ>, Int<kHeadDim>>{},
                           make_coord(bid_qo_block, 0));

    const int kv_block_min = smem_leftpad[0] / kBlockKV;
    const int kv_block_max = cute::ceil_div(kQKSquare ? (bid_qo_block + 1) * kBlockQ : smem_rightlim[cur_blocklen_qo - 1], kBlockKV);

    const int* block_table = params.block_table + bid_batch * params.max_num_pages;
    uint32_t offset_kv;
    if constexpr (kPagedKV) {
        const int block_table_idx = (kv_block_max - 1) * kBlockKV / kPageSize;
        const int block_table_offset = (kv_block_max - 1) * kBlockKV - block_table_idx * kPageSize;
        offset_kv = block_table[block_table_idx] * kPageSize * kNKVHeads * kHeadDim +
                    block_table_offset * kNKVHeads * kHeadDim +
                    (bid_qo_head / kNHeadsRatio) * kHeadDim;
    } else {
        offset_kv = row_offset_qo * kRowStrideKV + (kv_block_max - 1) * kBlockKV * kRowStrideKV +
                    (bid_qo_head / kNHeadsRatio) * kHeadDim;
    }

    // mK, mV: (num_pages, page_size, num_kv_heads, head_dim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.k_ptr) + offset_kv),
                            make_shape(Int<kBlockKV>{}, Int<kHeadDim>{}),
                            make_stride(Int<kNKVHeads * kHeadDim>{}, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.v_ptr) + offset_kv),
                            make_shape(Int<kBlockKV>{}, Int<kHeadDim>{}),
                            make_stride(Int<kNKVHeads * kHeadDim>{}, _1{}));

    // (kBlockQ, kHeadDim)
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half*>(smem)), typename Config::SmemLayoutQ{});
    // (kBlockKV, kHeadDim)
    Tensor sK = make_tensor(make_smem_ptr(sQ.data() + size(sQ)), typename Config::SmemLayoutKV{});
    Tensor sV = make_tensor(make_smem_ptr(sK.data() + size(sK)), typename Config::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Config::SmemLayoutVtransposed{});

    typename Config::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    ThrCopy gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tid);

    Tensor tSgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tDsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tSgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tDsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tSgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tDsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Config::TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_thread_slice(tid);
    Tensor tArQ = thr_mma.partition_fragment_A(sQ);
    Tensor tBrK = thr_mma.partition_fragment_B(sK);
    Tensor tBrVt = thr_mma.partition_fragment_B(sVt);

    Tensor tCrO = make_tensor<half>(partition_shape_C(tiled_mma, Shape<Int<kBlockQ>, Int<kHeadDim>>{}));

    TiledCopy smem_tiled_copy_Q = make_tiled_copy_A(typename Config::SmemCopyAtomQK{}, tiled_mma);
    ThrCopy smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    TiledCopy smem_tiled_copy_K = make_tiled_copy_B(typename Config::SmemCopyAtomQK{}, tiled_mma);
    ThrCopy smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    TiledCopy smem_tiled_copy_V = make_tiled_copy_B(typename Config::SmemCopyAtomVtransposed{}, tiled_mma);
    ThrCopy smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid);
    Tensor tSsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    Tensor tScQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tScKV = gmem_thr_copy_QKV.partition_S(cKV);

    // Copy Q
    ::copy<false, false>(gmem_tiled_copy_QKV, tSgQ, tDsQ, tScQ, cur_seqlen_qo - bid_qo_block * kBlockQ);
    // Copy K
    ::copy<false, false>(gmem_tiled_copy_QKV, tSgK, tDsK, tScKV, cur_seqlen_kv - (kv_block_max - 1) * kBlockKV);

    cp_async_fence();
    clear(tCrO);

    Softmax<2 * size<1>(tCrO)> softmax{params.softmax_scale, params.softmax_scale_log2};
    const Mask mask{cur_seqlen_qo, cur_seqlen_kv};

    for (int kv_block_idx = kv_block_max - 1; kv_block_idx >= kv_block_min; --kv_block_idx) {
        Tensor tCrS = make_tensor<half>(partition_shape_C(tiled_mma, Shape<Shape<Int<kBlockQ>>, Int<kBlockKV>>{}));
        clear(tCrS);

        ::cp_async_wait<0>();
        __syncthreads();

        if (kv_block_idx < kv_block_max - 1) {
            // Advance gV
            if constexpr (kPagedKV) {
                const int block_table_idx_cur = (kv_block_idx + 1) * kBlockKV / kPageSize;
                const int block_table_offset_cur = (kv_block_idx + 1) * kBlockKV - block_table_idx_cur * kPageSize;
                const int block_table_idx_next = kv_block_idx * kBlockKV / kPageSize;
                const int block_table_offset_next = kv_block_idx * kBlockKV - block_table_idx_next * kPageSize;
                tSgV.data() = tSgV.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * kPageSize * kNKVHeads * kHeadDim +
                              (block_table_offset_next - block_table_offset_cur) * kNKVHeads * kHeadDim;
            } else {
                tSgV.data() = tSgV.data() + (-int(kBlockKV * kRowStrideKV));
            }
            ::copy<true, false>(gmem_tiled_copy_QKV, tSgV, tDsV, tScKV);
        } else {
            // Need to clear V OOB region due to possible 0 * inf
            ::copy<false, true>(gmem_tiled_copy_QKV, tSgV, tDsV, tScKV, cur_seqlen_kv - kv_block_idx * kBlockKV);
        }
        cp_async_fence();

        gemm_ss(tCrS,
                tArQ, tBrK,
                tSsQ, tSsK,
                tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K);

        mask.template apply_mask<kQKSquare, kPrefix>(tCrS,
                                                     smem_leftpad,
                                                     kQKSquare ? nullptr : smem_rightlim,
                                                     prefix_pad,
                                                     bid_qo_block * kBlockQ,
                                                     kv_block_idx * kBlockKV,
                                                     bid_qo_block * kBlockQ + (tid / 32) * 16 + (tid % 32) / 4,
                                                     kNWarps * 16);

        if (kv_block_idx == kv_block_max - 1) {
            softmax.template rescale_o<true>(tCrS, tCrO);
        } else {
            softmax.template rescale_o<false>(tCrS, tCrO);
        }

        Tensor tArP = make_tensor(tCrS.data(), retile_A(tCrS.layout()));

        ::cp_async_wait<0>();
        __syncthreads();

        if (kv_block_idx > kv_block_min) {
            // Advance gK
            if constexpr (kPagedKV) {
                const int block_table_idx_cur = kv_block_idx * kBlockKV / kPageSize;
                const int block_table_offset_cur = kv_block_idx * kBlockKV - block_table_idx_cur * kPageSize;
                const int block_table_idx_next = (kv_block_idx - 1) * kBlockKV / kPageSize;
                const int block_table_offset_next = (kv_block_idx - 1) * kBlockKV - block_table_idx_next * kPageSize;
                tSgK.data() = tSgK.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * kPageSize * kNKVHeads * kHeadDim +
                              (block_table_offset_next - block_table_offset_cur) * kNKVHeads * kHeadDim;
            } else {
                tSgK.data() = tSgK.data() + (-int(kBlockKV * kRowStrideKV));
            }
            ::copy<true, false>(gmem_tiled_copy_QKV, tSgK, tDsK, tScKV);
            cp_async_fence();
        }

        gemm_rs(tCrO,
                tArP, tBrVt,
                tSsVt,
                tiled_mma,
                smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Normalize out
    Tensor lse = softmax.normalize(tCrO);

    // (kBlockQ * num_heads_per_kv, head_dim)
    Tensor sO = make_tensor(sQ.data(), typename Config::SmemLayoutO{});
    TiledCopy smem_tiled_copy_O = make_tiled_copy_C(typename Config::SmemCopyAtomO{}, tiled_mma);
    ThrCopy smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tid);
    Tensor tSrO = smem_thr_copy_O.retile_S(tCrO);
    Tensor tDsO = smem_thr_copy_O.partition_D(sO);
    copy(smem_tiled_copy_O, tSrO, tDsO);

    // mO: (total_qo, num_qo_heads, head_dim)
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.out_ptr) + offset_o),
                            make_shape(cur_seqlen_qo, Int<kNQOHeads>{}, Int<kHeadDim>{}),
                            GenRowMajor{});
    Tensor gO = local_tile(mO(_, bid_qo_head, _), Shape<Int<kBlockQ>, Int<kHeadDim>>{},
                           make_coord(bid_qo_block, 0));

    // mLSE: (total_qo, num_qo_heads)
    Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<half*>(params.lse_ptr) + offset_lse),
                              make_shape(cur_seqlen_qo, Int<kNQOHeads>{}),
                              GenRowMajor{});
    Tensor gLSE = local_tile(mLSE(_, bid_qo_head), Shape<Int<kBlockQ>>{},
                             make_coord(bid_qo_block));

    typename Config::GmemTiledCopyO gmem_tiled_copy_O;
    ThrCopy gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tid);
    Tensor tSsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tDgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor rO = make_tensor<half>(shape(tDgO));
    copy(gmem_tiled_copy_O, tSsO, rO);

    Tensor cO = make_identity_tensor(Shape<Int<kBlockQ>, Int<kHeadDim>>{});

    if constexpr (kReturnLSE) {
        // Write back LSE
        Tensor tLSEcO = thr_mma.partition_C(cO);
        static_assert(decltype(size<0>(tLSEcO))::value == 4);

        // ((2, 2), MMA_M, MMA_N) -> (2, MMA_M)
        Tensor tLSEcO_row_view = logical_divide(tLSEcO, Shape<_2>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(tLSEcO_row_view));
        if (get<1>(tLSEcO_row_view(0)) == 0) {
#pragma unroll
            for (int m = 0; m < size(lse); ++m) {
                const int row = get<0>(tLSEcO_row_view(m));
                if (row < cur_seqlen_qo - bid_qo_block * kBlockQ) {
                    gLSE(row) = lse(m);
                }
            }
        }
    }

    // Write back O
    Tensor tDcO = gmem_thr_copy_O.partition_D(cO);
    ::copy<false, false>(gmem_tiled_copy_O, rO, tDgO, tDcO, cur_seqlen_qo - bid_qo_block * kBlockQ);
}
