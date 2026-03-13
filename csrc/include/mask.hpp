#pragma once

#include <cute/tensor.hpp>

#include "softmax.hpp"
#include "cuda_compat.hpp"

using namespace cute;

struct Mask {
    const int seqlen_q, seqlen_k;

    Mask() = delete;

    __forceinline__ __device__
    Mask(const int seqlen_q, const int seqlen_k)
        : seqlen_q(seqlen_q), seqlen_k(seqlen_k) {}

    template <
        bool QKSquare,
        bool Prefix,
        typename Engine,
        typename Layout>
    __forceinline__ __device__ void
    apply_mask(
        Tensor<Engine, Layout>& rS,
        const int* smem_leftpad,
        const int* smem_rightlim,
        const int prefix_pad,
        const int pad_offset,
        int col_idx_offset,
        const int row_idx_offset,
        const int warp_row_stride) const {
        static_assert(Layout::rank == 3);
        static_assert(decltype(size<0>(rS))::value == 4);

        Tensor tensor = make_tensor(rS.data(), Softmax<0>::retile_mma(rS.layout()));
        const int lane_id = threadIdx.x % 32;
        col_idx_offset += (lane_id % 4) * 2;
#pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
#pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                const int col_idx_limit_left = smem_leftpad[row_idx - pad_offset];
                const int col_idx_limit_right = QKSquare ? prefix_pad + row_idx + 1 : smem_rightlim[row_idx - pad_offset];
#pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
#pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        if (col_idx >= col_idx_limit_right || ((Prefix ? col_idx >= prefix_pad : true) && col_idx < col_idx_limit_left)) {
                            tensor(make_coord(i, mi), make_coord(j, nj)) = -CUDART_INF_FP16;
                        }
                    }
                }
            }
        }
    }
};
