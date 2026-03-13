#pragma once

#if !defined(__CUDACC__)
#error "This header is CUDA-only. Include it only from a CUDA TU."
#endif

#include <c10/cuda/CUDAException.h>

#include <cute/tensor.hpp>

#include "config.hpp"
#include "kernel.hpp"
#include "params.hpp"

using namespace cute;

template <int HeadDim, int NQOHeads, int NKVHeads, int PageSize, bool PagedKV, bool QKVPacked, bool ReturnLSE, bool QKSquare, bool Prefix>
void launch_packed_attn(const Params& params, cudaStream_t stream) {
    static constexpr int kBlockQ = 128;
    static constexpr int kBlockKV = HeadDim <= 64 ? 256 : (HeadDim <= 128 ? 128 : 64);
    static constexpr int kNWarps = 8;

    using Config = Config<HeadDim, NQOHeads, NKVHeads, PageSize, kBlockQ, kBlockKV, kNWarps, PagedKV, QKVPacked, ReturnLSE, QKSquare, Prefix>;

    auto kernel = &packed_attn_kernel<Config>;

    static constexpr int kSmemSize = Config::kSmemSize;
    if constexpr (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    }

    const int num_q_block = cute::ceil_div(params.max_seqlen_q, kBlockQ);
    dim3 grid(num_q_block, params.batch_size, NQOHeads);
    kernel<<<grid, Config::kNThreads, kSmemSize, stream>>>(params);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
