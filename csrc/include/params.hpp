#pragma once

#include <cuda_fp16.h>

struct Params {
    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;
    void* __restrict__ out_ptr;
    void* __restrict__ lse_ptr;

    int* __restrict__ cu_seqlens_q;
    int* __restrict__ cu_seqlens_k;
    int* __restrict__ block_table;
    int* __restrict__ leftpad;
    int* __restrict__ rightlim;
    int* __restrict__ prefix_pad;

    half softmax_scale;
    half softmax_scale_log2;

    int max_seqlen_q;
    int batch_size;
    int max_num_pages;
};
