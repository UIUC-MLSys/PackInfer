#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>

#include "dispatch.hpp"
#include "params.hpp"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_STRIDE(x, ...) TORCH_CHECK(x.strides() == torch::IntArrayRef({__VA_ARGS__}), #x " must have stride (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template <int HeadDim, int NQOHeads, int NKVHeads, int PageSize, bool PagedKV, bool QKVPacked, bool ReturnLSE, bool QKSquare, bool Prefix>
void launch_packed_attn(const Params& params, cudaStream_t stream);

void packed_attn_interface(
    const at::Tensor& q,                            // (total_qo, num_heads_qo, head_dim)
    const at::Tensor& k,                            // (total_kv, num_heads_kv, head_dim) or (num_pages, page_size, num_kv_heads, head_dim)
    const at::Tensor& v,                            // (total_kv, num_heads_kv, head_dim) or (num_pages, page_size, num_kv_heads, head_dim)
    at::Tensor& out,                                // (total_qo, num_heads_qo, head_dim)
    std::optional<at::Tensor>& lse_,                // (total_qo, num_heads_qo)
    const std::optional<at::Tensor>& prefix_pad_,   // (batch_size,)
    const at::Tensor& cu_seqlens_q,                 // (batch_size + 1,)
    const at::Tensor& cu_seqlens_k,                 // (batch_size + 1,)
    const at::Tensor& leftpad,                      // (total_qo,)
    const std::optional<at::Tensor>& rightlim_,     // (total_qo,)
    const std::optional<at::Tensor>& block_table_,  // (batch_size, max_num_pages)
    const int max_seqlen_q,                         // (1,)
    const float softmax_scale                       // (1,)
) {
    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16, "only support fp16 data type");

    const int total_qo = q.size(0);
    const int num_qo_heads = q.size(1);
    const int head_dim = q.size(2);
    const int page_size = k.size(-3);
    const int num_kv_heads = k.size(-2);
    const int batch_size = cu_seqlens_q.size(0) - 1;
    const int max_num_pages = block_table_.has_value() ? block_table_->size(1) : 0;

    CHECK_DEVICE(q);
    CHECK_SHAPE(q, total_qo, num_qo_heads, head_dim);

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(out);
    CHECK_CONTIGUOUS(out);
    CHECK_SHAPE(out, total_qo, num_qo_heads, head_dim);
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");

    if (lse_.has_value()) {
        at::Tensor& lse = lse_.value();
        CHECK_DEVICE(lse);
        CHECK_CONTIGUOUS(lse);
        CHECK_SHAPE(lse, total_qo, num_qo_heads);
        TORCH_CHECK(lse.dtype() == q_dtype, "query and lse must have the same dtype");
    }

    if (block_table_.has_value()) {
        const int num_pages = k.size(0);
        CHECK_DEVICE(k);
        CHECK_CONTIGUOUS(k);
        CHECK_SHAPE(k, num_pages, page_size, num_kv_heads, head_dim);

        CHECK_DEVICE(v);
        CHECK_CONTIGUOUS(v);
        CHECK_SHAPE(v, num_pages, page_size, num_kv_heads, head_dim);

        CHECK_CONTIGUOUS(q);

        const auto& block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        CHECK_SHAPE(block_table, batch_size, max_num_pages);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    } else {
        const int total_kv = k.size(0);
        CHECK_DEVICE(k);
        CHECK_SHAPE(k, total_kv, num_kv_heads, head_dim);

        CHECK_DEVICE(v);
        CHECK_SHAPE(v, total_kv, num_kv_heads, head_dim);

        if (q.is_contiguous()) {
            CHECK_CONTIGUOUS(k);
            CHECK_CONTIGUOUS(v);
        } else {
            CHECK_STRIDE(q, (num_kv_heads * 2 + num_qo_heads) * head_dim, head_dim, 1);
            CHECK_STRIDE(k, (num_kv_heads * 2 + num_qo_heads) * head_dim, head_dim, 1);
            CHECK_STRIDE(v, (num_kv_heads * 2 + num_qo_heads) * head_dim, head_dim, 1);
        }
    }

    CHECK_DEVICE(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");

    CHECK_DEVICE(cu_seqlens_k);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlen_k must have dtype int32");

    CHECK_DEVICE(leftpad);
    CHECK_CONTIGUOUS(leftpad);
    CHECK_SHAPE(leftpad, total_qo);
    TORCH_CHECK(leftpad.dtype() == torch::kInt32, "leftpad must have dtype int32");

    if (rightlim_.has_value()) {
        const auto& rightlim = rightlim_.value();
        CHECK_DEVICE(rightlim);
        CHECK_CONTIGUOUS(rightlim);
        CHECK_SHAPE(rightlim, total_qo);
        TORCH_CHECK(rightlim.dtype() == torch::kInt32, "rightlim must have dtype int32");
    }

    if (prefix_pad_.has_value()) {
        const auto& prefix_pad = prefix_pad_.value();
        CHECK_DEVICE(prefix_pad);
        CHECK_CONTIGUOUS(prefix_pad);
        CHECK_SHAPE(prefix_pad, batch_size);
        TORCH_CHECK(prefix_pad.dtype() == torch::kInt32, "rightlim must have dtype int32");
    }

    TORCH_CHECK(!prefix_pad_.has_value() || rightlim_.has_value(), "qksqure and prefix cannot be used together");

    Params params{
        .q_ptr = q.data_ptr(),
        .k_ptr = k.data_ptr(),
        .v_ptr = v.data_ptr(),
        .out_ptr = out.data_ptr(),
        .lse_ptr = lse_.has_value() ? lse_->data_ptr() : nullptr,

        .cu_seqlens_q = reinterpret_cast<int*>(cu_seqlens_q.data_ptr()),
        .cu_seqlens_k = reinterpret_cast<int*>(cu_seqlens_k.data_ptr()),
        .block_table = reinterpret_cast<int*>(block_table_.has_value() ? block_table_->data_ptr() : nullptr),
        .leftpad = reinterpret_cast<int*>(leftpad.data_ptr()),
        .rightlim = reinterpret_cast<int*>(rightlim_.has_value() ? rightlim_->data_ptr() : nullptr),
        .prefix_pad = reinterpret_cast<int*>(prefix_pad_.has_value() ? prefix_pad_->data_ptr() : nullptr),

        .softmax_scale = static_cast<half>(softmax_scale),
        .softmax_scale_log2 = static_cast<half>(softmax_scale * M_LOG2E),

        .max_seqlen_q = max_seqlen_q,
        .batch_size = batch_size,
        .max_num_pages = max_num_pages,
    };

    value_switch<int, 128>(
        head_dim,
        [&](const auto expr) {
            static constexpr int kHeadDim = decltype(expr)::value;
            value_switch<std::pair<int, int>, {32, 32}, {32, 8}, {32, 4}, {16, 8}, {16, 4}, {8, 2}>(
                {num_qo_heads, num_kv_heads},
                [&](const auto expr) {
                    static constexpr int kNQOHeads = decltype(expr)::value.first;
                    static constexpr int kNKVHeads = decltype(expr)::value.second;
                    bool_switch(lse_.has_value(), [&](const auto expr) {
                        static constexpr int kReturnLSE = decltype(expr)::value;
                        value_switch<std::pair<bool, bool>, {true, false}, {false, true}, {false, false}>(
                            {!rightlim_.has_value(), prefix_pad_.has_value()},
                            [&](const auto expr) {
                                static constexpr int kQKSquare = decltype(expr)::value.first;
                                static constexpr int kPrefix = decltype(expr)::value.second;
                                bool_switch(!q.is_contiguous(), [&](const auto expr) {
                                    static constexpr int kQKVPacked = decltype(expr)::value;
                                    if (block_table_.has_value()) {
                                        value_switch<int, 128, 256>(
                                            page_size,
                                            [&](const auto expr) {
                                                static constexpr int kPageSize = decltype(expr)::value;
                                                launch_packed_attn<kHeadDim, kNQOHeads, kNKVHeads, kPageSize, true, kQKVPacked, kReturnLSE, kQKSquare, kPrefix>(params, stream);
                                            },
                                            [] {
                                                TORCH_CHECK(false, "unsupported page_size");
                                            });
                                    } else {
                                        launch_packed_attn<kHeadDim, kNQOHeads, kNKVHeads, -1, false, kQKVPacked, kReturnLSE, kQKSquare, kPrefix>(params, stream);
                                    }
                                });
                            },
                            [] {
                                TORCH_CHECK(false, "unsupported (qksquare, prefix)");
                            });
                    });
                },
                [] {
                    TORCH_CHECK(false, "unsupported (num_qo_heads, num_kv_heads)");
                });
        },
        [] {
            TORCH_CHECK(false, "unsupported head_dim");
        });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packed_attn", &::packed_attn_interface, "Packed Forward Pass");
}
