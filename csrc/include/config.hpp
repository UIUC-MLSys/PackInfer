#pragma once

#include <cute/tensor.hpp>

using namespace cute;

template <int HeadDim, int NQOHeads, int NKVHeads, int PageSize, int BlockQ, int BlockKV, int NWarps, bool PagedKV, bool QKVPacked, bool ReturnLSE, bool QKSquare, bool Prefix>
struct Config {
    static constexpr int kHeadDim = HeadDim;
    static constexpr int kNQOHeads = NQOHeads;
    static constexpr int kNKVHeads = NKVHeads;
    static constexpr int kPageSize = PageSize;
    static constexpr int kBlockQ = BlockQ;
    static constexpr int kBlockKV = BlockKV;
    static constexpr int kNWarps = NWarps;
    static constexpr int kNThreads = NWarps * 32;
    static constexpr bool kPagedKV = PagedKV;
    static constexpr bool kQKVPacked = QKVPacked;
    static constexpr bool kReturnLSE = ReturnLSE;
    static constexpr bool kQKSquare = QKSquare;
    static constexpr bool kPrefix = Prefix;

    using TiledMMA = TiledMMA<MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>, Layout<Shape<Int<NWarps>, _1, _1>>, Tile<Int<16 * NWarps>, _16, _16>>;

    static constexpr int kSwizzleB = 3;
    static constexpr int kSwizzleM = 3;
    static constexpr int kSwizzleS = 3;

    static constexpr int kSmemRowStride = 1 << (kSwizzleM + kSwizzleS);
    static constexpr int kSmemNRow = 1 << kSwizzleB;
    static_assert(HeadDim % kSmemRowStride == 0);

    using SmemLayoutAtomQKVO = decltype(composition(Swizzle<kSwizzleB, kSwizzleM, kSwizzleS>{}, Layout<Shape<Int<kSmemNRow>, Int<kSmemRowStride>>, Stride<Int<kSmemRowStride>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQKVO{}, Shape<Int<BlockQ>, Int<HeadDim>>{}));
    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomQKVO{}, Shape<Int<BlockKV>, Int<HeadDim>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockKV>>{}, GenRowMajor{})));
    using SmemLayoutO = SmemLayoutQ;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(half);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(half);
    static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

    using SmemCopyAtomQK = Copy_Atom<SM75_U32x4_LDSM_N, half>;
    using SmemCopyAtomVtransposed = Copy_Atom<SM75_U16x8_LDSM_T, half>;
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half>;

    static constexpr int kGmemCopyAtomSize = sizeof(uint128_t) / sizeof(half);
    static constexpr int kGmemThreadsPerRow = kSmemRowStride / kGmemCopyAtomSize;
    using GmemTiledCopyTLayoutQKVO = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half>{}, GmemTiledCopyTLayoutQKVO{}, Layout<Shape<_1, Int<kGmemCopyAtomSize>>>{}));
    using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half>{}, GmemTiledCopyTLayoutQKVO{}, Layout<Shape<_1, Int<kGmemCopyAtomSize>>>{}));
};
