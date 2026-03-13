#include "launch.hpp"

template void launch_packed_attn<128, 8, 2, -1, false, false, false, false, true>(const Params& params, cudaStream_t stream);
