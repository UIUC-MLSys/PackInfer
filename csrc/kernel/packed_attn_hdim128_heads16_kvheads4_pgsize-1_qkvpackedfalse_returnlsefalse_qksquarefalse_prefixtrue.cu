#include "launch.hpp"

template void launch_packed_attn<128, 16, 4, -1, false, false, false, false, true>(const Params& params, cudaStream_t stream);
