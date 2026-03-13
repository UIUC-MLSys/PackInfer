#include "launch.hpp"

template void launch_packed_attn<128, 32, 8, -1, false, false, false, false, false>(const Params& params, cudaStream_t stream);
