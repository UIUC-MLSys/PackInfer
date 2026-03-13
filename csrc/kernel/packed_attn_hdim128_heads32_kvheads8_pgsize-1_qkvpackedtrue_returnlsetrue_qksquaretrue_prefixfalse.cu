#include "launch.hpp"

template void launch_packed_attn<128, 32, 8, -1, false, true, true, true, false>(const Params& params, cudaStream_t stream);
