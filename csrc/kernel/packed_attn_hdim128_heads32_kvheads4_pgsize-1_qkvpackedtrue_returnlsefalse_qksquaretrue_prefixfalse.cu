#include "launch.hpp"

template void launch_packed_attn<128, 32, 4, -1, false, true, false, true, false>(const Params& params, cudaStream_t stream);
