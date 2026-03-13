#include "launch.hpp"

template void launch_packed_attn<128, 32, 32, -1, false, true, false, false, false>(const Params& params, cudaStream_t stream);
