#include "launch.hpp"

template void launch_packed_attn<128, 32, 8, 256, true, true, true, false, false>(const Params& params, cudaStream_t stream);
