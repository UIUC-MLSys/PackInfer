#include "launch.hpp"

template void launch_packed_attn<128, 32, 8, 128, true, true, true, true, false>(const Params& params, cudaStream_t stream);
