#include "launch.hpp"

template void launch_packed_attn<128, 32, 4, 256, true, false, false, false, true>(const Params& params, cudaStream_t stream);
