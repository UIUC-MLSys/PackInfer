#include "launch.hpp"

template void launch_packed_attn<128, 8, 2, 256, true, true, false, false, true>(const Params& params, cudaStream_t stream);
