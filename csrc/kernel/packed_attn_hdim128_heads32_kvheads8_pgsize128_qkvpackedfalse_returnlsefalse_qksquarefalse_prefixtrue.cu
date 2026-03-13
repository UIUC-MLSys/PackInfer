#include "launch.hpp"

template void launch_packed_attn<128, 32, 8, 128, true, false, false, false, true>(const Params& params, cudaStream_t stream);
