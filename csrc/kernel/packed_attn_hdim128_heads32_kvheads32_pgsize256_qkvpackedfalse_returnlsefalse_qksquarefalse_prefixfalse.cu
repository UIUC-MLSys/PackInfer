#include "launch.hpp"

template void launch_packed_attn<128, 32, 32, 256, true, false, false, false, false>(const Params& params, cudaStream_t stream);
