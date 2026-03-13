#include "launch.hpp"

template void launch_packed_attn<128, 8, 2, 256, true, true, true, true, false>(const Params& params, cudaStream_t stream);
