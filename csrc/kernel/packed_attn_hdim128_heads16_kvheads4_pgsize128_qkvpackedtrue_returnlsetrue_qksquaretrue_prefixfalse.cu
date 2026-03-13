#include "launch.hpp"

template void launch_packed_attn<128, 16, 4, 128, true, true, true, true, false>(const Params& params, cudaStream_t stream);
