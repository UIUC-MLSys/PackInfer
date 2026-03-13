#include "launch.hpp"

template void launch_packed_attn<128, 16, 4, -1, false, false, true, true, false>(const Params& params, cudaStream_t stream);
