#include "launch.hpp"

template void launch_packed_attn<128, 16, 4, -1, false, true, false, true, false>(const Params& params, cudaStream_t stream);
