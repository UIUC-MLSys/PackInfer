#include "launch.hpp"

template void launch_packed_attn<128, 16, 8, -1, false, false, false, true, false>(const Params& params, cudaStream_t stream);
