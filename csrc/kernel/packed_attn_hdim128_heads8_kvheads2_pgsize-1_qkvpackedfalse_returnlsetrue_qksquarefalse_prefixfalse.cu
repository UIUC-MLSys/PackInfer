#include "launch.hpp"

template void launch_packed_attn<128, 8, 2, -1, false, false, true, false, false>(const Params& params, cudaStream_t stream);
