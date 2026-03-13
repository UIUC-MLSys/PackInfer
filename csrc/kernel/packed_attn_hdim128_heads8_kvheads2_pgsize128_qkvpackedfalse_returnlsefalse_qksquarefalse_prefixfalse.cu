#include "launch.hpp"

template void launch_packed_attn<128, 8, 2, 128, true, false, false, false, false>(const Params& params, cudaStream_t stream);
