#include "launch.hpp"

template void launch_packed_attn<128, 8, 2, 128, true, true, true, false, false>(const Params& params, cudaStream_t stream);
