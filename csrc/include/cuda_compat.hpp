#pragma once

#include <cuda_fp16.h>
#include <cmath>

// CUDA 12 compatibility: CUDART_*_FP16 were removed
#ifndef CUDART_ZERO_FP16
#define CUDART_ZERO_FP16 __float2half(0.0f)
#endif

#ifndef CUDART_ONE_FP16
#define CUDART_ONE_FP16 __float2half(1.0f)
#endif

#ifndef CUDART_INF_FP16
#define CUDART_INF_FP16 __float2half(INFINITY)
#endif
