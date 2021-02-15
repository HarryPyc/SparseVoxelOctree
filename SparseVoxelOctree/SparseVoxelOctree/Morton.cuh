#pragma once
#include "cuda_runtime.h"
#include <cuda.h>
__host__ __device__ cuuint32_t EncodeMorton(cuuint32_t x, cuuint32_t y, cuuint32_t z);
__host__ __device__ cuuint32_t DecodeMortonX(cuuint32_t code);
__host__ __device__ cuuint32_t DecodeMortonY(cuuint32_t code);
__host__ __device__ cuuint32_t DecodeMortonZ(cuuint32_t code);