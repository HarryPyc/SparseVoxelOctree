#pragma once
#define TRIANGLE_PER_THREAD
#include "cuda_runtime.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "Mesh.h"

class Voxel {
public:
	__host__ __device__ Voxel();
	__host__ __device__ ~Voxel();

	__host__ __device__ void GetInfo(glm::vec3& color, glm::vec3& normal);
	__host__ __device__ void SetInfo(glm::vec3 color,  glm::vec3 normal);
	__host__ __device__ inline bool empty() { return c == 0; }

	unsigned int c, n;

};

__device__ __host__ glm::vec4 ConvUintToVec4(unsigned int val);
__device__ __host__ unsigned int ConvVec4ToUint(glm::vec4 val);

const unsigned short voxelDim = 256;

void Voxelization(CudaMesh &cuMesh, Voxel*& d_voxel);
void RunRayMarchingKernel(unsigned int* d_pbo, cudaArray_t front, cudaArray_t back, Voxel* d_voxel, CudaMesh cuMesh, const unsigned w, const unsigned h);
void initCudaTexture();