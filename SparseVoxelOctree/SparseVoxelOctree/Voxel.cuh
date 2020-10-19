#pragma once
#define TRIANGLE_PER_THREAD
#include "cuda_runtime.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "Mesh.h"

class Voxel {
public:
	__device__ Voxel();
	__device__ ~Voxel();

	__device__ void GetInfo(glm::vec3& color, glm::vec3& normal);
	__device__ void SetInfo(glm::vec3 color,  glm::vec3 normal);
	__device__ inline unsigned int GetCounter();
	__device__ inline void SetCounter(unsigned int counter);
	__device__ inline bool empty() { return c == 0; }
	__device__ glm::vec3 PhongLighting(glm::vec3 pos);

	unsigned int c, n;

};

struct VoxelizationInfo {
	glm::vec3 minAABB; float delta;
	glm::vec3 camPos; float ks;
	glm::vec3 lightPos; float kd;
	float ka, alpha; unsigned int Dim, PADDING;
};
__device__ __host__ glm::vec4 ConvUintToVec4(unsigned int val);
__device__ __host__ unsigned int ConvVec4ToUint(glm::vec4 val);

const unsigned short voxelDim = 512;

void Voxelization(CudaMesh &cuMesh, Voxel*& d_voxel);
void RunRayMarchingKernel(unsigned int* d_pbo, cudaArray_t front, cudaArray_t back, Voxel* d_voxel, CudaMesh cuMesh, const unsigned w, const unsigned h);
void initCudaVoxelization();
void uploadConstant(VoxelizationInfo Info);


