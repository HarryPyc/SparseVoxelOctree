#pragma once
#include "cuda_runtime.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "Mesh.h"
#define PRINT_INFO
typedef unsigned int uint;

class Voxel {
public:
	__host__ __device__ Voxel();
	__host__ __device__ ~Voxel();

	__device__ void GetInfo(glm::vec3& color, glm::vec3& normal);
	__device__ void SetInfo(glm::vec3 color,  glm::vec3 normal);

	__device__ inline bool empty() { return c == 0; }
	__device__ glm::vec3 PhongLighting(glm::vec3 pos);

	unsigned int c, n;

};

struct VoxelizationInfo {
	glm::vec3 minAABB; float delta;
	glm::vec3 camPos; float ks;
	glm::vec3 lightPos; float kd;
	float ka, alpha; uint Dim, Counter;
};

const int WINDOW_WIDTH = 1280, WINDOW_HEIGHT = 720;
const unsigned short voxelDim = 512;

void PreProcess(CudaMesh& cuMesh);
void InitVoxelization(Voxel*& d_voxel);
void Voxelization(CudaMesh &cuMesh, Voxel*& d_voxel);
//void RunRayMarchingKernel(uint* d_pbo, cudaArray_t front, cudaArray_t back, Voxel* d_voxel);

__device__ inline glm::vec4 ConvUintToVec4(unsigned int val)
{
	glm::vec4 res(float((val & 0x000000FF)), float((val & 0x0000FF00) >> 8U), float((val & 0x00FF0000) >> 16U), float((val & 0xFF000000) >> 24U));
	return res / 255.f;
}
__device__ inline unsigned int ConvVec4ToUint(glm::vec4 val) {
	val *= 255.f;
	return ((unsigned int(val.w) & 0x000000FF) << 24U) | ((unsigned int(val.z) & 0x000000FF) << 16U)
		| ((unsigned int(val.y) & 0x000000FF) << 8U) | (unsigned int(val.x) & 0x000000FF);
}

__device__ inline glm::uvec3 ConvUintToUvec3(unsigned int val) {
	return glm::uvec3(val >> 22U, val << 10U >> 22U, val << 20U >> 22U);
}
__device__ inline unsigned int ConvUvec3ToUint(glm::uvec3 val) {
	return val.x << 22U | val.y << 12U | val.z << 2U;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}