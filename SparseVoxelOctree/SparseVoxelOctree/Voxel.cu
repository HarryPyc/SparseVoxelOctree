#include "Voxel.cuh"
#include "Mesh.h"
typedef unsigned int uint;
__device__ __host__ glm::vec4 ConvUintToVec4(unsigned int val)
{
	glm::vec4 res(float((val & 0x000000FF)), float((val & 0x0000FF00) >> 8U), float((val & 0x00FF0000) >> 16U), float((val & 0xFF000000) >> 24U));
	return res / 255.f;
}
__device__ __host__ unsigned int ConvVec4ToUint(glm::vec4 val) {
	val *= 255.f;
	return (uint(val.w) & 0x000000FF) << 24U | (uint(val.z) & 0x000000FF) << 16U | (uint(val.y) & 0x000000FF) << 8U | (uint(val.x) & 0x000000FF);
}

Voxel::Voxel()
{
	c = 0, n = 0;
}

Voxel::~Voxel() {

}

__host__ __device__ void Voxel::GetInfo(glm::vec3& color, glm::vec3& normal) {
	color = glm::vec3(float((c & 0x000000FF)), float((c & 0x0000FF00) >> 8U), float((c & 0x00FF0000) >> 16U));
	normal = glm::vec3(float((n & 0x000000FF)), float((n & 0x0000FF00) >> 8U), float((n & 0x00FF0000) >> 16U));
	color /= 255.f;
	normal = (normal / 255.f) * 2.f - 1.f;
}

__host__ __device__ void Voxel::SetInfo(glm::vec3 color, glm::vec3 normal) {
	color *= 255.f;
	c = ((uint(color.z) & 0x000000FF) << 16U | (uint(color.y) & 0x000000FF) << 8U | (uint(color.x) & 0x000000FF));
	normal = (normal + 1.f) / 2.f * 255.f;
	n = ((uint(normal.z) & 0x000000FF) << 16U | (uint(normal.y) & 0x000000FF) << 8U | (uint(normal.x) & 0x000000FF));
}

__global__ void VoxelizationKernel(CudaMesh mesh, const unsigned short voxelDim) {

}

void Voxelization(CudaMesh mesh)
{

}