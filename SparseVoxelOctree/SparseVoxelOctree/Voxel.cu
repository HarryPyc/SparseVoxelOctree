#include "Voxel.cuh"
#include "Mesh.h"
#include "device_launch_parameters.h"


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

__device__ inline size_t ToArrayIdx(glm::uvec3 coord) {
	return coord.z + coord.y * voxelDim + coord.x * voxelDim * voxelDim;
}

__device__ inline glm::uvec3 GetIndex(glm::vec3 pos, glm::vec3 center, float vLength) {
	glm::vec3 minAABB = center - vLength;
	return glm::uvec3((pos - center) / vLength * float(voxelDim));
}

__device__ inline bool VoxelTriangleIntersection() {

}
__global__ void VoxelizationKernel(Voxel* voxelList, CudaMesh mesh, const unsigned short voxelDim) {
	const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= mesh.triNum) return;


}
__global__ void PreProcessTriangleKernel(CudaMesh mesh, const unsigned short voxelDim) {
	const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= mesh.triNum) return;
	Triangle tri;
	tri.i0 = mesh.d_idx[3 * idx], tri.i1 = mesh.d_idx[3 * idx + 1], tri.i2 = mesh.d_idx[3 * idx + 2];
	glm::vec3 v[3];
	v[0] = glm::vec3(mesh.d_v[3 * tri.i0], mesh.d_v[3 * tri.i0 + 1], mesh.d_v[3 * tri.i0 + 2]);
	v[1] = glm::vec3(mesh.d_v[3 * tri.i1], mesh.d_v[3 * tri.i1 + 1], mesh.d_v[3 * tri.i1 + 2]);
	v[2] = glm::vec3(mesh.d_v[3 * tri.i2], mesh.d_v[3 * tri.i2 + 1], mesh.d_v[3 * tri.i2 + 2]);
	const glm::vec3 e[3] = { v[1] - v[0], v[2] - v[1], v[0] - v[2] };
	const float delta = mesh.delta / (float)voxelDim;
	//Pre-compute parameters for voxel triangle intersection
	tri.n = glm::cross(e[0], e[1]);
	glm::vec3 c(tri.n.x > 0 ? delta : 0, tri.n.y > 0 ? delta : 0, tri.n.z > 0 ? delta : 0);
	tri.d1 = glm::dot(tri.n, c - v[0]), tri.d2 = glm::dot(tri.n, delta - c - v[0]);

	for (int i = 0; i < 3; i++) {
		tri.ne_xy[i] = glm::vec2(-e[i].y, e[i].x) * (tri.n.z >= 0.f ? 1.f : -1.f);
		tri.ne_xz[i] = glm::vec2(-e[i].z, e[i].x) * (tri.n.y >= 0.f ? 1.f : -1.f);
		tri.ne_yz[i] = glm::vec2(-e[i].z, e[i].y) * (tri.n.x >= 0.f ? 1.f : -1.f);

		tri.de_xy[i] = -glm::dot(tri.ne_xy[i], glm::vec2(v[i].x, v[i].y)) + glm::max(0.f, delta * tri.ne_xy[i].x)
			+ glm::max(0.f, delta * tri.ne_xy[i].y);
		tri.de_xz[i] = -glm::dot(tri.ne_xz[i], glm::vec2(v[i].x, v[i].z)) + glm::max(0.f, delta * tri.ne_xz[i].x)
			+ glm::max(0.f, delta * tri.ne_xz[i].y);
		tri.de_yz[i] = -glm::dot(tri.ne_yz[i], glm::vec2(v[i].y, v[i].z)) + glm::max(0.f, delta * tri.ne_yz[i].x)
			+ glm::max(0.f, delta * tri.ne_yz[i].y);
	}
	//Write to global memory
	mesh.d_tri[idx] = tri;
	printf("normal from thread %i : (%f, %f, %f)\n", idx, tri.n.x, tri.n.y, tri.n.z);
}

void Voxelization(Mesh& mesh, Voxel* d_voxel)
{
	CudaMesh cuMesh;
	mesh.UploatToDevice(cuMesh);
	cudaError_t cudaStatus;
	//PreProcess Triangle
	cudaStatus = cudaMalloc((void**)&cuMesh.d_tri, cuMesh.triNum * sizeof(Triangle));
	if (cudaStatus != cudaSuccess) printf("d_tri cudaMalloc Failed\n");
	const unsigned int blockDim = 256, gridDim = cuMesh.triNum / blockDim + 1;
	PreProcessTriangleKernel <<< gridDim, blockDim >>> (cuMesh, voxelDim);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) printf("PreprocessTriangle Launch Kernel Failed\n");
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) printf("cudaDeviceSynchronize Failed\n");
	cudaStatus = cudaFree(cuMesh.d_idx);
	if (cudaStatus != cudaSuccess) printf("d_idx cudaFree Failed, error: %s\n", cudaGetErrorString(cudaStatus));



	size_t voxelSize = voxelDim * voxelDim * voxelDim * sizeof(Voxel);

	cudaStatus = cudaMalloc((void**)&d_voxel, voxelSize);
	if (cudaStatus != cudaSuccess) printf("d_voxel cudaMalloc Failed\n");

#ifdef TRIANGLE_PER_THREAD
	
	VoxelizationKernel << <gridDim, blockDim >> > (d_voxel, cuMesh, voxelDim);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) printf("cuda Launch Kernel Failed\n");
#endif // TRIANGLE_PER_THREAD

}