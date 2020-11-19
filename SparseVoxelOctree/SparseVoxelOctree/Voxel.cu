#include "Voxel.cuh"
#include "Mesh.h"
#include "device_launch_parameters.h"
#include <time.h>
extern VoxelizationInfo Info;
extern __constant__ VoxelizationInfo d_Info;
//texture<float4, 2, cudaReadModeElementType> frontTex, backTex;
__device__ uint voxelCounter;

__host__ __device__ Voxel::Voxel()
{
	c = 0, n = 0;
}

__host__ __device__ Voxel::~Voxel() {

}

 __device__ void Voxel::GetInfo(glm::vec4& color, glm::vec3& normal) {
	color = glm::vec4(float((c & 0x000000FF)), float((c & 0x0000FF00) >> 8U), float((c & 0x00FF0000) >> 16U), float((c & 0xFF000000) >> 24U));
	normal = glm::vec3(float((n & 0x000000FF)), float((n & 0x0000FF00) >> 8U), float((n & 0x00FF0000) >> 16U));
	color /= 255.f;
	normal = (normal / 255.f) * 2.f - 1.f;
}

 __device__ void Voxel::SetInfo(glm::vec4 color, glm::vec3 normal) {

	color *= 255.f;
	c = ((unsigned int(color.w) & 0x000000FF) << 24U | (uint(color.z) & 0x000000FF) << 16U | (uint(color.y) & 0x000000FF) << 8U | (uint(color.x) & 0x000000FF));
	normal = (normal + 1.f) / 2.f * 255.f;
	n = ((uint(normal.z) & 0x000000FF) << 16U | (uint(normal.y) & 0x000000FF) << 8U | (uint(normal.x) & 0x000000FF));
}



__device__ glm::vec4 Voxel::PhongLighting(glm::vec3 pos)
{
	glm::vec4 c;
	glm::vec3 n;
	this->GetInfo(c, n);
	n = glm::normalize(n);
	glm::vec3 l = d_Info.lightPos - pos;
	float d = glm::length(l);
	float att = 1.f/(0.1f * d + d * d);

	l /= d;
	float dotnl = glm::dot(n, l);
	glm::vec3 r = n * 2.f * dotnl - l, v = glm::normalize(d_Info.camPos - pos);
	float intensity = (d_Info.ka + att * (d_Info.kd * glm::max(0.f, dotnl) + 
		d_Info.ks * glm::max(0.f, glm::pow(glm::dot(r, v), d_Info.alpha))));
	return glm::vec4(glm::vec3(c) * intensity, c.a);
}

//transfer 3D index to 1D array index
__device__ inline size_t ToArrayIdx(glm::uvec3 coord) {
	return coord.x + coord.y * d_Info.Dim + coord.z * d_Info.Dim * d_Info.Dim;
}
//Get voxel index for current position
__device__ inline glm::uvec3 GetVoxelIndex(glm::vec3 pos) {
	return glm::min(glm::uvec3((pos - d_Info.minAABB)/ d_Info.delta * float(d_Info.Dim)), glm::uvec3(d_Info.Dim - 1));
}
//Get world position given voxel index
__device__ inline glm::vec3 GetVoxelWorldPos(glm::uvec3 idx) {
	return glm::vec3(idx) * d_Info.delta / float(d_Info.Dim) + d_Info.minAABB;
}

__device__ inline glm::vec3 WorldSpaceInterpolation(glm::vec3 A, glm::vec3 B, glm::vec3 C, glm::vec3 P) {
	glm::vec3 AB = B - A, AC = C - A, AP = P - A, N = glm::cross(AB, AC);
	/*P = P - N * glm::dot(N, AP) / glm::length(N);
	AP = P - A;*/
	float DotNN = glm::dot(N, N);
	glm::vec3 uvw;
	uvw[1] = glm::dot(glm::cross(AP, AC), N) / DotNN;
	uvw[2] = glm::dot(glm::cross(AB, AP), N) / DotNN;
	uvw[0] = 1.f - uvw[1] - uvw[2];
	return uvw;
}

__device__ inline bool VoxelTriangleIntersection(Triangle tri, glm::vec3 vMinAABB) {
	const float dotnp = glm::dot(tri.n, vMinAABB);
	if ((dotnp + tri.d1) * (dotnp + tri.d2) > 0)
		return false;
	bool xy, xz, yz;
	xy = (glm::dot(tri.ne_xy[0], glm::vec2(vMinAABB.x, vMinAABB.y)) + tri.de_xy[0]) >= 0 &&
		(glm::dot(tri.ne_xy[1], glm::vec2(vMinAABB.x, vMinAABB.y)) + tri.de_xy[1]) >= 0 &&
		(glm::dot(tri.ne_xy[2], glm::vec2(vMinAABB.x, vMinAABB.y)) + tri.de_xy[2]) >= 0;
	xz = (glm::dot(tri.ne_xz[0], glm::vec2(vMinAABB.x, vMinAABB.z)) + tri.de_xz[0]) >= 0 &&
		(glm::dot(tri.ne_xz[1], glm::vec2(vMinAABB.x, vMinAABB.z)) + tri.de_xz[1]) >= 0 &&
		(glm::dot(tri.ne_xz[2], glm::vec2(vMinAABB.x, vMinAABB.z)) + tri.de_xz[2]) >= 0;
	yz = (glm::dot(tri.ne_yz[0], glm::vec2(vMinAABB.y, vMinAABB.z)) + tri.de_yz[0]) >= 0 &&
		(glm::dot(tri.ne_yz[1], glm::vec2(vMinAABB.y, vMinAABB.z)) + tri.de_yz[1]) >= 0 &&
		(glm::dot(tri.ne_yz[2], glm::vec2(vMinAABB.y, vMinAABB.z)) + tri.de_yz[2]) >= 0;
	return xy || xz || yz;
}

__global__ void VoxelizationKernel(Voxel* voxelList, uint* voxelIdxList, CudaMesh mesh) {

	const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= mesh.triNum) return;
	const Triangle tri = mesh.d_tri[idx];
	const glm::vec3 v0(mesh.d_v[3 * tri.i0], mesh.d_v[3 * tri.i0 + 1], mesh.d_v[3 * tri.i0 + 2]),
					v1(mesh.d_v[3 * tri.i1], mesh.d_v[3 * tri.i1 + 1], mesh.d_v[3 * tri.i1 + 2]),
					v2(mesh.d_v[3 * tri.i2], mesh.d_v[3 * tri.i2 + 1], mesh.d_v[3 * tri.i2 + 2]),
					n0(mesh.d_n[3 * tri.i0], mesh.d_n[3 * tri.i0 + 1], mesh.d_n[3 * tri.i0 + 2]),
					n1(mesh.d_n[3 * tri.i1], mesh.d_n[3 * tri.i1 + 1], mesh.d_n[3 * tri.i1 + 2]),
					n2(mesh.d_n[3 * tri.i2], mesh.d_n[3 * tri.i2 + 1], mesh.d_n[3 * tri.i2 + 2]);
	const float vDelta = d_Info.delta / float(d_Info.Dim);

	glm::vec3 maxAABB(glm::max(v0, glm::max(v1, v2))), minAABB(glm::min(v0, glm::min(v1, v2)));
	glm::uvec3 minVoxel = GetVoxelIndex(minAABB),
		maxVoxel = GetVoxelIndex(maxAABB);
	maxVoxel = glm::max(maxVoxel, minVoxel + 1U); //extend AABB when encounter tiny triangle
	maxVoxel = glm::min(maxVoxel, glm::uvec3(d_Info.Dim - 1));
	//printf("maxVoxel:(%i, %i, %i)\n", maxVoxel.x, maxVoxel.y, maxVoxel.z);
	for(uint i = minVoxel.x; i <= maxVoxel.x; i++)
		for (uint j = minVoxel.y; j <= maxVoxel.y; j++)
			for (uint k = minVoxel.z; k <= maxVoxel.z; k++) {
				glm::vec3 voxelPos = GetVoxelWorldPos(glm::uvec3(i, j, k));
				if (VoxelTriangleIntersection(tri, voxelPos - vDelta/2.f)) {
					glm::vec3 minVoxelAABB = voxelPos - vDelta / 2.f, maxVoxelAABB = minVoxelAABB + vDelta;
					//Voxel-Triangle AABB test

					glm::vec3 uvw = WorldSpaceInterpolation(v0, v1, v2, voxelPos);

					glm::vec4 color(0.9f);
					glm::vec3 normal = glm::normalize(uvw[0] * n0 + uvw[1] * n1 + uvw[2] * n2);
					size_t arrayIdx = atomicAdd(&voxelCounter, 1);

					voxelList[arrayIdx].SetInfo(color, normal);
					voxelIdxList[arrayIdx] = ConvUvec3ToUint(glm::uvec3(i, j, k));

				}

			}


}

__global__ void PreProcessTriangleKernel(CudaMesh mesh) {
	const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= mesh.triNum) return;
	Triangle tri;
	tri.i0 = mesh.d_idx[3 * idx], tri.i1 = mesh.d_idx[3 * idx + 1], tri.i2 = mesh.d_idx[3 * idx + 2];
	glm::vec3 v[3];
	v[0] = glm::vec3(mesh.d_v[3 * tri.i0], mesh.d_v[3 * tri.i0 + 1], mesh.d_v[3 * tri.i0 + 2]);
	v[1] = glm::vec3(mesh.d_v[3 * tri.i1], mesh.d_v[3 * tri.i1 + 1], mesh.d_v[3 * tri.i1 + 2]);
	v[2] = glm::vec3(mesh.d_v[3 * tri.i2], mesh.d_v[3 * tri.i2 + 1], mesh.d_v[3 * tri.i2 + 2]);
	const glm::vec3 e[3] = { v[1] - v[0], v[2] - v[1], v[0] - v[2] };
	const float delta = d_Info.delta / (float)d_Info.Dim;
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

}

//__global__ void RayMarchingKernel(unsigned int* d_pbo, Voxel* voxelList,  const unsigned int w, const unsigned int h) {
//	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x,
//		y = blockDim.y * blockIdx.y + threadIdx.y;
//	if (x >= w || y >= h) return;
//	d_pbo[y * w + x] = 0;
//	const float u = float(x) / float(w), v = float(y) / float(h);
//	float4 frontSample = tex2D(frontTex, u, v), backSample = tex2D(backTex, u, v);
//	if (frontSample.w < 1.f) return;
//
//	glm::vec3 frontPos(frontSample.x, frontSample.y, frontSample.z),
//		backPos(backSample.x, backSample.y, backSample.z);
//	glm::vec3 dir = backPos - frontPos;
//	const float stepSize = d_Info.delta / d_Info.Dim /2.f , dirLength = glm::length(dir);
//	const unsigned maxStep = dirLength / stepSize;
//	dir /= dirLength;//Normalize
//	glm::vec3 curPos = frontPos;
//	glm::uvec3 voxelIdx;
//	//Trace voxels
//	for (int i = 0; i < maxStep; i++) {
//		
//		voxelIdx = GetVoxelIndex(curPos);
//		Voxel voxel = voxelList[ToArrayIdx(voxelIdx)];
//		if (!voxel.empty()) {
//			glm::vec3 voxelPos = GetVoxelWorldPos(voxelIdx);
//			d_pbo[y * w + x] = ConvVec4ToUint(glm::vec4(voxel.PhongLighting(voxelPos), 1.f));
//			break;
//		}
//		curPos += dir * stepSize;
//	}
//
//}

//void initCudaTexture()
//{
//	frontTex.addressMode[0] = cudaAddressModeWrap;
//	frontTex.addressMode[1] = cudaAddressModeWrap;
//	frontTex.filterMode = cudaFilterModeLinear;
//	frontTex.normalized = true;
//
//	backTex.addressMode[0] = cudaAddressModeWrap;
//	backTex.addressMode[1] = cudaAddressModeWrap;
//	backTex.filterMode = cudaFilterModeLinear;
//	backTex.normalized = true;
//}

void Voxelization(CudaMesh& cuMesh, Voxel*& d_voxel, uint*& d_idx)
{
	if (cudaMemcpyToSymbol(d_Info, &Info, sizeof(VoxelizationInfo)) != cudaSuccess)
		printf("cudaMemcpy to constant failed\n");

	cudaError_t cudaStatus;
	//PreProcess Triangle
	cudaStatus = cudaMalloc((void**)&cuMesh.d_tri, cuMesh.triNum * sizeof(Triangle));
	if (cudaStatus != cudaSuccess) printf("d_tri cudaMalloc Failed\n");
	dim3 blockDim = 256, gridDim = cuMesh.triNum / blockDim.x + 1;
	PreProcessTriangleKernel <<< gridDim, blockDim >>> (cuMesh);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) printf("PreprocessTriangle Launch Kernel Failed\n");
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) printf("cudaDeviceSynchronize Failed\n");
	cudaStatus = cudaFree(cuMesh.d_idx);
	if (cudaStatus != cudaSuccess) printf("d_idx cudaFree Failed, error: %s\n", cudaGetErrorString(cudaStatus));
	
	cudaStatus = cudaMemcpyToSymbol(voxelCounter, &Info.Counter, sizeof(uint));
	if (cudaStatus != cudaSuccess) printf("counter cudaMemcpy Failed\n");

	size_t voxelSize = voxelDim * voxelDim * voxelDim * sizeof(Voxel);

	cudaStatus = cudaMalloc((void**)&d_voxel, voxelSize);
	if (cudaStatus != cudaSuccess) printf("d_voxel cudaMalloc Failed\n");
	cudaStatus = cudaMalloc((void**)&d_idx, voxelDim * voxelDim * voxelDim * sizeof(uint));
	if (cudaStatus != cudaSuccess) printf("d_idx cudaMalloc Failed\n");
	
	clock_t t;

	t = clock();
	VoxelizationKernel << <gridDim, blockDim >> > (d_voxel, d_idx, cuMesh);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) printf("cuda Launch Kernel Failed\n");
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) printf("cudaDeviceSynchronize Failed\n");
	t = clock() - t;
	printf("Voxelization finished, time : %f\n", (float)t / CLOCKS_PER_SEC);	

	cudaStatus = cudaMemcpyFromSymbol(&Info.Counter, voxelCounter, sizeof(uint));
	if (cudaStatus != cudaSuccess) printf("counter cudaMemcpyFromSymbol Failed\n");
	printf("Voxel Count: %i\n", Info.Counter);

	//Free CudaMesh
	cudaStatus = cudaFree(cuMesh.d_v);
	if (cudaStatus != cudaSuccess) printf("d_v cudaFree Failed, error: %s\n", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaFree(cuMesh.d_n);
	if (cudaStatus != cudaSuccess) printf("d_n cudaFree Failed, error: %s\n", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaFree(cuMesh.d_tri);
	if (cudaStatus != cudaSuccess) printf("d_tri cudaFree Failed, error: %s\n", cudaGetErrorString(cudaStatus));
	
	//initCudaTexture();
}

//void RunRayMarchingKernel(uint* d_pbo, cudaArray_t front, cudaArray_t back, Voxel* d_voxel)
//{
//	if (cudaMemcpyToSymbol(d_Info, &Info, sizeof(VoxelizationInfo)) != cudaSuccess)
//		printf("cudaMemcpy to constant failed\n");
//
//	cudaChannelFormatDesc format = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
//	cudaError_t cudaStatus;
//	if (cudaBindTextureToArray(&frontTex, front, &format) != cudaSuccess)
//		printf("front texture bind failed\n");
//	if (cudaBindTextureToArray(&backTex, back, &format) != cudaSuccess)
//		printf("back texture bind failed\n");
//	//launch cuda kernel
//	dim3 blockDim(16, 16, 1), gridDim(WINDOW_WIDTH / blockDim.x + 1, WINDOW_HEIGHT / blockDim.y + 1, 1);
//	RayMarchingKernel << <gridDim, blockDim >> > (d_pbo, d_voxel, WINDOW_WIDTH, WINDOW_HEIGHT);
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) printf("raymarching cuda Launch Kernel Failed\n");
//	cudaStatus = cudaDeviceSynchronize();
//	if(cudaStatus != cudaSuccess)
//		printf("cudaDeviceSynchronize Failed, error: %s\n", cudaGetErrorString(cudaStatus));
//
//	if (cudaUnbindTexture(frontTex) != cudaSuccess)
//		printf("cuda unbind texture failed\n");
//	if (cudaUnbindTexture(backTex) != cudaSuccess)
//		printf("cuda unbind texture failed\n");
//}



