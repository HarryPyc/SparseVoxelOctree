#include "Octree.cuh"
#include "device_launch_parameters.h"
#include <glm/gtc/integer.hpp>
#define NULLPTR 0U
#define MARKED 0xFFFFFFFF
extern VoxelizationInfo Info;
__constant__ VoxelizationInfo d_Info;
__constant__ uint maxLevel, curLevel, voxelCount;
__constant__ uint start, end;
//__constant__ VoxelizationInfo d_Info;
__device__ uint curIdx;
__device__ uint d_counter = 0;
texture<float4, 2, cudaReadModeElementType> frontTex, backTex;

__host__ __device__ Node::Node() : voxel() {
	ptr = NULLPTR;
}

__host__ __device__ Node::~Node() {
}

__global__ void MarkKernel(Node* d_node, uint* d_idx) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= voxelCount) return;
	uint nodeIdx = 0;
	//printf("marked: %u", MARKED);
	if (d_node[nodeIdx].ptr == NULLPTR || d_node[nodeIdx].ptr == MARKED) {
		d_node[nodeIdx].ptr = MARKED;
		return;
	}

	glm::uvec3 idx = ConvUintToUvec3(d_idx[x]), _idx = glm::uvec3(0);

	for (uint i = 0; i <= curLevel; i++) {
		_idx = idx % glm::uvec3(1 << (maxLevel - i + 1)) / glm::uvec3(1 << (maxLevel - i));
		nodeIdx = d_node[nodeIdx].ptr + _idx.x + _idx.y * 2 + _idx.z * 4;
		if (d_node[nodeIdx].ptr == NULLPTR || d_node[nodeIdx].ptr == MARKED) {
			d_node[nodeIdx].ptr = MARKED;
			return;
		}
	}
	
}
__global__ void AllocateKernel(Node* d_node) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + start;
	if (x >= end) return;

	if(d_node[x].ptr == MARKED)
		d_node[x].ptr = atomicAdd(&curIdx, 8);

}

__global__ void AllocateLeafNodeVoxelPtrKernel(Node* d_node) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + start;
	if (x >= end) return;
	if(d_node[x].ptr == MARKED)
		d_node[x].ptr = atomicAdd(&d_counter, 8);
}
__global__ void MemcpyVoxelToLeafNodeKernel(Node* d_node, Voxel* voxelSrc, Voxel* voxelDst, uint* d_idx) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= voxelCount) return;
	uint nodeIdx = 0;

	glm::uvec3 idx = ConvUintToUvec3(d_idx[x]), _idx = glm::uvec3(0);

	for (uint i = 0; i <= maxLevel; i++) {
		_idx = idx % glm::uvec3(1 << (maxLevel - i + 1)) / glm::uvec3(1 << (maxLevel - i));
		//printf("[%i,%i,%i]\n", _idx.x, _idx.y, _idx.z);
		nodeIdx = d_node[nodeIdx].ptr + _idx.x + _idx.y * 2 + _idx.z * 4;
	}

	//leaf node pointer points to voxel list
	voxelDst[d_node[nodeIdx].ptr + _idx.x + _idx.y * 2 + _idx.z * 4] = voxelSrc[x];

}
__global__ void MimmapKernel(Node* d_node, Voxel* d_voxel) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + start;
	if (x >= end) return;
	glm::vec3 color(0.f), normal(0.f);
	if (curLevel == maxLevel - 1) {
		const uint voxelPtr = d_node[x].ptr;
		for (uint i = 0; i < 8; i++) {
			glm::vec3 c, n;
			d_voxel[voxelPtr + i].GetInfo(c, n);
			color += c, normal += n;
		}
	}
	else {
		const Node root = d_node[x];
		for (uint i = 0; i < 8; i++) {
			glm::vec3 c, n;
			d_node[root.ptr + i].voxel.GetInfo(c, n);
			color += c, normal += n;
		}
	}
	d_node[x].voxel.SetInfo(color, normal);
}

void initCudaTexture()
{
	frontTex.addressMode[0] = cudaAddressModeWrap;
	frontTex.addressMode[1] = cudaAddressModeWrap;
	frontTex.filterMode = cudaFilterModeLinear;
	frontTex.normalized = true;

	backTex.addressMode[0] = cudaAddressModeWrap;
	backTex.addressMode[1] = cudaAddressModeWrap;
	backTex.filterMode = cudaFilterModeLinear;
	backTex.normalized = true;
}

void OctreeConstruction(Node*& d_node, Voxel*& d_voxel, uint* d_idx)
{
	
	cudaError_t cudaStatus;
	const uint h_maxLevel = glm::log2(Info.Dim);
	uint h_start = 0, h_curIdx = 1;
	uint* startArr = new uint[h_maxLevel], *endArr = new uint[h_maxLevel];
	cudaMemcpyToSymbol(maxLevel, &h_maxLevel, sizeof(uint));
	cudaMemcpyToSymbol(voxelCount, &Info.Counter, sizeof(uint));
	cudaMemcpyToSymbol(start, &h_start, sizeof(uint));
	cudaMemcpyToSymbol(end, &h_curIdx, sizeof(uint));
	cudaMemcpyToSymbol(curIdx, &h_curIdx, sizeof(uint));

	cudaStatus = cudaMalloc((void**)&d_node, Info.Counter * 3 * sizeof(Node));
	if (cudaStatus != cudaSuccess) printf("d_Node cudaMalloc Failed\n");

	for (uint i = 0; i < h_maxLevel; i++) {
		cudaMemcpyToSymbol(curLevel, &i, sizeof(uint));

		//Mark Node that need to be subdivied
		dim3 blockDim = 256, gridDim = Info.Counter / blockDim.x + 1;
		MarkKernel << <gridDim, blockDim >> > (d_node, d_idx);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) printf("MarkKernel launch Failed, error: %s\n", cudaGetErrorString(cudaStatus));
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) printf("MarkKernel cudaDeviceSynchronize Failed\n");
		//Allocate new node from node pool
		gridDim = (h_curIdx - h_start) / blockDim.x + 1;
		AllocateKernel << <gridDim, blockDim >> > (d_node);
		if (cudaStatus != cudaSuccess) printf("AllocateKernel launch Failed, error: %s\n", cudaGetErrorString(cudaStatus));
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) printf("AllocateKernel cudaDeviceSynchronize Failed\n");
		h_start = h_curIdx;
		cudaMemcpyFromSymbol(&h_curIdx, curIdx, sizeof(uint));
		cudaMemcpyToSymbol(start, &h_start, sizeof(uint));
		cudaMemcpyToSymbol(end, &h_curIdx, sizeof(uint));
		startArr[i] = h_start, endArr[i] = h_curIdx;
	}
	//Mark leaf node
	cudaMemcpyToSymbol(curLevel, &h_maxLevel, sizeof(uint));
	dim3 blockDim = 256, gridDim = Info.Counter / blockDim.x + 1;
	MarkKernel << <gridDim, blockDim >> > (d_node, d_idx);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) printf("MarkKernel launch Failed, error: %s\n", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) printf("MarkKernel cudaDeviceSynchronize Failed\n");
	//allocate leaf node
	blockDim = 256, gridDim = (h_curIdx - h_start) / blockDim.x + 1;
	AllocateLeafNodeVoxelPtrKernel << <gridDim, blockDim >> > (d_node);//A Leaf Node Per Thread
	if (cudaStatus != cudaSuccess) printf("AllocateLeafNodeVoxelPtrKernel launch Failed, error: %s\n", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) printf("AllocateLeafNodeVoxelPtrKernel cudaDeviceSynchronize Failed\n");
	uint h_counter;
	cudaMemcpyFromSymbol(&h_counter, d_counter, 4);
	Voxel* d_nodeVoxel;
	cudaMalloc((void**)&d_nodeVoxel, sizeof(Voxel) * (h_counter));
	//Copy voxel to leaf node
	blockDim = 256, gridDim = Info.Counter / blockDim.x + 1;
	MemcpyVoxelToLeafNodeKernel << <gridDim, blockDim >> > (d_node, d_voxel, d_nodeVoxel, d_idx);
	if (cudaStatus != cudaSuccess) printf("MemcpyVoxelToLeafNodeKernel launch Failed, error: %s\n", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) printf("MemcpyVoxelToLeafNodeKernel cudaDeviceSynchronize Failed\n");
	cudaStatus = cudaFree(d_voxel);
	if (cudaStatus != cudaSuccess) printf("d_voxel cudaFree Failed\n");
	cudaStatus = cudaFree(d_idx);
	if (cudaStatus != cudaSuccess) printf("d_idx cudaFree Failed\n");
	d_voxel = d_nodeVoxel;
	//Mimmap voxel value from bottom to up
	for (int i = h_maxLevel - 1; i >= 0; i--) {
		cudaMemcpyToSymbol(curLevel, &i, 4);
		cudaMemcpyToSymbol(start, startArr + i, 4);
		cudaMemcpyToSymbol(end, endArr + i, 4);

		gridDim = (endArr[i] - startArr[i]) / blockDim.x + 1;
		MimmapKernel << <gridDim, blockDim >> > (d_node, d_voxel);
		if (cudaStatus != cudaSuccess) printf("MimmapKernel launch Failed, error: %s\n", cudaGetErrorString(cudaStatus));
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) printf("MimmapKernel cudaDeviceSynchronize Failed\n");

	}
	//Node h_node[8777];
	//cudaMemcpy(h_node, d_node, sizeof(Node) * 8777, cudaMemcpyDeviceToHost);
	delete[] startArr, delete[] endArr;
	initCudaTexture();
}
struct Ray {
	glm::vec3 o, d, invD;
	__device__ Ray(glm::vec3 origin, glm::vec3 dir) : o(origin), d(dir) {
		invD = glm::vec3(1.f) / dir;
	}
	__device__ ~Ray() {};
	__device__ inline bool RayAABBIntersection(glm::vec3 minAABB, glm::vec3 maxAABB) {
		glm::vec3 t0s = (minAABB - o) * invD;
		glm::vec3 t1s = (maxAABB - o) * invD;

		glm::vec3 tsmaller = glm::min(t0s, t1s);
		glm::vec3 tbigger = glm::max(t0s, t1s);

		float tmin = glm::max(-999.f, glm::max(tsmaller[0], glm::max(tsmaller[1], tsmaller[2])));
		float tmax = glm::min(999.f, glm::min(tbigger[0], glm::min(tbigger[1], tbigger[2])));

		return (tmin < tmax);
	}
};

__device__ Voxel OctreeTraverse(Node* d_node, Voxel* d_voxel, Node root, Ray ray, glm::vec3 minAABB, uint currentLevel, uint targetLevel) {
	if (root.ptr == NULLPTR)
		return root.voxel;
	if (currentLevel == targetLevel)
		return root.voxel;
	currentLevel++;

	Voxel voxel;
	const float delta = d_Info.delta / float((1 << currentLevel));
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < 2; k++) {
				glm::uvec3 idx(i, j, k);
				glm::vec3 _minAABB = minAABB + glm::vec3(idx) * delta;
				if (ray.RayAABBIntersection(_minAABB, _minAABB + delta)) {
					if (currentLevel == maxLevel + 1)
						voxel = d_voxel[root.ptr + idx.x + idx.y * 2 + idx.z * 4];
					else {
						Node _root = d_node[root.ptr + idx.x + idx.y * 2 + idx.z * 4];
						//printf("curLevel: %i, root.ptr: %u\n", currentLevel, root.ptr);
						voxel = OctreeTraverse(d_node, d_voxel, _root, ray, _minAABB, currentLevel, targetLevel);
					}
					if (!voxel.empty())
						return voxel;
				}
			}
	return voxel;
}
__global__ void RayCastKernel(uint* d_pbo, Node* d_node, Voxel* d_voxel, const uint w, const uint h) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x,
			y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= w || y >= h) return;
	d_pbo[y * w + x] = 0;
	const float u = float(x) / float(w), v = float(y) / float(h);
	float4 frontSample = tex2D(frontTex, u, v), backSample = tex2D(backTex, u, v);
	if (frontSample.w < 1.f) return;
		
	glm::vec3 frontPos(frontSample.x, frontSample.y, frontSample.z),
		backPos(backSample.x, backSample.y, backSample.z);
	glm::vec3 dir = glm::normalize(backPos - frontPos);
	Ray ray(frontPos, dir);

	glm::vec3 color, normal;
	Voxel voxel = OctreeTraverse(d_node, d_voxel, d_node[0], ray, d_Info.minAABB, 0, 4);
	voxel.GetInfo(color, normal);
	d_pbo[y * w + x] = ConvVec4ToUint(glm::vec4(color, 1));

}

void RayCastingOctree(uint* d_pbo, cudaArray_t front, cudaArray_t back, Voxel* d_voxel, Node* d_node)
{
	if (cudaMemcpyToSymbol(d_Info, &Info, sizeof(VoxelizationInfo)) != cudaSuccess)
		printf("cudaMemcpy to constant failed\n");

	cudaChannelFormatDesc format = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaError_t cudaStatus;
	if (cudaBindTextureToArray(&frontTex, front, &format) != cudaSuccess)
		printf("front texture bind failed\n");
	if (cudaBindTextureToArray(&backTex, back, &format) != cudaSuccess)
		printf("back texture bind failed\n");
	//launch cuda kernel
	dim3 blockDim(16, 16, 1), gridDim(WINDOW_WIDTH / blockDim.x + 1, WINDOW_HEIGHT / blockDim.y + 1, 1);
	RayCastKernel << <gridDim, blockDim >> > (d_pbo, d_node, d_voxel, WINDOW_WIDTH, WINDOW_HEIGHT);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) printf("raymarching cuda Launch Kernel Failed\n");
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess)
		printf("raymarching cudaDeviceSynchronize Failed, error: %s\n", cudaGetErrorString(cudaStatus));

	if (cudaUnbindTexture(frontTex) != cudaSuccess)
		printf("cuda unbind texture failed\n");
	if (cudaUnbindTexture(backTex) != cudaSuccess)
		printf("cuda unbind texture failed\n");
}



