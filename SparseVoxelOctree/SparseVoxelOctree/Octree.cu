#include "Octree.cuh"
#include "device_launch_parameters.h"
#include <glm/gtc/integer.hpp>

extern VoxelizationInfo Info;
__constant__ VoxelizationInfo d_Info;
__constant__ uint maxLevel, curLevel, voxelCount;
__constant__ uint start, end;
//__constant__ VoxelizationInfo d_Info;
__device__ uint curIdx;
__device__ uint d_counter = 0;
texture<float4, 2, cudaReadModeElementType> frontTex, backTex;

__device__ Node::Node() {
	ptr = 0, voxelPtr = 0;
}

__device__ Node::~Node() {
}

__global__ void MarkKernel(Node* d_node, uint* d_idx) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= voxelCount) return;
	uint nodeIdx = 0;

	glm::uvec3 idx = ConvUintToUvec3(d_idx[x]), _idx = glm::uvec3(0);

	for (uint i = 0; i <= curLevel; i++) {
		
		if (d_node[nodeIdx].ptr == 0) {
			d_node[nodeIdx].ptr = 1 << 31U;
			break;
		}
		else{
			idx -= _idx * glm::uvec3(1 << (maxLevel - i + 1));
			_idx = idx / glm::uvec3(1 << (maxLevel - i));
			//printf("[%i,%i,%i]\n", _idx.x, _idx.y, _idx.z);
			nodeIdx = d_node[nodeIdx].ptr + _idx.x + _idx.y * 2 + _idx.z * 4;
		}
	}
	
}
__global__ void AllocateKernel(Node* d_node) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + start;
	if (x >= curIdx) return;
	//printf("node[%i].ptr = %i\n", x, d_node[x].ptr);
	if(d_node[x].ptr>>31U)
		d_node[x].ptr = atomicAdd(&curIdx, 8);

}

__global__ void AllocateNodeVoxelPtrKernel(Node* d_node) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= curIdx) return;
	if(x < start)
		d_node[x].voxelPtr = atomicAdd(&d_counter, 1);
	else
		d_node[x].voxelPtr = atomicAdd(&d_counter, 9);//1 for self, 8 for child voxels

}
__global__ void MemcpyVoxelToLeafNodeKernel(Node* d_node, Voxel* voxelSrc, Voxel* voxelDst, uint* d_idx) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= voxelCount) return;
	uint nodeIdx = 0;

	glm::uvec3 idx = ConvUintToUvec3(d_idx[x]), _idx = glm::uvec3(0);

	for (uint i = 0; i < maxLevel; i++) {
		idx -= _idx * glm::uvec3(1 << (maxLevel - i + 1));
		_idx = idx / glm::uvec3(1 << (maxLevel - i));
		//printf("[%i,%i,%i]\n", _idx.x, _idx.y, _idx.z);
		nodeIdx = d_node[nodeIdx].ptr + _idx.x + _idx.y * 2 + _idx.z * 4;
	}

	idx -= _idx * glm::uvec3(2);

	voxelDst[d_node[nodeIdx].voxelPtr + 1 + idx.x + idx.y * 2 + idx.z * 4] = voxelSrc[x];

}
__global__ void MimmapKernel(Node* d_node, Voxel* d_voxel) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + start;
	if (x >= curIdx) return;
	glm::vec3 color, normal;
	if (curLevel == maxLevel - 1) {
		const uint voxelPtr = d_node[x].voxelPtr + 1;
		for (uint i = 0; i < 8; i++) {
			glm::vec3 c, n;
			d_voxel[voxelPtr + i].GetInfo(c, n);
			color += c, normal += n;
		}
		d_voxel[voxelPtr].SetInfo(color / 8.f, glm::normalize(normal / 8.f));
	}
	else {
		const Node root = d_node[x];
		for (uint i = 0; i < 8; i++) {
			glm::vec3 c, n;
			d_voxel[d_node[root.ptr + i].voxelPtr].GetInfo(c, n);
			color += c, normal += n;
		}
		d_voxel[root.voxelPtr].SetInfo(color / 8.f, glm::normalize(normal / 8.f));
	}

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
	const uint h_maxLevel = glm::log2(Info.Dim) - 1;
	uint h_start = 0, h_curIdx = 1;
	uint* startArr = new uint[h_maxLevel], *endArr = new uint[h_maxLevel];
	cudaMemcpyToSymbol(maxLevel, &h_maxLevel, sizeof(uint));
	cudaMemcpyToSymbol(voxelCount, &Info.Counter, sizeof(uint));
	cudaMemcpyToSymbol(start, &h_start, sizeof(uint));
	cudaMemcpyToSymbol(curIdx, &h_curIdx, sizeof(uint));

	cudaStatus = cudaMalloc((void**)&d_node, Info.Counter * 2 * sizeof(Node));
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
		startArr[i] = h_start, endArr[i] = h_curIdx;
	}
	dim3 blockDim = 256, gridDim = h_curIdx / blockDim.x + 1;
	AllocateNodeVoxelPtrKernel << <gridDim, blockDim >> > (d_node);//A Leaf Node Per Thread
	if (cudaStatus != cudaSuccess) printf("AllocateLeafNodeKernel launch Failed, error: %s\n", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) printf("AllocateLeafNodeKernel cudaDeviceSynchronize Failed\n");
	uint h_counter;
	cudaMemcpyFromSymbol(&h_counter, d_counter, 4);
	Voxel* d_nodeVoxel;
	cudaMalloc((void**)&d_nodeVoxel, sizeof(Voxel) * (h_counter - 1));

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
	
	delete[] startArr, delete[] endArr;
	initCudaTexture();
}
__global__ void RayCastKernel(uint* d_pbo, Node* d_node, Voxel* d_voxel, const uint w, const uint h) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x,
			y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= w || y >= h) return;
	d_pbo[y * w + x] = 0;
	const float u = float(x) / float(w), v = float(y) / float(h);
	float4 frontSample = tex2D(frontTex, u, v), backSample = tex2D(backTex, u, v);
	//if (frontSample.w < 1.f) return;
		
	glm::vec3 frontPos(frontSample.x, frontSample.y, frontSample.z),
		backPos(backSample.x, backSample.y, backSample.z);
	glm::vec3 dir = backPos - frontPos;
	const float stepSize = d_Info.delta / d_Info.Dim /2.f , dirLength = glm::length(dir);
	const unsigned maxStep = dirLength / stepSize;
	dir /= dirLength;//Normalize
	glm::vec3 curPos = frontPos;

	glm::vec3 color, normal;
	d_voxel[d_node[4].voxelPtr].GetInfo(color, normal);
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
		printf("cudaDeviceSynchronize Failed, error: %s\n", cudaGetErrorString(cudaStatus));

	if (cudaUnbindTexture(frontTex) != cudaSuccess)
		printf("cuda unbind texture failed\n");
	if (cudaUnbindTexture(backTex) != cudaSuccess)
		printf("cuda unbind texture failed\n");
}



