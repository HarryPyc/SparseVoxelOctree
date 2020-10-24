#include "Octree.cuh"
#include "device_launch_parameters.h"
#include <glm/gtc/integer.hpp>
#include <time.h>
#define NULLPTR 0U
#define MARKED 0xFFFFFFFF
extern VoxelizationInfo Info;
extern uint h_MIPMAP;
__constant__ VoxelizationInfo d_Info;
__constant__ uint maxLevel, curLevel, voxelCount;
__constant__ uint start, end, MIPMAP;
//__constant__ VoxelizationInfo d_Info;
__device__ uint curIdx;
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
		_idx = idx % glm::uvec3(1 << (maxLevel - i)) / glm::uvec3(1 << (maxLevel - i - 1));
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


__global__ void MemcpyVoxelToLeafNodeKernel(Node* d_node, Voxel* voxelSrc, uint* d_idx) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= voxelCount) return;
	uint nodeIdx = 0;

	glm::uvec3 idx = ConvUintToUvec3(d_idx[x]), _idx = glm::uvec3(0);

	for (uint i = 0; i < maxLevel; i++) {
		_idx = idx % glm::uvec3(1 << (maxLevel - i)) / glm::uvec3(1 << (maxLevel - i - 1));
		//printf("[%i,%i,%i]\n", _idx.x, _idx.y, _idx.z);
		nodeIdx = d_node[nodeIdx].ptr + _idx.x + _idx.y * 2 + _idx.z * 4;
	}
	//printf("voxel[%u], node[%u]\n", x, nodeIdx);
	//leaf node pointer points to voxel list
	d_node[nodeIdx].voxel = voxelSrc[x];

}
__global__ void MimmapKernel(Node* d_node, Voxel* d_voxel) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + start;
	if (x >= end) return;
	const Node root = d_node[x];
	if (root.ptr == NULLPTR) return;

	glm::vec3 color(0.f), normal(0.f);
	float counter = 0.f;
	for (uint i = 0; i < 8; i++) {
		glm::vec3 c, n;
		Voxel voxel = d_node[root.ptr + i].voxel;
		if (!voxel.empty()) {
			voxel.GetInfo(c, n);
			color += c, normal += n;
			counter++;
		}
	}
	d_node[x].voxel.SetInfo(color/counter, normal/counter);
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

	cudaStatus = cudaMalloc((void**)&d_node, Info.Counter * sizeof(Node));
	if (cudaStatus != cudaSuccess) printf("d_Node cudaMalloc Failed\n");
	clock_t time = clock();
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
	
	//Copy voxel to leaf node
	dim3 blockDim = 256, gridDim = Info.Counter / blockDim.x + 1;
	MemcpyVoxelToLeafNodeKernel << <gridDim, blockDim >> > (d_node, d_voxel, d_idx);
	if (cudaStatus != cudaSuccess) printf("MemcpyVoxelToLeafNodeKernel launch Failed, error: %s\n", cudaGetErrorString(cudaStatus));
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) printf("MemcpyVoxelToLeafNodeKernel cudaDeviceSynchronize Failed\n");
	cudaStatus = cudaFree(d_voxel);
	if (cudaStatus != cudaSuccess) printf("d_voxel cudaFree Failed\n");
	cudaStatus = cudaFree(d_idx);
	if (cudaStatus != cudaSuccess) printf("d_idx cudaFree Failed\n");

	//Mimmap voxel value from bottom to up
	for (int i = h_maxLevel - 2; i >= 0; i--) {
		cudaMemcpyToSymbol(curLevel, &i, 4);
		cudaMemcpyToSymbol(start, startArr + i, 4);
		cudaMemcpyToSymbol(end, endArr + i, 4);

		gridDim = (endArr[i] - startArr[i]) / blockDim.x + 1;
		MimmapKernel << <gridDim, blockDim >> > (d_node, d_voxel);
		if (cudaStatus != cudaSuccess) printf("MimmapKernel launch Failed, error: %s\n", cudaGetErrorString(cudaStatus));
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) printf("MimmapKernel cudaDeviceSynchronize Failed\n");
	}
	time = clock() - time;
	printf("Octree Constructed, time: %f\n", float(time) / CLOCKS_PER_SEC);
	printf("Octree Total Nodes : %u\n", h_curIdx);
	//Node h_node[8777];
	//cudaMemcpy(h_node, d_node, sizeof(Node) * 8777, cudaMemcpyDeviceToHost);
	delete[] startArr, delete[] endArr;
	initCudaTexture();
	cudaStatus = cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 10);
	if (cudaStatus != cudaSuccess) printf("cudaDeviceSetLimit Failed\n");
}
struct Ray {
	glm::vec3 o, d, invD;
	__device__ Ray(glm::vec3 origin, glm::vec3 dir) : o(origin), d(dir) {
		invD = glm::vec3(1.f) / dir;
	}
	__device__ ~Ray() {};
	__device__ inline bool RayAABBIntersection(glm::vec3 minAABB, glm::vec3 maxAABB, float &t) {
		glm::vec3 t0s = (minAABB - o) * invD;
		glm::vec3 t1s = (maxAABB - o) * invD;

		glm::vec3 tsmaller = glm::min(t0s, t1s);
		glm::vec3 tbigger = glm::max(t0s, t1s);

		float tmin = glm::max(-999.f, glm::max(tsmaller[0], glm::max(tsmaller[1], tsmaller[2])));
		float tmax = glm::min(999.f, glm::min(tbigger[0], glm::min(tbigger[1], tbigger[2])));
		t = (tmin + tmax) / 2.f;
		return (tmin < tmax);
	}
};
struct HitInfo {
	glm::uvec3 idx;
	float t;
	__device__ HitInfo() {};
	__device__ HitInfo(glm::uvec3 _idx, float _t) : idx(_idx), t(_t) {};
};
__device__ Voxel OctreeTraverse(Node* d_node, Node root, Ray ray, glm::vec3 minAABB, uint currentLevel, float& t) {
	if (root.ptr == NULLPTR)
		return root.voxel;
	if (currentLevel == MIPMAP)
		return root.voxel;

	currentLevel++;
	HitInfo hits[8];
	int counter = 0;
	float temp = 999.f;
	Voxel res;
	const float delta = d_Info.delta / float((1 << (currentLevel)));
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < 2; k++) {
				glm::uvec3 idx(i, j, k);
				glm::vec3 _minAABB = minAABB + glm::vec3(idx) * delta;
				float _t;
				if (ray.RayAABBIntersection(_minAABB, _minAABB + delta, _t)) {
					HitInfo hit(idx, _t);
					hits[counter++] = hit;
				}
			}
	for(int i = 0; i < counter - 1; i++)
		for (int j = 0; j < counter - i - 1; j++) {
			if (hits[j].t > hits[j + 1].t) {
				HitInfo temp = hits[j];
				hits[j] = hits[j + 1];
				hits[j + 1] = temp;
			}
		}
	for (int i = 0; i < counter; i++) {
		Node _root = d_node[root.ptr + hits[i].idx.x + hits[i].idx.y * 2 + hits[i].idx.z * 4];
		glm::vec3 _minAABB = minAABB + glm::vec3(hits[i].idx) * delta;
		Voxel voxel = OctreeTraverse(d_node, _root, ray, _minAABB, currentLevel,  t);
		if (!voxel.empty()) {
			if (currentLevel == MIPMAP)
				t = hits[i].t;
			return voxel;
		}
	}
	return res;
}
__global__ void RayCastKernel(uint* d_pbo, Node* d_node, const uint w, const uint h) {
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

	glm::vec3 color, pos;
	float t = 999.f;
	Voxel voxel = OctreeTraverse(d_node, d_node[0], ray, d_Info.minAABB, 0, t);
	pos = ray.o + t * ray.d;
	color = voxel.PhongLighting(pos);
	d_pbo[y * w + x] = ConvVec4ToUint(glm::vec4(color, 1));

}

void RayCastingOctree(uint* d_pbo, cudaArray_t front, cudaArray_t back, Node* d_node)
{
	if (cudaMemcpyToSymbol(d_Info, &Info, sizeof(VoxelizationInfo)) != cudaSuccess)
		printf("cudaMemcpy to constant failed\n");
	cudaMemcpyToSymbol(MIPMAP, &h_MIPMAP, 4);

	cudaChannelFormatDesc format = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaError_t cudaStatus;
	if (cudaBindTextureToArray(&frontTex, front, &format) != cudaSuccess)
		printf("front texture bind failed\n");
	if (cudaBindTextureToArray(&backTex, back, &format) != cudaSuccess)
		printf("back texture bind failed\n");
	//launch cuda kernel
	dim3 blockDim(16, 16, 1), gridDim(WINDOW_WIDTH / blockDim.x + 1, WINDOW_HEIGHT / blockDim.y + 1, 1);
	RayCastKernel << <gridDim, blockDim >> > (d_pbo, d_node, WINDOW_WIDTH, WINDOW_HEIGHT);
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



