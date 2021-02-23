#include "Octree.cuh"
#include "device_launch_parameters.h"
#include <glm/gtc/integer.hpp>
#include <time.h>
#include "Morton.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define NULLPTR 0U
#define MARKED 0xFFFFFFFF
extern VoxelizationInfo Info;
extern uint h_MIPMAP;
__constant__ VoxelizationInfo d_Info;
__constant__ int curDepth, MIPMAP, d_MAX_DEPTH;
Node root;
__device__ int nodeCounter, dynamicNodeCounter, traverseCounter;

texture<float4, 2, cudaReadModeElementType> backTex;
texture<uint4, 1, cudaReadModeElementType> octree;
texture<float4, cudaTextureTypeCubemap, cudaReadModeElementType> skyBox;

__host__ __device__ Node::Node() : voxel() {
	ptr = NULLPTR;
}

__host__ __device__ Node::Node(uint _ptr, Voxel _vox) {
	ptr = _ptr;
	voxel = _vox;
}
__host__ __device__ Node::~Node() {
}

void initCudaTexture()
{
	backTex.addressMode[0] = cudaAddressModeWrap;
	backTex.addressMode[1] = cudaAddressModeWrap;
	backTex.filterMode = cudaFilterModeLinear;
	backTex.normalized = true;

	octree.addressMode[0] = cudaAddressModeWrap;
	octree.filterMode = cudaFilterModePoint;
	octree.normalized = false;
}

__global__ void OctreeConstructKernel(Node* d_node, Voxel* d_voxel, Voxel* d_nextVoxel, int* d_ptr, int* d_nextPtr) {
	//__shared__ Voxel voxels[8 * 256];
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= 1 << 3 * (curDepth - 1)) return;

	//for (int i = 0; i < 8; i++)
	//	voxels[threadIdx.x * 8 + i] = d_voxel[idx * 8 + i];//copy to shared memory

	glm::vec3 color(0.f), normal(0.f);
	int counter = 0;
	int bitMask = 0;
	Voxel v;
	for (int i = 0; i < 8; i++) {
		//v = voxels[threadIdx.x * 8 + i];
		v = d_voxel[idx * 8 + i];
		if (!v.empty()) {
			glm::vec3 c, n;
			v.GetInfo(c, n);
			color += c, normal += n;
			counter++;
			bitMask |= 1 << i;
		}
	}
	if (counter > 0) {
		size_t arrayIdx = atomicAdd(&nodeCounter, 8);
		for (int i = 0; i < 8; i++) {
			int ptr = (1 << curDepth) == d_Info.Dim ? NULLPTR : d_ptr[idx * 8 + i];
			d_node[arrayIdx + i] = Node(ptr, d_voxel[idx * 8 + i]);
		}
		Voxel voxel;
		voxel.SetInfo(color / float(counter), glm::normalize(normal / float(counter)));
		voxel.n |= bitMask << 24U;
		d_nextVoxel[idx] = voxel;
		d_nextPtr[idx] = arrayIdx;
	}

}

void OctreeConstruction(Node*& d_node, Voxel*& d_voxel)
{
	clock_t t = clock();
	const int MAX_DEPTH = log2(Info.Dim);
	gpuErrchk(cudaMemcpyToSymbol(d_MAX_DEPTH, &MAX_DEPTH, sizeof(int)));
	//size_t NODE_SIZE= ((1 << 3 * (MAX_DEPTH + 1)) - 1) / 7 * sizeof(Node);
	size_t NODE_SIZE = 5e6 * sizeof(Node);

	gpuErrchk(cudaMalloc((void**)&d_node, NODE_SIZE));
	gpuErrchk(cudaMemset(d_node, 0, NODE_SIZE));
	Voxel* d_nextVoxel; int* d_ptr, *d_nextPtr; //lower level voxel
	int h_nodeCounter = 0;
	gpuErrchk(cudaMemcpyToSymbol(nodeCounter, &h_nodeCounter, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(d_Info, &Info, sizeof(VoxelizationInfo)));

	for (int i = MAX_DEPTH; i > 0; i--) {
		gpuErrchk(cudaMemcpyToSymbol(curDepth, &i, sizeof(int)));
		size_t nextSize = 1 << 3 * (i - 1);
		gpuErrchk(cudaMalloc((void**)&d_nextVoxel, nextSize * sizeof(Voxel)));
		gpuErrchk(cudaMemset(d_nextVoxel, 0, nextSize * sizeof(Voxel)));
		gpuErrchk(cudaMalloc((void**)&d_nextPtr, nextSize * sizeof(int)));
		gpuErrchk(cudaMemset(d_nextPtr, 0, nextSize * sizeof(int)));

		dim3 blockDim = 256, gridDim = (1 << 3 * i) / 8 / blockDim.x + 1;
		OctreeConstructKernel << <gridDim, blockDim>> > (d_node, d_voxel, d_nextVoxel, d_ptr, d_nextPtr);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemcpyFromSymbol(&h_nodeCounter, nodeCounter, sizeof(int)));
		gpuErrchk(cudaFree(d_voxel));
		if(i != MAX_DEPTH)
			gpuErrchk(cudaFree(d_ptr));
		d_voxel = d_nextVoxel;
		d_ptr = d_nextPtr;
	}

	gpuErrchk(cudaMemcpy(&root.ptr, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&root.voxel, d_voxel, sizeof(Voxel), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_voxel));
	gpuErrchk(cudaFree(d_ptr));
	gpuErrchk(cudaMemcpyFromSymbol(&h_nodeCounter, nodeCounter, sizeof(int)));
	t = clock() - t;
	printf("Octree Construction Complete, %i total nodes in %f sec\n", h_nodeCounter, (float)t / CLOCKS_PER_SEC);
}

__device__ inline int hasNode(Node* d_node, Node root, int& arrayIdx) {
	int offset = arrayIdx;
	int rootPos, i;
	for (i = 0; i < curDepth - 1; i++) {
		int s = 1 << 3 * (curDepth - i - 1);//how many leaf nodes each node has.
		offset = arrayIdx / s;
		if (root.ptr != NULLPTR && root.hasVoxel(offset)) {
			rootPos = root.ptr + offset;
			root = d_node[root.ptr + offset];
		}
		else
			break;
		arrayIdx -= offset * s;
	}
	arrayIdx = rootPos;
	return i;
}
__global__ void OctreeUpdateKernel(Node* d_node, Node root, Voxel* d_voxel, Voxel* d_nextVoxel, int* d_ptr, int* d_nextPtr) {
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= 1 << 3 * (curDepth - 1)) return;

	glm::vec3 color(0.f), normal(0.f);
	int counter = 0;
	int bitMask = 0;
	Voxel v;
	int staticParLevel, staticParPos = idx * 8;
	staticParLevel = hasNode(d_node, root, staticParPos);
	for (int i = 0; i < 8; i++) {
		v = d_voxel[idx * 8 + i];
		if (!v.empty()) {
			glm::vec3 c, n;
			v.GetInfo(c, n);
			color += c, normal += n;
			counter++;
			bitMask |= 1 << i;
		}
	}
	if (counter > 0) {
		if (staticParLevel == curDepth - 1 && d_node[staticParPos].ptr != NULLPTR) {
			Node sRoot = d_node[staticParPos];
			for (int i = 0; i < 8; i++) {
				if (d_node[sRoot.ptr + i].voxel.empty()) {
					int ptr = (1 << curDepth) == d_Info.Dim ? NULLPTR : d_ptr[idx * 8 + i];
					d_node[sRoot.ptr + i] = Node(ptr, d_voxel[idx * 8 + i]);
				}
			}
			sRoot.voxel.n |= bitMask << 24U;
			d_node[staticParPos] = sRoot;
		}
		else {
			size_t arrayIdx = atomicAdd(&dynamicNodeCounter, 8) + nodeCounter;
			for (int i = 0; i < 8; i++) {
				int ptr = (1 << curDepth) == d_Info.Dim ? NULLPTR : d_ptr[idx * 8 + i];
				d_node[arrayIdx + i] = Node(ptr, d_voxel[idx * 8 + i]);
			}
			Voxel voxel;
			voxel.SetInfo(color / float(counter), glm::normalize(normal / float(counter)));
			voxel.n |= bitMask << 24U;
			d_nextVoxel[idx] = voxel;
			d_nextPtr[idx] = arrayIdx;
		}
	}
}

void OctreeUpdate(Node*& d_node, Voxel*& d_voxel)
{
	clock_t t = clock();
	const int MAX_DEPTH = log2(Info.Dim);

	Voxel* d_nextVoxel; int* d_ptr, * d_nextPtr; //lower level voxel
	int h_nodeCounter = 0;
	gpuErrchk(cudaMemcpyToSymbol(d_Info, &Info, sizeof(VoxelizationInfo)));
	gpuErrchk(cudaMemcpyToSymbol(dynamicNodeCounter, &h_nodeCounter, sizeof(int)));

	for (int i = MAX_DEPTH; i > 0; i--) {
		gpuErrchk(cudaMemcpyToSymbol(curDepth, &i, sizeof(int)));
		size_t nextSize = 1 << 3 * (i - 1);
		gpuErrchk(cudaMalloc((void**)&d_nextVoxel, nextSize * sizeof(Voxel)));
		gpuErrchk(cudaMemset(d_nextVoxel, 0, nextSize * sizeof(Voxel)));
		gpuErrchk(cudaMalloc((void**)&d_nextPtr, nextSize * sizeof(int)));
		gpuErrchk(cudaMemset(d_nextPtr, 0, nextSize * sizeof(int)));

		dim3 blockDim = 256, gridDim = (1 << 3 * i) / 8 / blockDim.x + 1;
		OctreeUpdateKernel << <gridDim, blockDim >> > (d_node, root, d_voxel, d_nextVoxel, d_ptr, d_nextPtr);
		gpuErrchk(cudaGetLastError());
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemcpyFromSymbol(&h_nodeCounter, dynamicNodeCounter, sizeof(int)));
		gpuErrchk(cudaFree(d_voxel));
		if (i != MAX_DEPTH)
			gpuErrchk(cudaFree(d_ptr));
		d_voxel = d_nextVoxel;
		d_ptr = d_nextPtr;
	}

	//gpuErrchk(cudaMemcpy(&root.ptr, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(&root.voxel, d_voxel, sizeof(Voxel), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_voxel));
	gpuErrchk(cudaFree(d_ptr));
	gpuErrchk(cudaMemcpyFromSymbol(&h_nodeCounter, dynamicNodeCounter, sizeof(int)));

	t = clock() - t;
	printf("Octree Update Complete, %i total nodes in %f sec\n", h_nodeCounter, (float)t / CLOCKS_PER_SEC);
}



struct Ray {
	glm::vec3 o, d, invD;
	uint depth; bool inside;
	__device__ Ray(glm::vec3 origin, glm::vec3 dir, uint Depth = 0, bool Inside = false) 
		: o(origin), d(dir), depth(Depth), inside(Inside) {
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
		return (tmin < tmax) && tmax > 0.f;
	}
};
struct HitInfo {
	glm::uvec3 idx;
	float t;
	__device__ HitInfo() {};
	__device__ HitInfo(glm::uvec3 _idx, float _t) : idx(_idx), t(_t) {};
};

__device__ Voxel OctreeTraverse(Node* d_node, Node root, Ray ray, glm::vec3 minAABB, uint currentLevel, float& t) {
	//atomicAdd(&traverseCounter, 1U);
	if (currentLevel == MIPMAP)
		return root.voxel;
	if (root.ptr == NULLPTR)
		return Voxel();
	
	currentLevel++;
	HitInfo hits[8];
	int counter = 0;
	float temp = 999.f;

	const float delta = d_Info.delta / float((1 << (currentLevel)));
	for (int i = 0; i < 8; i++) {
		glm::uvec3 idx(i & 1, (i >> 1) & 1, i >> 2);
		glm::vec3 _minAABB = minAABB + glm::vec3(idx) * delta;
		float _t;
		if (root.hasVoxel(i) && ray.RayAABBIntersection(_minAABB, _minAABB + delta, _t)) {// !d_node[root.ptr + i].voxel.empty()
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
			glm::vec3 c;
			glm::vec3 n;
			voxel.GetInfo(c, n);

			if (currentLevel == MIPMAP)
				t = hits[i].t;
			return voxel;
		}
	}
	return Voxel();
}

__global__ void RayCastKernel(uint* d_pbo, Node* d_node, const uint w, const uint h, Node root) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x,
			y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= w || y >= h) return;
	d_pbo[y * w + x] = 0;
	const float u = float(x) / float(w), v = float(y) / float(h);
	float4 backSample = tex2D(backTex, u, v);
		
	glm::vec3 dir = glm::normalize(glm::vec3(backSample.x, backSample.y, backSample.z));
	Ray ray(d_Info.camPos, dir, 0);

	glm::vec3 color(0.f);
	float t;
	if (ray.RayAABBIntersection(d_Info.minAABB, d_Info.minAABB + d_Info.delta, t)) {
		t = 999.f;
		Voxel voxel = OctreeTraverse(d_node, root, ray, d_Info.minAABB, 0, t);
		glm::vec3 pos = ray.o + t * ray.d;
		if(!voxel.empty())
			color = voxel.PhongLighting(pos);
		else {
			float4 texel = texCubemap(skyBox, dir.x, dir.y, dir.z);
			color = glm::vec4(texel.x, texel.y, texel.z, 1.f);
		}

	}
	else {
		float4 texel = texCubemap(skyBox, dir.x, dir.y, dir.z);
		color = glm::vec3(texel.x, texel.y, texel.z);
	}
	//Gamma Correction
	color = glm::pow(color, glm::vec3(1.f / 2.2f));
	d_pbo[y * w + x] = ConvVec4ToUint(glm::vec4(color, 1.f));

}


void RayCastingOctree(uint* d_pbo, glm::vec3 h_camPos, cudaArray_t back, Node* d_node)
{
	gpuErrchk(cudaMemcpyToSymbol(d_Info, &Info, sizeof(VoxelizationInfo)));

	cudaMemcpyToSymbol(MIPMAP, &h_MIPMAP, 4);
	uint h_tCounter = 0;
	cudaMemcpyToSymbol(traverseCounter, &h_tCounter, 4);

	cudaChannelFormatDesc format = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	gpuErrchk(cudaBindTextureToArray(&backTex, back, &format));

	//launch cuda kernel
	dim3 blockDim(16, 16, 1), gridDim(WINDOW_WIDTH / blockDim.x + 1, WINDOW_HEIGHT / blockDim.y + 1, 1);
	RayCastKernel << <gridDim, blockDim >> > (d_pbo, d_node, WINDOW_WIDTH, WINDOW_HEIGHT, root);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	cudaMemcpyFromSymbol(&h_tCounter, traverseCounter, 4);
	//printf("Traverse Count: %u\n", h_tCounter);
	gpuErrchk(cudaUnbindTexture(backTex));
}

void initSkyBox() {
	std::string faces[6]{
		"asset/texture/skybox/posx.jpg",
		"asset/texture/skybox/negx.jpg",
		"asset/texture/skybox/posy.jpg",
		"asset/texture/skybox/negy.jpg",
		"asset/texture/skybox/posz.jpg",
		"asset/texture/skybox/negz.jpg"
	};
	//Read Image
	int w, h, n, num_faces = 6;
	stbi_loadf(faces[0].c_str(), &w, &h, &n, 4);
	size_t face_size = w * h * 4;
	float* h_data = new float[face_size * num_faces];
	for (int i = 0; i < num_faces; i++) {
		float* image = stbi_loadf(faces[i].c_str(), &w, &h, &n, 4);
		memcpy(h_data + face_size * i, image, face_size * sizeof(float));
		delete image;
	}

	//Cuda Malloc
	cudaChannelFormatDesc format = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaArray_t cu3dArr;
	gpuErrchk(cudaMalloc3DArray(&cu3dArr, &format, make_cudaExtent(w, h, num_faces), cudaArrayCubemap));
	cudaMemcpy3DParms myparms = { 0 };
	myparms.srcPos = make_cudaPos(0, 0, 0);
	myparms.dstPos = make_cudaPos(0, 0, 0);
	myparms.srcPtr = make_cudaPitchedPtr(h_data, w * 4 * sizeof(float), w, h);
	myparms.dstArray = cu3dArr;
	myparms.extent = make_cudaExtent(w, h, num_faces);
	myparms.kind = cudaMemcpyHostToDevice;
	gpuErrchk(cudaMemcpy3D(&myparms));

	//Init texture
	skyBox.addressMode[0] = cudaAddressModeWrap;
	skyBox.addressMode[1] = cudaAddressModeWrap;
	skyBox.addressMode[2] = cudaAddressModeWrap;
	skyBox.filterMode = cudaFilterModeLinear;
	skyBox.normalized = true;
	gpuErrchk(cudaBindTextureToArray(&skyBox, cu3dArr, &format));

	delete h_data;
}

void initRayCasting()
{
	initCudaTexture();
	initSkyBox();
	//cudaChannelFormatDesc format = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindUnsigned);
	//size_t octree_size = size_t(h_curIdx) * sizeof(Node);
	//size_t offset = 0;
	//cudaStatus = cudaBindTexture(&offset, &octree, d_node, &format, octree_size);
	//if (cudaStatus != cudaSuccess) printf("cudaBindTexture Failed, error: %s\n", cudaGetErrorString(cudaStatus));
	gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 16));

}






