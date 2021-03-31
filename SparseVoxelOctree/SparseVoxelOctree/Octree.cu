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
Node root, sRoot, dRoot;
int sNodeCounter, dNodeCounter;
__device__ int nodeCounter, dynamicNodeCounter, traverseCounter;

texture<float4, 2, cudaReadModeElementType> backTex, posTex, normalTex;
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
	//backTex.addressMode[0] = cudaAddressModeWrap;
	//backTex.addressMode[1] = cudaAddressModeWrap;
	//backTex.filterMode = cudaFilterModeLinear;
	//backTex.normalized = true;

	posTex.addressMode[0] = cudaAddressModeWrap;
	posTex.addressMode[1] = cudaAddressModeWrap;
	posTex.filterMode = cudaFilterModeLinear;
	posTex.normalized = true;

	normalTex.addressMode[0] = cudaAddressModeWrap;
	normalTex.addressMode[1] = cudaAddressModeWrap;
	normalTex.filterMode = cudaFilterModeLinear;
	normalTex.normalized = true;
}

__global__ void OctreeConstructKernel(Node* d_node, Voxel* d_voxel, Voxel* d_nextVoxel, int* d_ptr, int* d_nextPtr) {
	//__shared__ Voxel voxels[8 * 256];
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= 1 << 3 * (curDepth - 1)) return;

	//for (int i = 0; i < 8; i++)
	//	voxels[threadIdx.x * 8 + i] = d_voxel[idx * 8 + i];//copy to shared memory

	glm::vec3 normal(0.f);
	float color = 0.f;
	int counter = 0;
	int bitMask = 0;
	Voxel v;
	for (int i = 0; i < 8; i++) {
		//v = voxels[threadIdx.x * 8 + i];
		v = d_voxel[idx * 8 + i];
		if (!v.empty()) {
			glm::vec3 n;
			float c;
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
		voxel.SetInfo(color / 8.f, glm::normalize(normal / float(counter)));
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
	size_t NODE_SIZE= ((1 << 3 * (MAX_DEPTH + 1)) - 1) / 7 * sizeof(Node);
	//size_t NODE_SIZE = 5e6 * sizeof(Node);

	gpuErrchk(cudaMalloc((void**)&d_node, NODE_SIZE));
	gpuErrchk(cudaMemset(d_node, 0, NODE_SIZE));
	Voxel* d_nextVoxel; int* d_ptr, *d_nextPtr; //lower level voxel
	sNodeCounter = 0;
	gpuErrchk(cudaMemcpyToSymbol(nodeCounter, &sNodeCounter, sizeof(int)));
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

		gpuErrchk(cudaFree(d_voxel));
		if(i != MAX_DEPTH)
			gpuErrchk(cudaFree(d_ptr));
		d_voxel = d_nextVoxel;
		d_ptr = d_nextPtr;
	}

	gpuErrchk(cudaMemcpy(&sRoot.ptr, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&sRoot.voxel, d_voxel, sizeof(Voxel), cudaMemcpyDeviceToHost));

	root = sRoot;

	gpuErrchk(cudaFree(d_voxel));
	gpuErrchk(cudaFree(d_ptr));
	gpuErrchk(cudaMemcpyFromSymbol(&sNodeCounter, nodeCounter, sizeof(int)));
	t = clock() - t;
#ifdef PRINT_INFO
	printf("Octree Construction Complete, %i total nodes in %f sec\n", sNodeCounter, (float)t / CLOCKS_PER_SEC);
#endif // PRINT_INFO

}

__device__ inline int hasNode(Node* d_node, Node root, int& arrayIdx) {
	int offset = arrayIdx;
	int rootPos, i = 0;
	for (; i < curDepth - 1; i++) {
		int s = 1 << 3 * (curDepth - i - 1);//how many leaf nodes each node has.
		offset = arrayIdx / s;
		if (root.ptr != NULLPTR && root.hasVoxel(offset)) {
			rootPos = root.ptr + offset;
			root = d_node[rootPos];
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

	glm::vec3 normal(0.f);
	float color = 0.f;
	int counter = 0;
	int bitMask = 0;
	Voxel v;
	int staticParLevel, staticParPos = idx * 8;
	staticParLevel = hasNode(d_node, root, staticParPos);
	for (int i = 0; i < 8; i++) {
		v = d_voxel[idx * 8 + i];
		if (!v.empty()) {
			glm::vec3  n;
			float c;
			v.GetInfo(c, n);
			color += c, normal += n;
			counter++;
			bitMask |= 1 << i;
		}
	}
	if (counter > 0) {
		if (staticParLevel == curDepth - 1) {
			Node _root = curDepth == 1? root : d_node[staticParPos];
			for (int i = 0; i < 8; i++) {
				if (_root.hasVoxel(i) && d_voxel[idx * 8 + i].empty()) {
					d_voxel[idx * 8 + i] = d_node[_root.ptr + i].voxel;
					if(curDepth != d_MAX_DEPTH)
						d_ptr[idx * 8 + i] = d_node[_root.ptr + i].ptr;
					bitMask |= 1 << i;
				}
			}
		}

		size_t arrayIdx = atomicAdd(&dynamicNodeCounter, 8) + nodeCounter;
		for (int i = 0; i < 8; i++) {
			int ptr = (1 << curDepth) == d_Info.Dim ? NULLPTR : d_ptr[idx * 8 + i];
			d_node[arrayIdx + i] = Node(ptr, d_voxel[idx * 8 + i]);
		}
		Voxel voxel;
		voxel.SetInfo(color / 8.f, glm::normalize(normal / float(counter)));
		voxel.n |= bitMask << 24U;
		d_nextVoxel[idx] = voxel;
		d_nextPtr[idx] = arrayIdx;
	}
}

void OctreeUpdate(Node*& d_node, Voxel*& d_voxel)
{
	clock_t t = clock();
	gpuErrchk(cudaMemset(d_node + sNodeCounter, 0, dNodeCounter * sizeof(Node)));//Clear Dynamic Part
	const int MAX_DEPTH = log2(Info.Dim);
	root = sRoot;

	Voxel* d_nextVoxel; int* d_ptr, * d_nextPtr; //lower level voxel
	gpuErrchk(cudaMemcpyToSymbol(d_Info, &Info, sizeof(VoxelizationInfo)));
	dNodeCounter = 0;
	gpuErrchk(cudaMemcpyToSymbol(dynamicNodeCounter, &dNodeCounter, sizeof(int)));

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

		gpuErrchk(cudaFree(d_voxel));
		if (i != MAX_DEPTH)
			gpuErrchk(cudaFree(d_ptr));

		d_voxel = d_nextVoxel;
		d_ptr = d_nextPtr;
	}

	gpuErrchk(cudaMemcpy(&dRoot.ptr, d_ptr, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&dRoot.voxel, d_voxel, sizeof(Voxel), cudaMemcpyDeviceToHost));

	root = dRoot;

	gpuErrchk(cudaFree(d_voxel));
	gpuErrchk(cudaFree(d_ptr));
	gpuErrchk(cudaMemcpyFromSymbol(&dNodeCounter, dynamicNodeCounter, sizeof(int)));

	t = clock() - t;
#ifdef PRINT_INFO
	printf("Octree Update Complete, %i total nodes in %f sec\n\n", dNodeCounter, (float)t / CLOCKS_PER_SEC);
#endif // PRINT_INFO

}


struct Ray {
	glm::vec3 o, d, invD;
	float tanT;
	__device__ Ray(glm::vec3 origin, glm::vec3 dir, float theta) 
		: o(origin), d(dir){
		invD = glm::vec3(1.f) / dir;
		tanT = glm::tan(glm::radians(theta));
	}
	__device__ ~Ray() {};
	__device__ inline bool Reach(float t, int curLevel) {
		float curDelta = d_Info.delta / float(1 << curLevel);
		float diam = 2.f * t * tanT;
		return diam >= curDelta && diam < curDelta * 2.f;
	}
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

__device__ float OctreeTraverse(Node* d_node, Node &root, Ray &ray, glm::vec3 &minAABB, uint currentLevel, float t) {
	//atomicAdd(&traverseCounter, 1U);
	if (ray.Reach(t, currentLevel))
		return root.voxel.c;
	if (currentLevel == MIPMAP)
		return root.voxel.c;
	if (root.ptr == NULLPTR)
		return 0.f;
	
	currentLevel++;
	HitInfo hits[8];
	int counter = 0;

	const float delta = d_Info.delta / float((1 << currentLevel));
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
		float ret = OctreeTraverse(d_node, _root, ray, _minAABB, currentLevel,  hits[i].t);
		if (ret != 0.f) {
			//float f = glm::length(_minAABB + delta / 2.f - (ray.o + ray.d * hits[i].t)) / (1.4142f * delta / 4.f);
			return ret ;
		}
	}
	return 0.f;
}

//__device__ float OctreeTraverse(Node* d_node, Node& root, Ray &ray, glm::vec3 &minAABB, uint currentLevel, float t) {
//	//atomicAdd(&traverseCounter, 1U);
//	if (ray.Reach(t, currentLevel))
//		return root.voxel.c;
//	if (currentLevel == MIPMAP)
//		return root.voxel.c;
//	if (root.ptr == NULLPTR)
//		return 0.f;
//
//	currentLevel++;
//	float sum = 0.f;
//
//	const float delta = d_Info.delta / float((1 << (currentLevel)));
//	for (int i = 0; i < 8; i++) {
//		glm::uvec3 idx(i & 1, (i >> 1) & 1, i >> 2);
//		glm::vec3 _minAABB = minAABB + glm::vec3(idx) * delta;
//		float _t;
//		if (root.hasVoxel(i) && ray.RayAABBIntersection(_minAABB, _minAABB + delta, _t)) {// !d_node[root.ptr + i].voxel.empty()
//			sum += OctreeTraverse(d_node, d_node[root.ptr + i], ray, _minAABB, currentLevel, _t) / (1.f + 0.5f * _t);
//		}
//	}
//
//	return sum;
//}

//__global__ void RayCastKernel(uint* d_pbo, Node* d_node, const uint w, const uint h, Node root) {
//	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x,
//			y = blockDim.y * blockIdx.y + threadIdx.y;
//	if (x >= w || y >= h) return;
//	d_pbo[y * w + x] = 0;
//	const float u = float(x) / float(w), v = float(y) / float(h);
//	float4 backSample = tex2D(backTex, u, v);
//		
//	glm::vec3 dir = glm::normalize(glm::vec3(backSample.x, backSample.y, backSample.z));
//	Ray ray(d_Info.camPos, dir, 0);
//
//	glm::vec3 color(0.f);
//	float t;
//	if (ray.RayAABBIntersection(d_Info.minAABB, d_Info.minAABB + d_Info.delta, t)) {
//		t = 999.f;
//		Voxel voxel = OctreeTraverse(d_node, root, ray, d_Info.minAABB, 0, t);
//		glm::vec3 pos = ray.o + t * ray.d;
//		if(!voxel.empty())
//			color = voxel.PhongLighting(pos);
//		else {
//			float4 texel = texCubemap(skyBox, dir.x, dir.y, dir.z);
//			color = glm::vec4(texel.x, texel.y, texel.z, 1.f);
//		}
//
//	}
//	else {
//		float4 texel = texCubemap(skyBox, dir.x, dir.y, dir.z);
//		color = glm::vec3(texel.x, texel.y, texel.z);
//	}
//	//Gamma Correction
//	color = glm::pow(color, glm::vec3(1.f / 2.2f));
//	d_pbo[y * w + x] = ConvVec4ToUint(glm::vec4(color, 1.f));
//
//}
//
//
//void RayCastingOctree(uint* d_pbo, cudaArray_t back, Node* d_node)
//{
//	gpuErrchk(cudaMemcpyToSymbol(d_Info, &Info, sizeof(VoxelizationInfo)));
//
//	cudaMemcpyToSymbol(MIPMAP, &h_MIPMAP, 4);
//	uint h_tCounter = 0;
//	cudaMemcpyToSymbol(traverseCounter, &h_tCounter, 4);
//
//	cudaChannelFormatDesc format = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
//
//	gpuErrchk(cudaBindTextureToArray(&backTex, back, &format));
//
//	//launch cuda kernel
//	dim3 blockDim(16, 16, 1), gridDim(WINDOW_WIDTH / blockDim.x + 1, WINDOW_HEIGHT / blockDim.y + 1, 1);
//	RayCastKernel << <gridDim, blockDim >> > (d_pbo, d_node, WINDOW_WIDTH, WINDOW_HEIGHT, root);
//	gpuErrchk(cudaGetLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//
//	cudaMemcpyFromSymbol(&h_tCounter, traverseCounter, 4);
//	//printf("Traverse Count: %u\n", h_tCounter);
//	gpuErrchk(cudaUnbindTexture(backTex));
//}


__global__ void ConeTracingKernel(float* d_pbo, Node* d_node, const uint w, const uint h, Node root) {
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x,
		y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= w || y >= h) return;
	const unsigned long long idx = (unsigned long long)y * w + x;
	const float u = float(x) / float(w), v = float(y) / float(h);
	
	float4 posSample = tex2D(posTex, u, v), normalSample = tex2D(normalTex, u, v);
	glm::vec3 pos, normal;
	memcpy(&pos[0], &posSample, 3 * sizeof(float));
	memcpy(&normal[0], &normalSample, 3 * sizeof(float));
	pos += d_Info.delta * normal / float(1 << d_MAX_DEPTH - 1);
	glm::vec3 nu = glm::normalize(glm::cross(normal, glm::vec3(1.f))), nv = glm::cross(normal, nu);
	const float sin30 = 0.5f, cos30 = sqrt(3.f) / 2.f;

	glm::vec3 visibility(0.f);
	float t = 1e5f;
	visibility += OctreeTraverse(d_node, root, Ray(pos, normal, 30.f), d_Info.minAABB, 0, t);
	visibility += OctreeTraverse(d_node, root, Ray(pos, normal * sin30 + nu * cos30, 30.f), d_Info.minAABB, 0, t);
	visibility += OctreeTraverse(d_node, root, Ray(pos, normal * sin30 + -nu * cos30, 30.f), d_Info.minAABB, 0, t);
	visibility += OctreeTraverse(d_node, root, Ray(pos, normal * sin30 + nv * cos30, 30.f), d_Info.minAABB, 0, t);
	visibility += OctreeTraverse(d_node, root, Ray(pos, normal * sin30 + -nv * cos30, 30.f), d_Info.minAABB, 0, t);


	visibility = 1.0f - visibility / 5.f;
	memcpy(d_pbo + 4 * idx, &visibility[0], 3 * sizeof(float));
}

void VoxelConeTracing(float* d_pbo, cudaArray_t posArray, cudaArray_t normalArray, Node* d_node)
{
	gpuErrchk(cudaMemcpyToSymbol(d_Info, &Info, sizeof(VoxelizationInfo)));
	cudaMemcpyToSymbol(MIPMAP, &h_MIPMAP, 4);

	cudaChannelFormatDesc format = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	gpuErrchk(cudaBindTextureToArray(&posTex, posArray, &format));
	gpuErrchk(cudaBindTextureToArray(&normalTex, normalArray, &format));

	dim3 blockDim(16, 16, 1), gridDim(WINDOW_WIDTH / blockDim.x + 1, WINDOW_HEIGHT / blockDim.y + 1, 1);
	ConeTracingKernel << <gridDim, blockDim >> > (d_pbo, d_node, WINDOW_WIDTH, WINDOW_HEIGHT, root);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaUnbindTexture(posTex));
	gpuErrchk(cudaUnbindTexture(normalTex));
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
	//initSkyBox();

	gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 16));

}






