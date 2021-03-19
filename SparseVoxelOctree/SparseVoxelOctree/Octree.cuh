#pragma once
#include "Voxel.cuh"
class Node {
public:
	uint ptr;
	Voxel voxel;
	__host__ __device__ Node();
	__host__ __device__ Node(uint _ptr, Voxel _vox);
	__host__ __device__ ~Node();
	__device__ inline bool hasVoxel(uint i) {
		//printf("hasLeaf: %u\n", voxel.c >> 24U);
		return voxel.n >> 24U >> i & 1;
	}
};


void OctreeConstruction(Node*& d_node, Voxel*& d_voxel);
void RayCastingOctree(uint* d_pbo, cudaArray_t back,  Node* d_node);
void VoxelConeTracing(float* d_pbo, cudaArray_t posArray, cudaArray_t normalArray, Node* d_node);
void initRayCasting();
void OctreeUpdate(Node*& d_node, Voxel*& d_voxel);