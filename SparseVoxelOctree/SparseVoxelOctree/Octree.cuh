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
		return (voxel.n >> 24U & (1U << i)) >> i;
	}
};


void OctreeConstruction(Node*& d_node, Voxel*& d_voxel);
void RayCastingOctree(uint* d_pbo, glm::vec3 camPos, cudaArray_t back,  Node* d_node);
void initRayCasting();
void OctreeUpdate(Node*& d_node, Voxel*& d_voxel);