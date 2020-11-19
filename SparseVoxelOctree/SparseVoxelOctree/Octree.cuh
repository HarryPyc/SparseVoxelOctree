#pragma once
#include "Voxel.cuh"
class Node {
public:
	uint ptr;
	Voxel voxel;
	__host__ __device__ Node();
	__host__ __device__ Node(uint4 u);
	__host__ __device__ ~Node();
	__device__ inline bool hasVoxel(uint i) {
		//printf("hasLeaf: %u\n", voxel.c >> 24U);
		return (voxel.n >> 24U & (1U << i)) >> i;
	}
};

const uint MAX_TRACE_DEPTH = 8;

void OctreeConstruction(Node*& d_node, Voxel*& d_voxel, uint* d_idx);
void RayCastingOctree(uint* d_pbo, glm::vec3 camPos, cudaArray_t back,  Node* d_node);
void initRayCasting();
