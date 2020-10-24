#pragma once
#include "Voxel.cuh"
class Node {
public:
	uint ptr;
	Voxel voxel;
	__host__ __device__ Node();
	__host__ __device__ ~Node();
};

void OctreeConstruction(Node*& d_node, Voxel*& d_voxel, uint* d_idx);
void RayCastingOctree(uint* d_pbo, cudaArray_t front, cudaArray_t back,  Node* d_node);

