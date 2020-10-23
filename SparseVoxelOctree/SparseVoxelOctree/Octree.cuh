#pragma once
#include "Voxel.cuh"
class Node {
public:
	uint ptr, voxelPtr;
	__device__ Node();
	__device__ ~Node();
};

void OctreeConstruction(Node*& d_node, Voxel*& d_voxel, uint* d_idx);
void RayCastingOctree(uint* d_pbo, cudaArray_t front, cudaArray_t back, Voxel* d_voxel, Node* d_node);

