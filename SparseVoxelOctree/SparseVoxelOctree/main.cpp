
#include "cuda_runtime.h"

#include <stdio.h>
#include "Mesh.h"
#include "Voxel.cuh"

Voxel* d_voxel = NULL;

int main() {
	/*if (cudaSetDevice(0) != cudaSuccess) {
		printf("cudaSetDevice Failed");
		return 0;
	}*/

	Mesh mesh("asset/model/cube.obj");
	Voxelization(mesh, d_voxel);

	cudaFree(d_voxel);
	return 0;
}
