
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Mesh.h"
#include "Voxel.cuh"

int main() {
	/*if (cudaSetDevice(0) != cudaSuccess) {
		printf("cudaSetDevice Failed");
		return 0; 
	}*/
	CudaMesh cuMesh;
	Mesh mesh("asset/model/bunny.obj");
	mesh.UploatToDevice(cuMesh);
	
	return 0;
}
