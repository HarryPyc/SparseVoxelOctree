#include "Mesh.h"
#include "cuda_runtime.h"
Mesh::Mesh(const std::string& path)
{
	data = obj::loadModelFromFile(path);
}

void Mesh::UploatToDevice(CudaMesh cuMesh)
{
	cuMesh.vSize = data.vertex.size(), cuMesh.nSize = data.normal.size(), cuMesh.idxSize = data.faces["default"].size();
	cuMesh.minAABB = glm::vec3(data.min[0], data.min[1], data.min[2]);
	cuMesh.maxAABB = glm::vec3(data.max[0], data.max[1], data.max[2]);
	size_t vert_size = data.vertex.size() * sizeof(float), normal_size = data.normal.size() * sizeof(float),
		index_size = data.faces["default"].size() * sizeof(unsigned short);
	cudaError_t cudaStatus;
	//Copy data to device
	cudaStatus = cudaMalloc((void**)&cuMesh.d_v, vert_size);
	if (cudaStatus != cudaSuccess) printf("d_v cudaMalloc Failed\n");
	cudaStatus = cudaMemcpy(cuMesh.d_v, data.vertex.data(), vert_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) printf("d_v cudaMemcpy Failed\n");

	cudaStatus = cudaMalloc((void**)&cuMesh.d_n, normal_size);
	if (cudaStatus != cudaSuccess) printf("d_n cudaMalloc Failed\n");
	cudaStatus = cudaMemcpy(cuMesh.d_n, data.normal.data(), normal_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) printf("d_n cudaMemcpy Failed\n");

	cudaStatus = cudaMalloc((void**)&cuMesh.d_idx, index_size);
	if (cudaStatus != cudaSuccess) printf("d_idx cudaMalloc Failed\n");
	cudaStatus = cudaMemcpy(cuMesh.d_idx, data.faces["default"].data(), index_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) printf("d_idx cudaMemcpy Failed\n");
}

CudaMesh::CudaMesh()
{
	d_v = NULL, d_n = NULL, d_idx = NULL;
}

CudaMesh::~CudaMesh()
{
	cudaFree(d_v);
	cudaFree(d_n);
	cudaFree(d_idx);
}
