#include "Mesh.h"
#include "cuda_runtime.h"
#include "Voxel.cuh"
Mesh::Mesh(const std::string& path)
{
	data = obj::loadModelFromFile(path);
}

//void Mesh::UploatToDevice(CudaMesh &cuMesh)
//{
//	cuMesh.triNum = data.faces["default"].size() / 3;
//	printf("Mesh Triangle Count: %i\n", cuMesh.triNum);
//	cuMesh.vertNum = data.vertex.size();
//	size_t vert_size = data.vertex.size() * sizeof(float), normal_size = data.normal.size() * sizeof(float),
//		index_size = data.faces["default"].size() * sizeof(unsigned );
//	cudaError_t cudaStatus;
//	//Copy data to device
//	cudaStatus = cudaMalloc((void**)&cuMesh.d_v, vert_size);
//	if (cudaStatus != cudaSuccess) printf("d_v cudaMalloc Failed\n");
//	cudaStatus = cudaMemcpy(cuMesh.d_v, data.vertex.data(), vert_size, cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) printf("d_v cudaMemcpy Failed\n");
//
//	cudaStatus = cudaMalloc((void**)&cuMesh.d_n, normal_size);
//	if (cudaStatus != cudaSuccess) printf("d_n cudaMalloc Failed\n");
//	cudaStatus = cudaMemcpy(cuMesh.d_n, data.normal.data(), normal_size, cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) printf("d_n cudaMemcpy Failed\n");
//
//	cudaStatus = cudaMalloc((void**)&cuMesh.d_idx, index_size);
//	if (cudaStatus != cudaSuccess) printf("d_idx cudaMalloc Failed\n");
//	cudaStatus = cudaMemcpy(cuMesh.d_idx, data.faces["default"].data(), index_size, cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) printf("d_idx cudaMemcpy Failed\n");
//
//	gpuErrchk(cudaMalloc((void**)&cuMesh.d_tri, cuMesh.triNum * sizeof(Triangle)));
//
//	PreProcess(cuMesh);
//}

GLuint Mesh::CreateVao()
{
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * data.vertex.size(), data.vertex.data(), GL_STATIC_DRAW);

	GLuint ebo;
	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned) * data.faces["default"].size(), data.faces["default"].data(), GL_STATIC_DRAW);

	const GLuint pos_loc = 0;

	glEnableVertexAttribArray(pos_loc);

	glVertexAttribPointer(pos_loc, 3, GL_FLOAT, false, 0, 0);
	glBindVertexArray(0);
	return vao;
}

void Mesh::Draw()
{
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, data.faces["default"].size(), GL_UNSIGNED_INT, 0);
}


__host__ void CudaMesh::init(Mesh* mesh)
{
	triNum = mesh->data.faces["default"].size() / 3;
	printf("Mesh Triangle Count: %i\n", triNum);
	vertNum = mesh->data.vertex.size();

	size_t vert_size = mesh->data.vertex.size() * sizeof(float), normal_size = mesh->data.normal.size() * sizeof(float),
				index_size = mesh->data.faces["default"].size() * sizeof(unsigned );
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, vert_size + normal_size, 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, vert_size, mesh->data.vertex.data());
	glBufferSubData(GL_ARRAY_BUFFER, vert_size, normal_size, mesh->data.normal.data());

	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_size, mesh->data.faces["default"].data(), GL_STATIC_DRAW);

	const GLuint pos_loc = 0;
	glEnableVertexAttribArray(pos_loc);
	glVertexAttribPointer(pos_loc, 3, GL_FLOAT, false, 0, 0);

	const GLuint normal_loc = 1;
	glEnableVertexAttribArray(normal_loc);
	glVertexAttribPointer(normal_loc, 3, GL_FLOAT, false, 0, (void*)(vert_size));

	glBindVertexArray(0);

	gpuErrchk(cudaGraphicsGLRegisterBuffer(&resources[0], vbo, cudaGraphicsRegisterFlagsNone));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&resources[1], ebo, cudaGraphicsRegisterFlagsNone));

	gpuErrchk(cudaMalloc((void**)&d_tri, triNum * sizeof(Triangle)));
}

__host__ void CudaMesh::MapResources()
{
	gpuErrchk(cudaGraphicsMapResources(2, resources));

	size_t numBytes;
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_v, &numBytes, resources[0]));
	d_n = d_v + vertNum;
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&d_idx, &numBytes, resources[1]));

}

__host__ void CudaMesh::UnMapResources()
{
	gpuErrchk(cudaGraphicsUnmapResources(2, resources));
}

__host__ void CudaMesh::DrawMesh()
{
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, triNum * 3, GL_UNSIGNED_INT, 0);
}
