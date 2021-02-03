#include "Mesh.h"
#include "cuda_runtime.h"
Mesh::Mesh(const std::string& path)
{
	data = obj::loadModelFromFile(path);
}

void Mesh::UploatToDevice(CudaMesh &cuMesh)
{
	cuMesh.triNum = data.faces["default"].size() / 3;
	printf("Mesh Triangle Count: %i\n", cuMesh.triNum);
	cuMesh.vertNum = data.vertex.size();
	size_t vert_size = data.vertex.size() * sizeof(float), normal_size = data.normal.size() * sizeof(float),
		index_size = data.faces["default"].size() * sizeof(unsigned );
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




