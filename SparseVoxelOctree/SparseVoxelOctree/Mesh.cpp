#include "Mesh.h"
#include "cuda_runtime.h"
Mesh::Mesh(const std::string& path)
{
	data = obj::loadModelFromFile(path);
}

void Mesh::UploatToDevice(CudaMesh &cuMesh)
{
	cuMesh.triNum = data.faces["default"].size() / 3;
	//Reconstruct AABB
	glm::vec3 _minAABB = glm::vec3(data.min[0], data.min[1], data.min[2]);
	glm::vec3 _maxAABB = glm::vec3(data.max[0], data.max[1], data.max[2]);
	glm::vec3 l = _maxAABB - _minAABB;
	cuMesh.delta = glm::max(l.x, glm::max(l.y, l.z));
	cuMesh.minAABB = (_minAABB + _maxAABB) / 2.f - cuMesh.delta / 2.f;

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

void Mesh::CreateVao()
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
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * data.faces["default"].size(), data.faces["default"].data(), GL_STATIC_DRAW);

	const GLuint pos_loc = 0;

	glEnableVertexAttribArray(pos_loc);

	glVertexAttribPointer(pos_loc, 3, GL_FLOAT, false, 0, 0);
	glBindVertexArray(0);

}

void Mesh::Draw()
{
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, data.faces["default"].size(), GL_UNSIGNED_SHORT, 0);
}




