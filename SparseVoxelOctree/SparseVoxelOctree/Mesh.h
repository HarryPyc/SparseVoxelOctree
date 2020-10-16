#pragma once
#include <ObjLoad.h>
#include <string>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct CudaMesh {
	float* d_v, * d_n;
	unsigned short* d_idx;
	unsigned int vSize, nSize, idxSize;
	glm::vec3 minAABB, maxAABB;
	CudaMesh();
	~CudaMesh();
};
class Mesh
{
public:
	obj::Model data;

	Mesh(const std::string& path);
	~Mesh() {}

	void UploatToDevice(CudaMesh cuMesh);
};

