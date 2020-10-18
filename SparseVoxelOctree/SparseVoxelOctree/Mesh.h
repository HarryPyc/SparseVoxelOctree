#pragma once
#include <ObjLoad.h>
#include <string>
#include <GL/glew.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct Triangle {
	unsigned short i0, i1, i2;
	glm::vec3 n;
	glm::vec2 ne_xy[3], ne_xz[3], ne_yz[3];
	float d1, d2, de_xy[3], de_xz[3], de_yz[3];
};
struct CudaMesh {
	float* d_v, * d_n;
	unsigned short* d_idx;
	Triangle* d_tri;
	unsigned short triNum;
	//AABB
	glm::vec3 minAABB;
	float delta; //maxAABB = minAABB + delta
};
class Mesh
{
public:
	obj::Model data;
	//transform matrix
	glm::mat4 M;

	Mesh(const std::string& path);
	~Mesh() {}

	void UploatToDevice(CudaMesh& cuMesh);
	void CreateVao();
	void Draw();
private:
	GLuint vao;

};

