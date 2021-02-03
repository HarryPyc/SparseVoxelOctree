#include "Scene.h"
#include <glm/gtx/transform.hpp>
CudaMesh cuStatic, cuDynamic;
Scene::Scene() {
	static_mesh = new Mesh("asset/model/static_box.obj");
	dynamic_mesh = new Mesh("asset/model/dragon.obj");
}

Scene::~Scene() {
	delete static_mesh;
	delete dynamic_mesh;
}

void Scene::Upload() {
	static_mesh->UploatToDevice(cuStatic);
	cuStatic.color = glm::vec3(0.75f);
	cuStatic.M = glm::translate(glm::vec3(0));
	dynamic_mesh->UploatToDevice(cuDynamic);
	cuDynamic.color = glm::vec3(0.75f, 0.25f, 0.25f);
	cuDynamic.M = glm::rotate(glm::radians(90.f), glm::vec3(0.f, 1.f, 0.f));
	//cuDynamic.M = glm::translate(glm::vec3(0.f, 0.f, 0.3f));
}

void Scene::SceneVoxelization(Voxel*& d_voxel, uint*& d_idx)
{
	Voxelization(cuStatic, d_voxel, d_idx);
	Voxelization(cuDynamic, d_voxel, d_idx);
}
