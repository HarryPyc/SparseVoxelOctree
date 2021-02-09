#include "Scene.h"
#include <glm/gtx/transform.hpp>
CudaMesh cuStatic, cuDynamic;
Scene::Scene() {
	static_mesh = new Mesh("asset/model/test_scene.obj");
	dynamic_mesh = new Mesh("asset/model/dragon.obj");
}

Scene::~Scene() {
	delete static_mesh;
	delete dynamic_mesh;
}

void Scene::Upload() {
	cuStatic.color = glm::vec3(0.75f);
	cuStatic.M = glm::translate(glm::vec3(0));
	static_mesh->UploatToDevice(cuStatic);

	cuDynamic.color = glm::vec3(0.75f, 0.25f, 0.25f);
	cuDynamic.M = glm::rotate(glm::radians(90.f), glm::vec3(0.f, 1.f, 0.f));
	dynamic_mesh->UploatToDevice(cuDynamic);
	//cuDynamic.M = glm::translate(glm::vec3(0.f, 0.f, 0.3f));
}

void Scene::SceneVoxelization(Voxel*& d_voxel, uint*& d_idx)
{
	InitVoxelization(d_voxel, d_idx);
	Voxelization(cuStatic, d_voxel, d_idx);
	Voxelization(cuDynamic, d_voxel, d_idx);
}
