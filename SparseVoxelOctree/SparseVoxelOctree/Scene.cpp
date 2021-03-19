#include "Scene.h"
#include <glm/gtx/transform.hpp>



Scene::Scene() {
	static_mesh = new Mesh("asset/model/sponza.obj");
	dynamic_mesh = new Mesh("asset/model/bunny.obj");
}

Scene::~Scene() {
}

void Scene::Upload() {
	cuStatic.color = glm::vec3(1.f);
	cuStatic.M = glm::translate(glm::vec3(0));
	cuStatic.init(static_mesh);
	
	cuDynamic.color = glm::vec3(1.f);
	cuDynamic.M = glm::rotate(glm::radians(1.f), glm::vec3(0.f, 1.f, 0.f));
	cuDynamic.init(dynamic_mesh);
	//cuDynamic.M = glm::translate(glm::vec3(0.f, 0.f, 0.3f));
	delete static_mesh, delete dynamic_mesh;
}

void Scene::StaticVoxelization(Voxel*& d_voxel)
{
	InitVoxelization(d_voxel);

	PreProcess(cuStatic);
	Voxelization(cuStatic, d_voxel);
}

void Scene::DynamicVoxelization(Voxel*& d_voxel)
{
	InitVoxelization(d_voxel);

	PreProcess(cuDynamic);
	Voxelization(cuDynamic, d_voxel);
}



void Scene::DrawMesh()
{
	cuStatic.DrawMesh();
	cuDynamic.DrawMesh();
}
