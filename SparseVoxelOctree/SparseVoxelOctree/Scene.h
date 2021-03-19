#pragma once
#include "Mesh.h"
#include "Voxel.cuh"
class Scene
{
public:
	Mesh* static_mesh;
	Mesh* dynamic_mesh;
	Scene();
	~Scene();

	void Upload();
	void StaticVoxelization(Voxel*& d_voxel);
	void DynamicVoxelization(Voxel*& d_voxel);

	void DrawMesh();
private:

	CudaMesh cuStatic, cuDynamic;
};

