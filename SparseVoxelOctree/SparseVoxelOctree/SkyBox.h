#pragma once
#include "Mesh.h"
class SkyBox
{
public:
	SkyBox();
	~SkyBox();
	Mesh* Cube;

	void Draw();
private:
	GLuint vao;
	GLuint shader;
};

