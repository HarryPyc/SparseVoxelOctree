#include "SkyBox.h"
#include <glm/gtx/transform.hpp>
#include <InitShader.h>

SkyBox::SkyBox()
{
	Cube = new Mesh("asset/model/cube.obj");
	Cube->M = glm::scale(glm::vec3(5.f));
	Cube->CreateVao();
	shader = InitShader("shader/skybox.vert", "shader/skybox.frag");
}

SkyBox::~SkyBox()
{

}

void SkyBox::Draw() {
	glUseProgram(shader);
	glUniformMatrix4fv(glGetUniformLocation(shader, "M"), 1, false, &Cube->M[0][0]);
}