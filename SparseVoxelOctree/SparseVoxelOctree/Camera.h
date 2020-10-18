#pragma once
#include <glm/glm.hpp>
#include <GL/glew.h>

class Camera
{
public:
	glm::vec3 pos, dir, up;
	float w, h, hfov;

	glm::vec3 getTarget() { return pos + dir; }

	void UpdatePerspectiveMatrix();
	void UpdateViewMatrix();

	void upload(GLuint program);

	Camera(float width, float height, float _hfov = 3.1415926f / 3.f, glm::vec3 _pos = glm::vec3(0, 0, 2), glm::vec3 _target = glm::vec3(0, 0, 0),
		glm::vec3 _up = glm::vec3(0, 1, 0));
	~Camera() {}
private:
	glm::mat4 P, V;
};

