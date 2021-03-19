#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void Camera::UpdatePerspectiveMatrix()
{
    P = glm::perspective(hfov, w/h, 0.1f, 300.f);
}

void Camera::UpdateViewMatrix()
{
	V = glm::lookAt(pos,target,up);
}

void Camera::upload(GLuint program)
{
    int PV_loc = glGetUniformLocation(program, "PV");
    if (PV_loc != -1)
    {
        glUniformMatrix4fv(PV_loc, 1, false, glm::value_ptr(P * V));
    }

    glUniform3fv(glGetUniformLocation(program, "camPos"), 1, &pos[0]);
}

Camera::Camera(float width, float height, float _hfov, glm::vec3 _pos, glm::vec3 _target, glm::vec3 _up) :
    pos(_pos), up(_up), target(_target), w(width), h(height), hfov(_hfov)
{
    UpdatePerspectiveMatrix();
    UpdateViewMatrix();
}

