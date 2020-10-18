#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void Camera::UpdatePerspectiveMatrix()
{
    P = glm::perspective(hfov, w/h, 0.1f, 100.f);
}

void Camera::UpdateViewMatrix()
{
	V = glm::lookAt(pos,getTarget(),up);
}

void Camera::upload(GLuint program)
{
    int P_loc = glGetUniformLocation(program, "P");
    if (P_loc != -1)
    {
        glUniformMatrix4fv(P_loc, 1, false, glm::value_ptr(P));
    }
    int V_loc = glGetUniformLocation(program, "V");
    if (V_loc != -1)
    {
        glUniformMatrix4fv(V_loc, 1, false, glm::value_ptr(V));
    }
    glUniform3fv(glGetUniformLocation(program, "camPos"), 1, &pos[0]);
}

Camera::Camera(float width, float height, float _hfov, glm::vec3 _pos, glm::vec3 _target, glm::vec3 _up) :
    pos(_pos), up(_up), w(width), h(height), hfov(_hfov)
{
    dir = glm::normalize(_target - pos);
    UpdatePerspectiveMatrix();
    UpdateViewMatrix();
}

