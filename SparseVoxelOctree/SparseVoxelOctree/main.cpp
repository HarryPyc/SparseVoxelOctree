
#include "cuda_runtime.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtx/transform.hpp>
#include <stdio.h>
#include <InitShader.h>
#include "Mesh.h"
#include "Voxel.cuh"
#include "Camera.h"
#include "FrameBuffer.h"

const int WINDOW_WIDTH = 1280, WINDOW_HEIGHT = 720;
Voxel* d_voxel = NULL;
GLFWwindow* window;
GLuint shader;
Camera cam(WINDOW_WIDTH, WINDOW_HEIGHT, 3.1415926f / 3.f, glm::vec3(3, 3, 3));
FrameBuffer *front, *back;
Mesh mesh("asset/model/bunny.obj"), Cube("asset/model/cube.obj");
CudaMesh cuMesh;

void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}
void printGlInfo()
{
	std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
}
void initOpenGL() {
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	if (!glfwInit()) {
		std::cout << "GLFW Init Failed" << std::endl;
	}
	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CS535_Yucong", NULL, NULL);
	if (!window) {
		std::cout << "Window Creation Failed" << std::endl;
	}
	glfwMakeContextCurrent(window);
	if (glewInit() != GLEW_OK)
	{
		std::cout << "GLEW initialization failed.\n";
	}
	glfwSwapInterval(1);
}
void init() {
	mesh.UploatToDevice(cuMesh);
	Cube.CreateVao();
	Cube.M = glm::translate(cuMesh.minAABB + cuMesh.delta/2.f) * glm::scale(glm::vec3(cuMesh.delta / 2.f));
	front = new FrameBuffer(WINDOW_WIDTH, WINDOW_HEIGHT), back = new FrameBuffer(WINDOW_WIDTH, WINDOW_HEIGHT);
	shader = InitShader("shader/vs.vert", "shader/fs.frag");
	glUseProgram(shader);
	glUniformMatrix4fv(glGetUniformLocation(shader, "M"), 1, false, &Cube.M[0][0]);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

}

void RayMarching() {
	glCullFace(GL_BACK);
	front->Enable();
	front->DrawBuffer();
	Cube.Draw();
	front->DisAble();

	glCullFace(GL_FRONT);
	back->Enable();
	back->DrawBuffer();
	Cube.Draw();
	back->DisAble();

}

int main() {
	/*if (cudaSetDevice(0) != cudaSuccess) {
		printf("cudaSetDevice Failed");
		return 0;
	}*/
	initOpenGL();
	init();
	glfwSetErrorCallback(error_callback);
	printGlInfo();

	//Display
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

		cam.upload(shader);
		RayMarching();
		//Cube.Draw();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	delete front, delete back;
	//cudaFree(d_voxel);
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
