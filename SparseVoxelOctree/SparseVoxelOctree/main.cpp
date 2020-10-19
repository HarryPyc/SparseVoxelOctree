#define NOMINMAX
#include "cuda_runtime.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "cuda_gl_interop.h"
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
GLuint shader, pbo, textureID;
cudaGraphicsResource_t frontCuda, backCuda, pboCuda;
Camera cam(WINDOW_WIDTH, WINDOW_HEIGHT, 3.1415926f / 3.f, glm::vec3(0, 1.5, 1.5));
FrameBuffer *front, *back;
//870K triangle dragon
Mesh mesh("asset/model/dragon.obj"), Cube("asset/model/cube.obj");
CudaMesh cuMesh; 
VoxelizationInfo Info;

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

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(unsigned int), 0, GL_DYNAMIC_COPY);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}
void initInfo() {
	//Reconstruct AABB
	glm::vec3 _minAABB = glm::vec3(mesh.data.min[0], mesh.data.min[1], mesh.data.min[2]);
	glm::vec3 _maxAABB = glm::vec3(mesh.data.max[0], mesh.data.max[1], mesh.data.max[2]);
	glm::vec3 l = _maxAABB - _minAABB;
	Info.delta = glm::max(l.x, glm::max(l.y, l.z));
	Info.minAABB = (_minAABB + _maxAABB) / 2.f - Info.delta / 2.f;
	Info.camPos = cam.pos;
	Info.lightPos = glm::vec3(0.f, 2.f, 2.f);
	Info.ka = 0.2f, Info.kd = 1.0f, Info.ks = 1.0f;
	Info.alpha = 5.f;
	Info.Dim = voxelDim;
	uploadConstant(Info);
}
void init() {
	mesh.UploatToDevice(cuMesh);
	Cube.CreateVao();
	Cube.M = glm::translate(Info.minAABB + Info.delta/2.f) * glm::scale(glm::vec3(Info.delta / 2.f));
	front = new FrameBuffer(WINDOW_WIDTH, WINDOW_HEIGHT), back = new FrameBuffer(WINDOW_WIDTH, WINDOW_HEIGHT);
	shader = InitShader("shader/vs.vert", "shader/fs.frag");
	glUseProgram(shader);
	glUniformMatrix4fv(glGetUniformLocation(shader, "M"), 1, false, &Cube.M[0][0]);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	//Bind opengl buffer to cuda
	front->BindToDevice(frontCuda);
	back->BindToDevice(backCuda);
	if (cudaGraphicsGLRegisterBuffer(&pboCuda, pbo, cudaGraphicsRegisterFlagsNone) != cudaSuccess)
		printf("cuda bind pbo failed\n");

}
void display() {
	glUseProgram(0);
	glEnable(GL_TEXTURE_2D);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, 0.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, -1.0f, 0.0f);
	glEnd();

	glDisable(GL_TEXTURE_2D);
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
	//cuda map resources
	cudaGraphicsResource_t resources[3] = { frontCuda, backCuda, pboCuda };
	if (cudaGraphicsMapResources(3, resources) != cudaSuccess)
		printf("cuda map resources failed\n");
	cudaArray_t frontArray, backArray;
	unsigned int* d_pbo;
	if (cudaGraphicsSubResourceGetMappedArray(&frontArray, frontCuda, 0, 0) != cudaSuccess)
		printf("d_front map pointer failed\n");
	if (cudaGraphicsSubResourceGetMappedArray(&backArray, backCuda, 0, 0) != cudaSuccess)
		printf("d_back map pointer failed\n");

	//bind pixel buffer
	size_t numBytes = WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(unsigned int);
	if (cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboCuda) != cudaSuccess)
		printf("d_pbo map pointer failed\n");
	//run cuda kernel
	RunRayMarchingKernel(d_pbo, frontArray, backArray, d_voxel, cuMesh, WINDOW_WIDTH, WINDOW_HEIGHT);

	//unmap resource
	if (cudaGraphicsUnmapResources(3, resources) != cudaSuccess)
		printf("cuda unmap resources failed\n");
}



int main() {
	/*if (cudaSetDevice(0) != cudaSuccess) {
		printf("cudaSetDevice Failed");
		return 0;
	}*/

	initOpenGL();
	glfwSetErrorCallback(error_callback);
	printGlInfo();
	initInfo();
	init();
	initCudaVoxelization();

	Voxelization(cuMesh, d_voxel);

	//Display
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

		cam.pos = glm::vec3(glm::rotate(glm::radians(1.f), cam.up) * glm::vec4(cam.pos, 1.f));
		cam.UpdateViewMatrix();
		Info.camPos = cam.pos;
		uploadConstant(Info);

		glUseProgram(shader);
		cam.upload(shader);
		RayMarching();
		display();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	delete front, delete back;
	cudaFree(d_voxel);
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
