#define NOMINMAX
#include "cuda_runtime.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "cuda_gl_interop.h"
#include <glm/gtx/transform.hpp>
#include <glm/gtc/integer.hpp>
#include <stdio.h>
#include <time.h>
#include <InitShader.h>
#include "Voxel.cuh"
#include "Octree.cuh"
#include "Camera.h"
#include "FrameBuffer.h"
#include "Scene.h"

Voxel* d_voxel = NULL; Node* d_node = NULL;
GLFWwindow* window; uint WIDTH = WINDOW_WIDTH, HEIGHT = WINDOW_HEIGHT;
GLuint shader, pbo, textureID;
cudaGraphicsResource_t frontCuda, backCuda, pboCuda;
Camera cam(WINDOW_WIDTH, WINDOW_HEIGHT, 3.1415926f / 3.f, glm::vec3(2.f, 0.8f, 0.f), glm::vec3(0.f, 0.0f, 0.f));
FrameBuffer *back;

Mesh Cube("asset/model/cube.obj");
Scene scene;
extern VoxelizationInfo Info;
VoxelizationInfo Info;
uint h_MIPMAP = 0;
bool pause = true;

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
	glm::vec3 _minAABB = glm::vec3(scene.static_mesh->data.min[0], scene.static_mesh->data.min[1], scene.static_mesh->data.min[2]);
	glm::vec3 _maxAABB = glm::vec3(scene.static_mesh->data.max[0], scene.static_mesh->data.max[1], scene.static_mesh->data.max[2]);
	_minAABB -= 0.1f;
	_maxAABB += 0.1f;//Small Offset
	glm::vec3 l = _maxAABB - _minAABB;
	printf("Mesh AABB size: %f\n", glm::length(_maxAABB - _minAABB));
	Info.delta = glm::max(l.x, glm::max(l.y, l.z));
	Info.minAABB = (_minAABB + _maxAABB) / 2.f - Info.delta / 2.f;
	Info.camPos = cam.pos;
	Info.lightPos = glm::vec3(0.5f, 0.5f, 0.f);
	Info.ka = 0.2f, Info.kd = 0.4f, Info.ks = 0.4f;
	Info.alpha = 5.f;
	Info.Dim = voxelDim;
	h_MIPMAP = glm::log2(voxelDim);
}
void init() {
	scene.Upload();
	Cube.CreateVao();
	//Cube.M = glm::translate(Info.minAABB + Info.delta/2.f) * glm::scale(glm::vec3(Info.delta / 2.f));
	back = new FrameBuffer(WINDOW_WIDTH, WINDOW_HEIGHT);
	shader = InitShader("shader/vs.vert", "shader/fs.frag");
	glUseProgram(shader);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	//Bind opengl buffer to cuda
	back->BindToDevice(backCuda);
	if (cudaGraphicsGLRegisterBuffer(&pboCuda, pbo, cudaGraphicsRegisterFlagsNone) != cudaSuccess)
		printf("cuda bind pbo failed\n");

}
void display() {
	glUseProgram(0);
	glViewport(0, 0, WIDTH, HEIGHT);
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
	
	glCullFace(GL_FRONT);
	back->Enable();
	back->DrawBuffer();
	Cube.Draw();
	back->DisAble();
	//cuda map resources
	cudaGraphicsResource_t resources[2] = { backCuda, pboCuda };
	if (cudaGraphicsMapResources(2, resources) != cudaSuccess)
		printf("cuda map resources failed\n");
	cudaArray_t backArray;
	unsigned int* d_pbo;
	if (cudaGraphicsSubResourceGetMappedArray(&backArray, backCuda, 0, 0) != cudaSuccess)
		printf("d_back map pointer failed\n");

	//bind pixel buffer
	size_t numBytes = WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(unsigned int);
	if (cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &numBytes, pboCuda) != cudaSuccess)
		printf("d_pbo map pointer failed\n");
	//run cuda kernel
	RayCastingOctree(d_pbo, cam.pos, backArray, d_node);

	//unmap resource
	if (cudaGraphicsUnmapResources(2, resources) != cudaSuccess)
		printf("cuda unmap resources failed\n");
}

bool update = false;
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	const uint maxLevel = glm::log2(Info.Dim);
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_0:
			h_MIPMAP = maxLevel; break;
		case GLFW_KEY_1:
			h_MIPMAP = maxLevel - 1; break;
		case GLFW_KEY_2:
			h_MIPMAP = maxLevel - 2; break;
		case GLFW_KEY_3:
			h_MIPMAP = maxLevel - 3; break;
		case GLFW_KEY_4:
			h_MIPMAP = maxLevel - 4; break;
		case GLFW_KEY_5:
			h_MIPMAP = maxLevel - 5; break;
		case GLFW_KEY_6:
			h_MIPMAP = maxLevel - 6; break;
		case GLFW_KEY_7:
			h_MIPMAP = maxLevel - 7; break;
		case GLFW_KEY_P:
			pause = !pause;
			break;
		case GLFW_KEY_SPACE:
			update = !update;
			break;
		}
	}
}
void resize_callback(GLFWwindow* window, int width, int height) {
	WIDTH = width, HEIGHT = height;
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
	glfwSetKeyCallback(window, key_callback);
	glfwSetFramebufferSizeCallback(window, resize_callback);

	initRayCasting();
	scene.SceneVoxelization(d_voxel);
	OctreeConstruction(d_node, d_voxel);



	clock_t t = clock();
	float t_total = 0.f, frames = 0.f;
	std::string title;
	//Display
	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
		//Dynamic Update
		if (update) {
			scene.DynamicVoxelization(d_voxel);
			OctreeUpdate(d_node, d_voxel);
		}

		if(!pause)
			cam.pos = glm::vec3(glm::rotate(glm::radians(1.f), cam.up) * glm::vec4(cam.pos, 1.f));
		cam.UpdateViewMatrix();
		Info.camPos = cam.pos;

		glUseProgram(shader);
		cam.upload(shader);
		RayMarching();
		display();


		if (frames++ > 500.f) {
			float averageTime = (clock() - t) / 500.f / CLOCKS_PER_SEC;
			printf("Computation time: %f\n", averageTime);
			title = "Fps: " + std::to_string(int(1.f / averageTime));
			glfwSetWindowTitle(window, title.c_str());
			frames = 0.f; 
			t = clock();
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	delete back;
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
