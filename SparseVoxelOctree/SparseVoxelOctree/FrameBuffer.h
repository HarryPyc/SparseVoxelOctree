#pragma once
#include <GL/glew.h>
#include "cuda_gl_interop.h"

class FrameBuffer
{
public:
	FrameBuffer(unsigned int w, unsigned int h, int size = 1);
	~FrameBuffer();

	void BindToDevice(cudaGraphicsResource_t **resource);
	void Enable();
	void DisAble();
	void DrawBuffer();

	GLuint fbo, depth_buffer, *texID;
	int tex_size;
};

