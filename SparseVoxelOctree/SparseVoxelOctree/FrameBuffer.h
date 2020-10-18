#pragma once
#include <GL/glew.h>
#include "cuda_gl_interop.h"

class FrameBuffer
{
public:
	FrameBuffer(unsigned int w, unsigned int h);
	~FrameBuffer();

	void BindToDevice(cudaGraphicsResource_t &resource);
	void Enable();
	void DisAble();
	void DrawBuffer();

	GLuint fbo, depth_buffer, textureID;
};

