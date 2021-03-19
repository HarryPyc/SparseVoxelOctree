#include "FrameBuffer.h"
#include <stdio.h>
#include "Voxel.cuh"
FrameBuffer::FrameBuffer(unsigned int w, unsigned int h, int size)
{
	//Depth buffer
	glGenRenderbuffers(1, &depth_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
	//texture
	tex_size = size;
	texID = new GLuint[tex_size];

	glGenTextures(tex_size, texID);
	for (int i = 0; i < tex_size; i++) {
		glBindTexture(GL_TEXTURE_2D, texID[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	//Frame Buffer Object
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	for (int i = 0; i < tex_size; i++) {
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, texID[i], 0);
	}

	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

FrameBuffer::~FrameBuffer()
{
	glDeleteTextures(tex_size, texID);
	delete[] texID;
}

void FrameBuffer::BindToDevice(cudaGraphicsResource_t **resource)
{
	for (int i = 0; i < tex_size; i++) {
		gpuErrchk(cudaGraphicsGLRegisterImage(resource[i], texID[i], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));		
	}
}

void FrameBuffer::Enable()
{
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
}

void FrameBuffer::DisAble()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FrameBuffer::DrawBuffer()
{
	GLenum* buffers = new GLenum[tex_size];
	for (int i = 0; i < tex_size; i++) {
		buffers[i] = GL_COLOR_ATTACHMENT0 + i;
	}
	glDrawBuffers(tex_size, buffers);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	delete[] buffers;
}
