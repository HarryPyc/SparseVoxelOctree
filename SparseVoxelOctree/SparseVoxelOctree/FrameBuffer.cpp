#include "FrameBuffer.h"

FrameBuffer::FrameBuffer(unsigned int w, unsigned int h)
{

	//Depth buffer
	glGenRenderbuffers(1, &depth_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
	//texture
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	//Frame Buffer Object
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

FrameBuffer::~FrameBuffer()
{
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
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
}
