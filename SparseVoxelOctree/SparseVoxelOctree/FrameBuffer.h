#pragma once
#include <GL/glew.h>

class FrameBuffer
{
public:
	FrameBuffer(unsigned int w, unsigned int h);
	~FrameBuffer();

	void Enable();
	void DisAble();
	void DrawBuffer();
private:
	GLuint fbo, depth_buffer, textureID;
};

