#version 400            
uniform mat4 PV;

layout(location = 0)in vec3 pos_attrib; 
layout(location = 1)in vec3 normal_attrib;

out vec3 pos;
out vec3 normal;

void main(void)
{
   gl_Position = PV*vec4(pos_attrib, 1.0);     //w = 1 becase this is a point
   pos = pos_attrib;
   normal = normal_attrib;
}
