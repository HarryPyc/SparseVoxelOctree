#version 400

layout(location = 0) out vec4 pos_out;
layout(location = 1) out vec4 normal_out;

in vec3 pos;
in vec3 normal;

void main(void)
{   
   pos_out = vec4(pos, 1.0);
   normal_out = vec4(normalize(normal), 0.0);
}




















