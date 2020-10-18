#version 400            
uniform mat4 P;
uniform mat4 V;

layout(location = 0)in vec3 pos_attrib; 
out vec3 pos;
void main(void)
{

   gl_Position = P*V*vec4(pos_attrib, 1.0);     //w = 1 becase this is a point
   pos = pos_attrib;
}
