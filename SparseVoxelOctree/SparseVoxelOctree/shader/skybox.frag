#version 400

out vec4 fragcolor;           
in vec3 pos;

void main(void)
{   
   fragcolor = vec4(pos, 1.0);
}




















