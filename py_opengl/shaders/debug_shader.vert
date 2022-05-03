# version 430 core

layout (location = 0) in vec3 a_pos;

uniform mat4 mvp;

void main(void)
{
    gl_Position = mvp * vec4(a_pos, 1.0);
}