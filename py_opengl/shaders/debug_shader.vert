# version 430 core

in vec3 a_pos;

uniform mat4 mvp;

void main(void)
{
    gl_Position = mvp * vec4(a_pos, 1.0);
}