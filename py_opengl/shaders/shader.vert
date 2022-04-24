# version 430 core
// A simple shader that includes vert position, color and texture
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_col;
layout (location = 2) in vec3 a_norm;
layout (location = 3) in vec2 a_texture;

out vec3 b_col;
out vec3 b_norm;
out vec2 b_texture;
uniform mat4 mvp;

void main(void)
{
    b_col = a_col;
    b_norm = a_norm;
    b_texture = a_texture;

    gl_Position = mvp * vec4(a_pos, 1.0);
}