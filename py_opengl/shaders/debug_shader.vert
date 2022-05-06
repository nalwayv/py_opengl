# version 430 core

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_col;

out vec3 b_col;

uniform mat4 m_matrix;
uniform mat4 v_matrix;
uniform mat4 p_matrix;


void main(void)
{
    b_col = a_col;

    mat4 mvp = p_matrix * v_matrix * m_matrix;
    gl_Position = mvp * vec4(a_pos, 1.0);
}