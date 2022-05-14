# version 430 core

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_col;

out vec3 b_col;

uniform mat4 m_matrix;
uniform mat4 v_matrix;
uniform mat4 p_matrix;

void main(void)
{
    vec3 current_pos = vec3(m_matrix * vec4(a_pos, 1.0));
    mat4 cam_matrix = p_matrix * v_matrix;

    gl_Position = cam_matrix * vec4(current_pos, 1.0);
    
    // out
    b_col = a_col;
}