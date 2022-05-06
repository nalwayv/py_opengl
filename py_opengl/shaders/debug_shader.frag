# version 430 core

in vec3 b_col;

out vec4 c_col;

void main(void)
{
    c_col = vec4(b_col, 1.0);
}