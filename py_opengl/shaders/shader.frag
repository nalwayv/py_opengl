# version 430 core
// A simple shader that includes vert position, color and texture
in vec3 b_col;
in vec2 b_texture;

out vec4 c_col;
uniform sampler2D c_texture;

void main(void)
{
    c_col = texture(c_texture, b_texture) * vec4(b_col, 1.0);
}