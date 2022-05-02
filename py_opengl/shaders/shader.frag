# version 430 core
// A simple shader
in vec3 b_col;
in vec2 b_texture;

out vec4 c_col;
uniform sampler2D c_texture;

void main(void)
{
    vec4 color = vec4(b_col, 1.0);
    c_col = texture(c_texture, b_texture) * color;
}