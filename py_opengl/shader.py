'''Shader
'''
from dataclasses import dataclass
from OpenGL import GL
from py_opengl import glm
from OpenGL.GL.shaders import compileShader, compileProgram


@dataclass(eq=False, repr=False, slots=True)
class Shader:
    shader_id: int = 0

    def __post_init__(self):
        self.shader_id = GL.glCreateProgram()

    def compile(self, v_src: str, f_src: str) -> None:
        """Compile shader

        Parameters
        ---
        v_src : str
            vertex shader code in str format
        f_src : str
            fragment shader code in str format
        """
        self.shader_id = compileProgram(
                compileShader(v_src, GL.GL_VERTEX_SHADER),
                compileShader(f_src, GL.GL_FRAGMENT_SHADER))

    def clean(self) -> None:
        """Clean shader by deleteing the stored shader program id
        """
        GL.glDeleteProgram(self.shader_id)

    def use(self) -> None:
        """Use this shader
        """
        GL.glUseProgram(self.shader_id)

    def set_vec2(self, var_name: str, data: glm.Vec2) -> None:
        """Set a global uniform vec2 variable within the shader program

        Parameters
        ---
        var_name : str
            name of the uniform vec2 variable
        data : glm.Vec2
            the data that it will be set to
        """
        location_id = GL.glGetUniformLocation(self.shader_id, var_name)
        GL.glUniform2f(location_id, data.x, data.y)

    def set_vec3(self, var_name: str, data: glm.Vec3) -> None:
        """Set a global uniform vec3 variable within the shader program

        Parameters
        ---
        var_name : str
            name of the uniform vec3 variable
        data : glm.Vec3
            the data that it will be set to
        """
        location_id = GL.glGetUniformLocation(self.shader_id, var_name)
        GL.glUniform3f(location_id, data.x, data.y, data.z)

    def set_vec4(self, var_name: str, data: glm.Vec4) -> None:
        """Set a global uniform vec4 variable within the shader program

        Parameters
        ---
        var_name : str
            name of the uniform vec4 variable
        data : glm.Vec4
            the data that it will be set to
        """
        location_id = GL.glGetUniformLocation(self.shader_id, var_name)
        GL.glUniform4f(location_id, data.x, data.y, data.z, data.w)

    def set_m4(self, var_name: str, data: glm.Mat4) -> None:
        """Set a global uniform mat4 variable within the shader program

        Parameters
        ---
        var_name : str
            name of the uniform mat4 variable
        data : glm.Mat4
            the data that it will be set to
        """
        location_id = GL.glGetUniformLocation(self.shader_id, var_name)
        GL.glUniformMatrix4fv(location_id, 1, GL.GL_FALSE, data.multi_array())