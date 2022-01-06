'''Shader
'''
from dataclasses import dataclass
from OpenGL import GL
from py_opengl import glm
from OpenGL.GL.shaders import compileShader, compileProgram


@dataclass(eq=False, repr=False, slots=True)
class Shader:
    program_id: int = 0

    def __post_init__(self):
        self.program_id = GL.glCreateProgram()

    def compile(self, v_src: str, f_src: str) -> None:
        """Compile shader

        Parameters
        ---
        v_src : str
            vertex shader code in str format
        f_src : str
            fragment shader code in str format
        """
        self.program_id = compileProgram(
                compileShader(v_src, GL.GL_VERTEX_SHADER),
                compileShader(f_src, GL.GL_FRAGMENT_SHADER))

    def clean(self) -> None:
        """Clean shader by deleteing the stored shader program id
        """
        GL.glDeleteProgram(self.program_id)

    def use(self) -> None:
        """Use this shader
        """
        GL.glUseProgram(self.program_id)

    def set_vec3(self, var_name: str, data: glm.Vec3) -> None:
        """Set a global uniform vec3 variable within the shader program

        Parameters
        ---
        var_name : str
            name of the uniform vec3 variable
        data : glm.Vec3
            the data that it will be set to
        """
        location_id = GL.glGetUniformLocation(self.program_id, var_name)
        GL.glUniform3f(location_id, data.x, data.y, data.z)

    def set_m4(self, var_name: str, data: glm.Mat4) -> None:
        """Set a global uniform mat4 variable within the shader program

        Parameters
        ---
        var_name : str
            name of the uniform mat4 variable
        data : glm.Mat4
            the data that it will be set to
        """
        location_id = GL.glGetUniformLocation(self.program_id, var_name)
        GL.glUniformMatrix4fv(location_id, 1, GL.GL_FALSE, data.multi_array())