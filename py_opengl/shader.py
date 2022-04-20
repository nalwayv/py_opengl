"""Shader
"""
from dataclasses import dataclass
from pathlib import Path

from OpenGL import GL
from OpenGL.GL.shaders import compileShader, compileProgram

from py_opengl import maths


# ---


class ShaderError(Exception):
    '''Custom error for Shader'''

    def __init__(self, msg: str):
        super().__init__(msg)



@dataclass(eq= False, repr= False, slots= True)
class Shader:
    shader_id: int= 0

    def __post_init__(self):
        self.shader_id= GL.glCreateProgram()

    def compile(self, vert_file: str, frag_file: str) -> None:
        """Compile shader

        Parameters
        ---
        vert_file : str

        frag_file : str

        Raises
        ---
        ShaderError
            if shader is not located within *shaders* folder
        """
        v_file: Path= Path(f'py_opengl/shaders/{vert_file}').absolute()
        f_file: Path= Path(f'py_opengl/shaders/{frag_file}').absolute()

        if not v_file.exists():
            raise ShaderError('vert file was not found within shaders folder')
        
        if not f_file.exists():
            raise ShaderError('frag file was not found within shaders folder')

        with(open(v_file.as_posix(), mode= 'r') as v, open(f_file.as_posix(), mode= 'r') as f):
            self.shader_id = compileProgram(
                compileShader(v, GL.GL_VERTEX_SHADER),
                compileShader(f, GL.GL_FRAGMENT_SHADER)
            )

    def clean(self) -> None:
        """Clean shader by deleteing the stored shader program id
        """
        GL.glDeleteProgram(self.shader_id)

    def use(self) -> None:
        """Use this shader
        """
        GL.glUseProgram(self.shader_id)

    def set_vec2(self, variable_name: str, value: maths.Vec2) -> None:
        """Set a global uniform vec2 variable within the shader program

        Parameters
        ---
        variable_name : str

        value : glm.Vec2

        """
        GL.glUniform2f(
            GL.glGetUniformLocation(self.shader_id, variable_name),
            value.x,
            value.y
        )

    def set_vec3(self, variable_name: str, value: maths.Vec3) -> None:
        """Set a global uniform vec3 variable within the shader program

        Parameters
        ---
        variable_name : str

        value : glm.Vec3

        """
        GL.glUniform3f(
            GL.glGetUniformLocation(self.shader_id, variable_name),
            value.x,
            value.y,
            value.z
        )

    def set_vec4(self, variable_name: str, value: maths.Vec4) -> None:
        """Set a global uniform vec4 variable within the shader program

        Parameters
        ---
        variable_name : str

        value : glm.Vec4

        """
        GL.glUniform4f(
            GL.glGetUniformLocation(self.shader_id, variable_name),
            value.x,
            value.y,
            value.z,
            value.w
        )

    def set_m4(self, variable_name: str, value: maths.Mat4) -> None:
        """Set a global uniform mat4 variable within the shader program

        Parameters
        ---
        variable_name : str

        value : glm.Mat4

        """
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self.shader_id, variable_name),
            1,
            GL.GL_FALSE,
            value.multi_array()
        )