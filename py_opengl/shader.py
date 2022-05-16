"""Shader
"""
from pathlib import Path

from OpenGL import GL
from OpenGL.GL.shaders import compileShader, compileProgram

from py_opengl import maths


# ---


class ShaderError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)



class Shader:
    
    __slots__= ('vshader', 'fshader', '_id')

    def __init__(self, vshader: str, fshader: str) -> None:
        """
        Raises
        ---
        ShaderError:
            if vertex or fragment shader file is not found or failed to compile
        """
        self.vshader: str= vshader
        self.fshader: str= fshader
        self._id: int= GL.glCreateProgram()

        v_file: Path= Path(f'py_opengl/shaders/{self.vshader}').absolute()
        f_file: Path= Path(f'py_opengl/shaders/{self.fshader}').absolute()

        if not v_file.exists():
            GL.glDeleteProgram(self._id)
            raise ShaderError('vert file was not found within shaders folder')
        
        if not f_file.exists():
            GL.glDeleteProgram(self._id)
            raise ShaderError('frag file was not found within shaders folder')

        with(
            open(v_file.as_posix(), mode= 'r') as v,
            open(f_file.as_posix(), mode= 'r') as f
        ):
            self._id= compileProgram(
                compileShader(v, GL.GL_VERTEX_SHADER),
                compileShader(f, GL.GL_FRAGMENT_SHADER)    
            )

    def delete(self) -> None:
        """Delete the stored shader id
        """
        GL.glDeleteProgram(self._id)

    def use(self) -> None:
        """Activate this shader
        """
        GL.glUseProgram(self._id)

    def disable(self) -> None:
        """Detach this shader
        """
        GL.glUseProgram(0)

    def set_vec2(self, variable_name: str, value: maths.Vec2) -> None:
        """Set a global uniform vec2 variable within shader
        """
        idx: int= GL.glGetUniformLocation(self._id, variable_name)
        if idx >= 0:
            GL.glUniform2f(idx, value.x, value.y)

    def set_vec3(self, variable_name: str, value: maths.Vec3) -> None:
        """Set a global uniform vec3 variable within shader
        """
        idx: int= GL.glGetUniformLocation(self._id, variable_name)
        if idx >= 0:
            GL.glUniform3f(idx, value.x, value.y, value.z)

    def set_vec4(self, variable_name: str, value: maths.Vec4) -> None:
        """Set a global uniform vec4 variable within shader
        """
        idx: int= GL.glGetUniformLocation(self._id, variable_name)
        if idx >= 0:
            GL.glUniform4f(idx, value.x, value.y, value.z, value.w)

    def set_mat4(self, variable_name: str, value: maths.Mat4) -> None:
        """Set a global uniform mat4 variable within shader 
        """
        idx: int= GL.glGetUniformLocation(self._id, variable_name)
        if idx >= 0:
            GL.glUniformMatrix4fv(idx, 1, GL.GL_FALSE, value.array())