"""Shader
"""
from pathlib import Path

from OpenGL import GL
from py_opengl import maths


# ---


class ShaderError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)



class Shader:
    
    __slots__= ('vshader', 'fshader', 'ID')

    def __init__(self, vshader: str, fshader: str) -> None:
        """
        Raises
        ---
        ShaderError:
            if vertex or fragment shader file is not found or failes to compile
        """
        self.vshader: str= vshader
        self.fshader: str= fshader
        self.ID: int= GL.glCreateProgram()

        v_file: Path= Path(f'py_opengl/shaders/{self.vshader}').absolute()
        f_file: Path= Path(f'py_opengl/shaders/{self.fshader}').absolute()

        if not v_file.exists():
            GL.glDeleteProgram(self.ID)
            raise ShaderError('vert file was not found within shaders folder')
        
        if not f_file.exists():
            GL.glDeleteProgram(self.ID)
            raise ShaderError('frag file was not found within shaders folder')

        with(
            open(v_file.as_posix(), mode= 'r') as v,
            open(f_file.as_posix(), mode= 'r') as f
        ):

            vs= GL.glCreateShader(GL.GL_VERTEX_SHADER)
            GL.glShaderSource(vs, v)
            GL.glCompileShader(vs)
            compile_status= GL.glGetShaderiv(vs, GL.GL_COMPILE_STATUS)
            if not compile_status:
                raise ShaderError('failed to compile vertex shader')

            fs= GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
            GL.glShaderSource(fs, f)
            GL.glCompileShader(fs)
            compile_status= GL.glGetShaderiv(fs, GL.GL_COMPILE_STATUS)
            if not compile_status:
                raise ShaderError('failed to compile fragment shader')

            self.ID= GL.glCreateProgram()
            GL.glAttachShader(self.ID, vs)
            GL.glAttachShader(self.ID, fs)
            GL.glLinkProgram(self.ID)
            link_status= GL.glGetProgramiv(self.ID, GL.GL_LINK_STATUS)
            if link_status == GL.GL_FALSE:
                raise ShaderError('failed to link shader')

            valid_status= GL.glGetProgramiv(self.ID, GL.GL_VALIDATE_STATUS)
            if valid_status == GL.GL_FALSE:
                raise ShaderError('falied to validate shader')

            GL.glDetachShader(self.ID, vs)
            GL.glDetachShader(self.ID, fs)
            GL.glDeleteShader(vs)
            GL.glDeleteShader(fs)

    def delete(self) -> None:
        """Delete the stored shader id
        """
        GL.glDeleteProgram(self.ID)

    def use(self) -> None:
        """Activate this shader
        """
        GL.glUseProgram(self.ID)

    def disable(self) -> None:
        """Detach this shader
        """
        GL.glUseProgram(0)

    def set_vec2(self, variable_name: str, value: maths.Vec2) -> None:
        """Set a global uniform vec2 variable within shader
        """
        idx: int= GL.glGetUniformLocation(self.ID, variable_name)
        if idx >= 0:
            GL.glUniform2f(idx, value.x, value.y)

    def set_vec3(self, variable_name: str, value: maths.Vec3) -> None:
        """Set a global uniform vec3 variable within shader
        """
        idx: int= GL.glGetUniformLocation(self.ID, variable_name)
        if idx >= 0:
            GL.glUniform3f(idx, value.x, value.y, value.z)

    def set_vec4(self, variable_name: str, value: maths.Vec4) -> None:
        """Set a global uniform vec4 variable within shader
        """
        idx: int= GL.glGetUniformLocation(self.ID, variable_name)
        if idx >= 0:
            GL.glUniform4f(idx, value.x, value.y, value.z, value.w)

    def set_mat4(self, variable_name: str, value: maths.Mat4) -> None:
        """Set a global uniform mat4 variable within shader 
        """
        idx: int= GL.glGetUniformLocation(self.ID, variable_name)
        if idx >= 0:
            GL.glUniformMatrix4fv(idx, 1, GL.GL_FALSE, value.array())
