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


def shader_compile(shader: Shader, v_filepath: str, f_filepath: str) -> None:
    '''Compile shader using OpenGL.GL.shaders.compileShader

    Parameters
    ---
    shader: Shader
        shader class

    v_filepath: str
        vertex shader code in string format

    f_filepath: str
        fragment shader code in string format

    Returns
    ---
    None
    '''
    shader.program_id = compileProgram(
            compileShader(v_filepath, GL.GL_VERTEX_SHADER),
            compileShader(f_filepath, GL.GL_FRAGMENT_SHADER))


def shader_clean(shader: Shader) -> None:
    '''Delete shader program id'''
    GL.glDeleteProgram(shader.program_id)


def shader_use(shader: Shader) -> None:
    '''Use shader program id'''
    GL.glUseProgram(shader.program_id)


def shader_set_vec3(shader: Shader, var_name: str, data: glm.Vec3) -> None:
    '''Set a global uniform vec3 variable within shader program code

    Parameters
    ---
    shader: Shader
        shader class

    var_name: str
        name of the uniform vec3 variable within the shader code

    data: Vec3
        the vec3 data that is being passed to this global variable

    Returns
    ---
    None
    '''
    location_id = GL.glGetUniformLocation(shader.program_id, var_name)
    GL.glUniform3f(location_id, data.x, data.y, data.z)


def shader_set_m4(shader: Shader, var_name: str, data: glm.Mat4) -> None:
    '''Set a global uniform mat4 variable within shader program

    Parameters
    ---
    shader: Shader
        shader class

    var_name: str
        name of the uniform mat4 variable within the shader code

    data: Mat4
        the mat4 data that is being passed to this global variable

    Returns
    ---
    None
    '''
    location_id = GL.glGetUniformLocation(shader.program_id, var_name)
    GL.glUniformMatrix4fv(location_id, 1, GL.GL_FALSE, glm.m4_multi_array(data))