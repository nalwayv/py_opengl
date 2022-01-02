'''Vbo
'''
from dataclasses import dataclass, field
from OpenGL import GL
from py_opengl import utils

@dataclass(eq=False, repr=False, slots=True)
class Vbo:
    vao: int = 0
    components: int = 3     # x y z
    normalized: bool = False
    vbos: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.vao = GL.glGenVertexArrays(1)


def vbo_draw(vbo: Vbo) -> None:
    '''Bind vbo to vertex Array'''
    GL.glBindVertexArray(vbo.vao)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, vbo.components)


def vbo_clean(vbo: Vbo) -> None:
    '''Clean vbo'''
    GL.glDeleteVertexArrays(1, vbo.vao)
    for v in vbo.vbos:
        GL.glDeleteBuffers(1, v)


def vbo_add_data(vbo: Vbo, arr: list[float]) -> None:
    '''Add data to vbo'''
    v_buffer = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, v_buffer)
    GL.glBindVertexArray(vbo.vao)
    vbo.vbos.append(v_buffer)

    normal: bool = GL.GL_TRUE if vbo.normalized else GL.GL_FALSE
    size_of: int = len(arr) * utils.FLOAT_SIZE

    GL.glBufferData(GL.GL_ARRAY_BUFFER, size_of, utils.c_array(arr), GL.GL_STATIC_DRAW)

    at = len(vbo.vbos) - 1
    GL.glVertexAttribPointer(at, vbo.components, GL.GL_FLOAT, normal, 0, utils.c_cast(0))
    GL.glEnableVertexAttribArray(at)