"""VBO
"""
from dataclasses import dataclass, field
from OpenGL import GL
from py_opengl import utils

@dataclass(eq=False, repr=False, slots=True)
class Vbo:
    vao_id: int = 0
    components: int = 3     # x y z
    length: int = 0
    normalized: bool = False
    vbos: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.vao_id = GL.glGenVertexArrays(1)

    def draw(self) -> None:
        """Bind vbo
        """
        GL.glBindVertexArray(self.vao_id)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.length)

    def clean(self) -> None:
        """Clean vbo of currently stored vbo's
        """
        GL.glDeleteVertexArrays(1, self.vao_id)
        for v in self.vbos:
            GL.glDeleteBuffers(1, v)

    def add_data(self, arr: list[float]) -> None:
        """Add data to the vbo's list

        Parameters
        ----------
        arr : list[float]
            list of float values
        """
        v_buffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, v_buffer)
        GL.glBindVertexArray(self.vao_id)
        self.vbos.append(v_buffer)

        normal: bool = GL.GL_TRUE if self.normalized else GL.GL_FALSE
        size_of: int = len(arr) * utils.FLOAT_SIZE

        GL.glBufferData(GL.GL_ARRAY_BUFFER, size_of, utils.c_array(arr), GL.GL_STATIC_DRAW)

        at = len(self.vbos) - 1
        GL.glVertexAttribPointer(at, self.components, GL.GL_FLOAT, normal, 0, utils.c_cast(0))
        GL.glEnableVertexAttribArray(at)