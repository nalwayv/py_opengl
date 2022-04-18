"""VBO
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from OpenGL import GL
from py_opengl import utils


class VboDrawMode(Enum):
    TRIANGLES = auto()
    LINE_LOOP = auto()
    LINE_STRIP = auto()
    DEFAULT = auto()


@dataclass(eq=False, repr=False, slots=True)
class Vbo:
    vao_id: int = 0
    components: int = 3     # x y z
    length: int = 0
    normalized: bool = False
    vbos: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.vao_id = GL.glGenVertexArrays(1)

    def use(self) -> None:
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

        GL.glBufferData(GL.GL_ARRAY_BUFFER, size_of, utils.c_arrayF(arr), GL.GL_STATIC_DRAW)

        at = len(self.vbos) - 1
        GL.glVertexAttribPointer(at, self.components, GL.GL_FLOAT, normal, 0, utils.C_VOID_POINTER)
        GL.glEnableVertexAttribArray(at)


@dataclass(eq=False, repr=False, slots=True)
class VboIbo:
    vao_id: int = 0
    ibo_id: int = 0
    vbo_ids: list[int] = field(default_factory=list)
    length: int = 0
    normalized: bool = False

    def use(self, mode: VboDrawMode) -> None:
        """use vbo
        """
        GL.glBindVertexArray(self.vao_id)
        match mode:
           case VboDrawMode.TRIANGLES:
               GL.glDrawElements(GL.GL_TRIANGLES, self.length, GL.GL_UNSIGNED_INT, utils.C_VOID_POINTER)

           case VboDrawMode.LINE_LOOP:
               GL.glDrawElements(GL.GL_LINE_LOOP, self.length, GL.GL_UNSIGNED_INT, utils.C_VOID_POINTER)

           case VboDrawMode.LINE_STRIP:
               GL.glDrawElements(GL.GL_LINE_STRIP, self.length, GL.GL_UNSIGNED_INT, utils.C_VOID_POINTER)

           case VboDrawMode.DEFAULT:
               return

           case _:
               return

    def clean(self) -> None:
        """Clean vbo of currently stored vbo's
        """
        GL.glDeleteVertexArrays(1, self.vao_id)
        GL.glDeleteBuffers(1, self.ibo_id)

        for v in self.vbo_ids:
            GL.glDeleteBuffers(1, v)
        self.vao_id = 0
        self.ibo_id = 0
        self.vbo_ids.clear()

    def setup(
            self,
            verts: list[float],
            color: list[float],
            tex_coords: list[float],
            indices: list[int]) -> None:
        """Setup vbo
        """
        vsize: int = len(verts) * utils.FLOAT_SIZE
        csize: int = len(color) * utils.FLOAT_SIZE
        tsize: int = len(tex_coords) * utils.FLOAT_SIZE
        isize: int = len(indices) * utils.UINT_SIZE
        components: int = 3
        components_texture: int = 2
        normal: bool = GL.GL_TRUE if self.normalized else GL.GL_FALSE
        
        # vao
        self.vao_id = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao_id)

        # vbos
        self.vbo_ids.clear()
        self.vbo_ids = [
            GL.glGenBuffers(1),
            GL.glGenBuffers(1),
            GL.glGenBuffers(1)]

        # verts
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_ids[0])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vsize, utils.c_arrayF(verts), GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, components, GL.GL_FLOAT, normal, 0, utils.C_VOID_POINTER)
        GL.glEnableVertexAttribArray(0)

        # color
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_ids[1])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, csize, utils.c_arrayF(color), GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(1, components, GL.GL_FLOAT, normal, 0, utils.C_VOID_POINTER)
        GL.glEnableVertexAttribArray(1)

        # texture coords
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_ids[2])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, tsize, utils.c_arrayF(tex_coords), GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(2, components_texture, GL.GL_FLOAT, normal, 0, utils.C_VOID_POINTER)
        GL.glEnableVertexAttribArray(2)

        # ibo
        self.ibo_id = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ibo_id)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, isize, utils.c_arrayU(indices), GL.GL_STATIC_DRAW)
