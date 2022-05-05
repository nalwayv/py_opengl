"""Vbo
"""
from dataclasses import dataclass

from OpenGL import GL
from py_opengl import utils


# ---


@dataclass(eq=False, repr= False, slots= True)
class Vao:
    ref: int= -1

    def __post_init__(self):
        self.ref= GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.ref)


    def clean(self) -> None:
        GL.glDeleteVertexArrays(1, self.ref)
        self.ref= -1


@dataclass(eq=False, repr= False, slots= True)
class Vbo:
    ref: int= -1
    components: int= 3
    index: int= 0
    normalized: bool= False

    def __post_init__(self):
        self.ref= GL.glGenBuffers(1)

    def setup(self, data: list[float]) -> None:
        length= len(data) * utils.FLOAT_SIZE
        normal: bool= GL.GL_TRUE if self.normalized else GL.GL_FALSE

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.ref)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, length, utils.c_arrayF(data), GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(self.index, self.components, GL.GL_FLOAT, normal, 0, utils.C_VOID_POINTER)
        GL.glEnableVertexAttribArray(self.index)

    def clean(self) -> None:
        GL.glDeleteBuffers(1, self.ref)
        self.ref= -1


@dataclass(eq=False, repr= False, slots= True)
class Ibo:
    ref: int= -1
    length: int= -1

    def __post_init__(self):
        self.ref = GL.glGenBuffers(1)

    def setup(self, indices: list[int]) -> None:
        self.length = len(indices)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ref)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.length * utils.UINT_SIZE, utils.c_arrayU(indices), GL.GL_STATIC_DRAW)

    def clean(self) -> None:
        GL.glDeleteBuffers(1, self.ref)
        self.ref= -1
