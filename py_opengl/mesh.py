"""Mesh
"""
# TODO
from dataclasses import dataclass, field

from OpenGL import GL

from py_opengl import utils
from py_opengl import maths


# ---


@dataclass(eq= False, repr= False, slots= True)
class Vertex:
    position: maths.Vec3
    color: maths.Vec3

    def to_list(self) -> list[float]:
        return [
            self.position.x,
            self.position.y,
            self.position.z,
            self.color.x,
            self.color.y,
            self.color.z,
        ]



@dataclass(eq= False, repr= False, slots= True)
class Mesh:
    vertices: list[Vertex]= field(default_factory=list)
    indices: list[int]= field(default_factory=list)
    vao: int= -1
    vbo: int= -1
    ebo: int= -1

    def setup(self) -> None:
        self.vao= GL.glGenVertexArrays(1)
        self.vbo= GL.glGenBuffers(1)
        self.ebo= GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao)
        
        v_array= [value for vertex in self.vertices for value in vertex.to_list()]

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            len(v_array) * utils.SIZEOF_FLOAT,
            utils.c_arrayF(v_array),
            GL.GL_STATIC_DRAW
        )

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            len(self.indices) * utils.SIZEOF_UINT,
            utils.c_arrayU(self.indices),
            GL.GL_STATIC_DRAW
        )

        # pos
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * utils.SIZEOF_FLOAT, utils.c_cast(0))
        
        # color
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 6 * utils.SIZEOF_FLOAT, utils.c_cast(3 * utils.SIZEOF_FLOAT))
        
        GL.glBindVertexArray(0)

    def use(self):
        GL.glBindVertexArray(self.vao)
        i_len= len(self.indices)
        GL.glDrawElements(GL.GL_TRIANGLES, i_len, GL.GL_UNSIGNED_INT, utils.c_cast(0))
        GL.glBindVertexArray(0)

    def clean(self) -> None:
        GL.glDeleteVertexArrays(1, self.vao)
        GL.glDeleteBuffers(1, self.vbo)
        GL.glDeleteBuffers(1, self.ebo)
