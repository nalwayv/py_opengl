"""Mesh
"""
# TODO
from OpenGL import GL

from py_opengl import utils
from py_opengl import maths
from py_opengl import geometry
from py_opengl import transform


# ---


class Vertex:

    __slots__= ('position', 'normal', 'color')

    def __init__(self,
        pos: maths.Vec3,
        norm: maths.Vec3,
        col: maths.Vec3,
    ) -> None:
        self.position: maths.Vec3= pos
        self.normal: maths.Vec3= norm
        self.color: maths.Vec3= col

    def to_list(self) -> list[float]:
        # TODO include normals when needed
        return [
            self.position.x, self.position.y, self.position.z,
            self.color.x, self.color.y, self.color.z,
        ]


# ---


class VBO:

    __slots__= ('ID',)

    def __init__(self) -> None:
        self.ID= GL.glGenBuffers(1)
    
    def set_data(self, vertices: list[Vertex]) -> None:
        v_array= [
            value
            for vertex in vertices
            for value in vertex.to_list()
        ]

        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            len(v_array) * utils.SIZEOF_FLOAT,
            utils.c_arrayF(v_array),
            GL.GL_STATIC_DRAW
        )

    def link(self, index:int, components:int, stride: int, offset: int) -> None:
        GL.glEnableVertexAttribArray(index)
        GL.glVertexAttribPointer(
            index,
            components,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            stride,
            utils.c_cast(offset)
        )

    def bind(self) -> None:
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.ID)

    def unbind(self) -> None:
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def delete(self) -> None:
        GL.glDeleteBuffers(1, self.ID)


class VAO:

    __slots__= ('ID',)

    def __init__(self) -> None:
        self.ID: int= GL.glGenVertexArrays(1)

    def bind(self) -> None:
        GL.glBindVertexArray(self.ID)

    def unbind(self) -> None:
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        GL.glDeleteVertexArrays(1, self.ID)


class EBO:

    __slots__= ('ID',)

    def __init__(self) -> None:
        self.ID: int= GL.glGenBuffers(1)

    def set_data(self, indices: list[int]) -> None:
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            len(indices) * utils.SIZEOF_UINT,
            utils.c_arrayU(indices),
            GL.GL_STATIC_DRAW
        )

    def bind(self) -> None:
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ID)

    def unbind(self) -> None:
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def delete(self) -> None:
        GL.glDeleteBuffers(1, self.ID)


# ---


class Mesh:

    __slots__= ('vertices', 'indices', '_vao', '_vbo', '_ebo')

    def __init__(self, vertices: list[Vertex], indices: list[int]) -> None:
        self.vertices: list[Vertex]= vertices
        self.indices: list[int]= indices

        self._vao= VAO()
        self._vbo= VBO()
        self._ebo= EBO()

        self._vao.bind()
        self._vbo.bind()

        self._vbo.set_data(self.vertices)

        self._ebo.bind()
        self._ebo.set_data(self.indices)

        self._vbo.link(0, 3, 6 * utils.SIZEOF_FLOAT, 0 * utils.SIZEOF_FLOAT) 
        self._vbo.link(1, 3, 6 * utils.SIZEOF_FLOAT, 3 * utils.SIZEOF_FLOAT)

        self._vbo.unbind()
        self._vao.unbind()
        self._ebo.unbind()

    def compute_aabb(self, transform: transform.Transform) -> geometry.AABB3:
        p0: maths.Vec3= maths.Vec3.create_from_value(maths.MAX_FLOAT)
        p1: maths.Vec3= maths.Vec3.create_from_value(-maths.MAX_FLOAT)

        for vert in self.vertices:
            pt= transform.get_transformed(vert.position)

            if pt.x < p0.x:
                p0.x = pt.x
            elif pt.x > p1.x:
                p1.x = pt.x

            if pt.y < p0.y:
                p0.y = pt.y
            elif pt.y > p1.y:
                p1.y = pt.y

            if pt.z < p0.z:
                p0.z = pt.z
            elif pt.z > p1.z:
                p1.z = pt.z

        return geometry.AABB3.create_from_min_max(p0, p1)

    def get_positions(self) -> list[maths.Vec3]:
        """Return a list of position verts only
        """
        return [v.position for v in self.vertices]

    def get_normals(self) -> list[maths.Vec3]:
        return [v.normal for v in self.vertices]

    def render(self):
        count: int=len(self.indices)
        if count == 0:
            return
        self._vao.bind()
        GL.glDrawElements(GL.GL_TRIANGLES, count, GL.GL_UNSIGNED_INT, utils.c_cast(0))
        self._vao.unbind()

    def delete(self) -> None:
        self._vbo.delete()
        self._vao.delete()
        self._ebo.delete()


# ---


class SphereMesh(Mesh):

    __slots__= ('radius',)

    def __init__(self, radius: float= 1.0) -> None:
        self.radius: float= radius

        prec: int= 24
        vnum: int= (prec + 1) * (prec + 1)
        inum: int= prec * prec * 6

        vertices: list[Vertex]= [Vertex] * vnum
        indices: list[int]= [0] * inum

        for i in range(prec + 1):
            for j in range(prec + 1):
                y: float= maths.cos(maths.to_rad(180.0 - i * 180.0 / prec))
                x: float= -maths.cos(maths.to_rad(j * 360.0 / prec)) * maths.absf(maths.cos(maths.arcsin(y)))
                z: float= maths.sin(maths.to_rad(j * 360.0 / prec)) * maths.absf(maths.cos(maths.arcsin(y)))

                vertices[i * (prec + 1) + j]= Vertex(
                    position= maths.Vec3(x, y, z) * self.radius,
                    normal= maths.Vec3(x, y, z),
                    color= maths.Vec3(x, y, z)
                )

        for i in range(prec):
            for j in range(prec):
                indices[6 * (i * prec + j) + 0]= i * (prec + 1) + j
                indices[6 * (i * prec + j) + 1]= i * (prec + 1) + j + 1
                indices[6 * (i * prec + j) + 2]= (i + 1) * (prec + 1) + j
                indices[6 * (i * prec + j) + 3]= i * (prec + 1) + j + 1
                indices[6 * (i * prec + j) + 4]= (i + 1) * (prec + 1) + j + 1
                indices[6 * (i * prec + j) + 5]= (i + 1) * (prec + 1) + j

        super().__init__(vertices, indices)


# ---


class CubeMesh(Mesh):

    __slots__= ('size',)
    
    def __init__(self, size: maths.Vec3) -> None:
        self.size: maths.Vec3= size

        vertices: list[Vertex]= [
            # front
            Vertex(
                maths.Vec3(self.size.x,  self.size.y,  self.size.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                maths.Vec3(-self.size.x,  self.size.y,  self.size.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                maths.Vec3(-self.size.x, -self.size.y,  self.size.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                maths.Vec3(self.size.x, -self.size.y,  self.size.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            # right
            Vertex(
                maths.Vec3(self.size.x,  self.size.y,  self.size.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                maths.Vec3(self.size.x, -self.size.y,  self.size.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                maths.Vec3(self.size.x, -self.size.y, -self.size.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                maths.Vec3(self.size.x,  self.size.y, -self.size.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            # top
            Vertex(
                maths.Vec3(self.size.x,  self.size.y,  self.size.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                maths.Vec3(self.size.x,  self.size.y, -self.size.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                maths.Vec3(-self.size.x,  self.size.y, -self.size.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                maths.Vec3(-self.size.x,  self.size.y,  self.size.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            # left
            Vertex(
                maths.Vec3(-self.size.x,  self.size.y,  self.size.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                maths.Vec3(-self.size.x,  self.size.y, -self.size.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                maths.Vec3(-self.size.x, -self.size.y, -self.size.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                maths.Vec3(-self.size.x, -self.size.y,  self.size.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            # bottom
            Vertex(
                maths.Vec3(-self.size.x, -self.size.y, -self.size.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                maths.Vec3( self.size.x, -self.size.y, -self.size.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                maths.Vec3( self.size.x, -self.size.y,  self.size.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                maths.Vec3(-self.size.x, -self.size.y,  self.size.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            # back
            Vertex(
                maths.Vec3( self.size.x, -self.size.y, -self.size.z),
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                maths.Vec3(-self.size.x, -self.size.y, -self.size.z),
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                maths.Vec3(-self.size.x,  self.size.y, -self.size.z),
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                maths.Vec3( self.size.x,  self.size.y, -self.size.z),
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            )
        ]

        indices: list[int]= [
             0,  1,  2,  2,  3,  0,
             4,  5,  6,  6,  7,  4,
             8,  9, 10, 10, 11,  8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20
        ]

        super().__init__(vertices, indices)
