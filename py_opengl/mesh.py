"""Mesh
"""
# TODO
from dataclasses import dataclass

from OpenGL import GL

from py_opengl import utils
from py_opengl import shader
from py_opengl import camera
from py_opengl import maths
from py_opengl import geometry
from py_opengl import transform


# ---


@dataclass(eq= False, repr= False, slots= True)
class Vertex:
    position: maths.Vec3= maths.Vec3()
    normal: maths.Vec3= maths.Vec3()
    color: maths.Vec3= maths.Vec3()

    def to_list(self) -> list[float]:
        # TODO include normals when needed
        return [
            self.position.x, self.position.y, self.position.z,
            self.color.x, self.color.y, self.color.z,
        ]


# ---


class VBO:

    __slots__= ('_id',)

    def __init__(self, vertices: list[Vertex]) -> None:
        self._id= GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._id)
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


    def bind(self) -> None:
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._id)

    def unbind(self) -> None:
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def delete(self) -> None:
        GL.glDeleteBuffers(1, self._id)


class VAO:

    __slots__= ('_id',)

    def __init__(self) -> None:
        self._id: int= GL.glGenVertexArrays(1)

    def link(self, vbo: VBO, index:int, components:int, stride: int, offset: int) -> None:
        vbo.bind()
        GL.glEnableVertexAttribArray(index)
        GL.glVertexAttribPointer(
            index,
            components,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            stride * utils.SIZEOF_FLOAT,
            utils.c_cast(offset)
        )
        vbo.unbind()

    def bind(self) -> None:
        GL.glBindVertexArray(self._id)

    def unbind(self) -> None:
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        GL.glDeleteVertexArrays(1, self._id)


class EBO:

    __slots__= ('_id',)

    def __init__(self, indices: list[int]) -> None:
        self._id: int= GL.glGenBuffers(1)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._id)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            len(indices) * utils.SIZEOF_UINT,
            utils.c_arrayU(indices),
            GL.GL_STATIC_DRAW
        )

    def bind(self) -> None:
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._id)

    def unbind(self) -> None:
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def delete(self) -> None:
        GL.glDeleteBuffers(1, self._id)


# ---


class Mesh:

    __slots__= ('vertices', 'indices', '_vao', '_vbo', '_ebo')

    def __init__(self, vertices: list[Vertex], indices: list[int]) -> None:
        self.vertices: list[Vertex]= vertices
        self.indices: list[int]= indices

        self._vao= VAO()
        self._vao.bind()

        self._vbo= VBO(self.vertices)
        self._ebo= EBO(self.indices)

        # position:: offset=0
        self._vao.link(self._vbo, 0, 3, 6, 0) 
        # color:: offset= 3 * floatsize
        self._vao.link(self._vbo, 1, 3, 6, 12)

        self._vao.unbind()
        self._vbo.unbind()
        self._ebo.unbind()

    def compute_aabb(self, transform: transform.Transform) -> geometry.AABB3:
        min_pt= maths.Vec3()
        max_pt= maths.Vec3()

        for vert in self.vertices:
            pt= transform.get_transformed(vert.position)

            if pt.x < min_pt.x:
                min_pt.x= pt.x
            elif pt.x > max_pt.x:
                max_pt.x= pt.x

            if pt.y < min_pt.y:
                min_pt.y= pt.y
            elif pt.y > max_pt.y:
                max_pt.y= pt.y

            if pt.z < min_pt.z:
                min_pt.z= pt.z
            elif pt.z > max_pt.z:
                max_pt.z= pt.z

        return geometry.AABB3.create_from_min_max(min_pt, max_pt)

    def get_positions(self) -> list[maths.Vec3]:
        """Return a list of position verts only
        """
        return [v.position for v in self.vertices]

    def get_normals(self) -> list[maths.Vec3]:
        return [v.normal for v in self.vertices]

    def use(self):
        self._vao.bind()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, utils.c_cast(0))

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

