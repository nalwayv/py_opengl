"""Mesh
"""
from OpenGL import GL

from py_opengl import utils
from py_opengl import maths
from py_opengl import geometry


# ---


class MeshError(Exception):
    '''
    '''
    def __init__(self, msg: str):
        super().__init__(msg)


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


# ---


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


# ---


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
    
    def compute_aabb(self) -> geometry.AABB3:
        pmin: maths.Vec3= maths.Vec3.create_from_value(maths.MAX_FLOAT)
        pmax: maths.Vec3= maths.Vec3.create_from_value(maths.MIN_FLOAT)

        for vert in self.vertices:
            pt= vert.position

            if pt.x < pmin.x:
                pmin.x = pt.x

            if pt.x > pmax.x:
                pmax.x = pt.x

            if pt.y < pmin.y:
                pmin.y = pt.y

            if pt.y > pmax.y:
                pmax.y = pt.y

            if pt.z < pmin.z:
                pmin.z = pt.z
                
            if pt.z > pmax.z:
                pmax.z = pt.z

        return geometry.AABB3.create_from_min_max(pmin, pmax)

    def get_furthest_pt(self, dir: maths.Vec3) -> maths.Vec3:
        if not dir.is_unit():
            dir.to_unit()
    
        max_pt: maths.Vec3= maths.Vec3.zero()
        max_dis: float= maths.MIN_FLOAT

        for vert in self.vertices:
            dis: float= vert.position.dot(dir)
            if dis > max_dis:
                max_dis= dis
                max_pt.set_from(vert.position)

        return max_pt

    def get_positions(self) -> list[maths.Vec3]:
        """Return a list of position verts only
        """
        return [v.position for v in self.vertices]

    def get_normals(self) -> list[maths.Vec3]:
        return [v.normal for v in self.vertices]

    def render(self, debug: bool):
        count: int=len(self.indices)
        if count == 0:
            return
        self._vao.bind()

        if debug:
            GL.glDrawElements(GL.GL_LINE_LOOP, count, GL.GL_UNSIGNED_INT, utils.c_cast(0))
        else:
            GL.glDrawElements(GL.GL_TRIANGLES, count, GL.GL_UNSIGNED_INT, utils.c_cast(0))

        self._vao.unbind()

    def delete(self) -> None:
        self._vbo.delete()
        self._vao.delete()
        self._ebo.delete()


# ---


class LineMesh(Mesh):

    def __init__(self, start: maths.Vec3, end: maths.Vec3) -> None:
        vertices: list[Vertex]= [
            Vertex(
                start,
                maths.Vec3(0.0, 0.0 ,0.0),
                maths.Vec3(1.0, 0.0, 0.0),
            ),

            Vertex(
                end,
                maths.Vec3(0.0, 0.0, 0.0),
                maths.Vec3(0.0, 1.0, 0.0),
            ), 
        ]

        indices: list[int]= [0, 1]

        super().__init__(vertices, indices)


# ---


class SphereMesh(Mesh):

    def __init__(self, radius: float= 1.0) -> None:
        prec: int= 24
        vnum: int= (prec + 1) * (prec + 1)
        inum: int= prec * prec * 6

        vertices: list[Vertex]= [Vertex(maths.Vec3(), maths.Vec3(), maths.Vec3())] * vnum
        indices: list[int]= [0] * inum

        for i in range(prec + 1):
            for j in range(prec + 1):
                y: float= maths.cos(maths.PI - i * maths.PI / prec)
                x: float= -maths.cos(j * maths.TAU / prec) * maths.absf(maths.cos(maths.arcsin(y)))
                z: float= maths.sin(j * maths.TAU / prec) * maths.absf(maths.cos(maths.arcsin(y)))

                vertices[i * (prec + 1) + j]= Vertex(
                    maths.Vec3(x, y, z) * radius,
                    maths.Vec3(x, y, z),
                    maths.Vec3(x, y, z)
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


class PyramidMesh(Mesh):

    def __init__(self, scale: float= 1.0) -> None:
        vertices: list[Vertex]= [
            Vertex(
                maths.Vec3(scale, -scale, scale),
                maths.Vec3(0, 0, 0),
                maths.Vec3(1, 0, 0)
            ),
            Vertex(
                maths.Vec3(scale, -scale, -scale),
                maths.Vec3(0, 0, 0),
                maths.Vec3(1, 0, 0)
            ),
            Vertex(
                maths.Vec3(-scale, -scale, -scale),
                maths.Vec3(0, 0, 0),
                maths.Vec3(0, 0, 1)
            ),
            Vertex(
                maths.Vec3(-scale, -scale, scale),
                maths.Vec3(0, 0, 0),
                maths.Vec3(0, 0, 1)
            ),
            Vertex(
                maths.Vec3(0, scale, 0),
                maths.Vec3(0, 0, 0),
                maths.Vec3(0, 1, 0)
            ),
        ]

        indices: list[int]= [
            1, 0, 3, 3, 2, 1,
            1, 4, 0,
            0, 4, 3,
            3, 4, 2,
            2, 4, 1
        ]

        super().__init__(vertices, indices)


# ---


class Traingle(Mesh):

    def __init__(self, tri: geometry.Triangle3) -> None:
        vertices: list[Vertex]=[
            Vertex(
                tri.p0,
                maths.Vec3(0,0,0),
                maths.Vec3(1,0,0)
            ),

            Vertex(
                tri.p1,
                maths.Vec3(0,0,0),
                maths.Vec3(0,1,0)
            ),

            Vertex(
                tri.p2,
                maths.Vec3(0,0,0),
                maths.Vec3(0,0,1)
            ),

        ]
        
        indices: list[int]= [0, 1, 2]

        super().__init__(vertices, indices)


# ---


class CubeMesh(Mesh):

    
    def __init__(self, size: maths.Vec3) -> None:

        vertices: list[Vertex]= [
            # front
            Vertex(
                maths.Vec3(size.x, size.y, size.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                maths.Vec3(-size.x, size.y, size.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                maths.Vec3(-size.x, -size.y, size.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                maths.Vec3(size.x, -size.y, size.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            # right
            Vertex(
                maths.Vec3(size.x, size.y, size.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                maths.Vec3(size.x, -size.y, size.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                maths.Vec3(size.x, -size.y, -size.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                maths.Vec3(size.x, size.y, -size.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            # top
            Vertex(
                maths.Vec3(size.x, size.y, size.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                maths.Vec3(size.x, size.y, -size.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                maths.Vec3(-size.x, size.y, -size.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                maths.Vec3(-size.x, size.y, size.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            # left
            Vertex(
                maths.Vec3(-size.x, size.y, size.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                maths.Vec3(-size.x, size.y, -size.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                maths.Vec3(-size.x, -size.y, -size.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                maths.Vec3(-size.x, -size.y, size.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            # bottom
            Vertex(
                maths.Vec3(-size.x, -size.y, -size.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                maths.Vec3(size.x, -size.y, -size.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                maths.Vec3(size.x, -size.y, size.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                maths.Vec3(-size.x, -size.y, size.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            # back
            Vertex(
                maths.Vec3(size.x, -size.y, -size.z),
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                maths.Vec3(-size.x, -size.y, -size.z),
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                maths.Vec3(-size.x, size.y, -size.z),
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                maths.Vec3(size.x, size.y, -size.z),
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


# ---


class CubeMeshAABB(Mesh):

    
    def __init__(self, bounds: geometry.AABB3) -> None:
        minpt= bounds.get_min()
        maxpt= bounds.get_max()

        vertices: list[Vertex]= [
            # front
            Vertex(
                maxpt,
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                maths.Vec3(minpt.x,  maxpt.y,  maxpt.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                maths.Vec3(minpt.x, minpt.y, maxpt.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                maths.Vec3(maxpt.x, minpt.y, maxpt.z),
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            # right
            Vertex(
                maxpt,
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                maths.Vec3(maxpt.x, minpt.y, maxpt.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                maths.Vec3(maxpt.x, minpt.y, minpt.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                maths.Vec3(maxpt.x, maxpt.y, minpt.z),
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            # top
            Vertex(
                maxpt,
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                maths.Vec3(maxpt.x, maxpt.y, minpt.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                maths.Vec3(minpt.x, maxpt.y, minpt.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                maths.Vec3(minpt.x, maxpt.y, maxpt.z),
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            # left
            Vertex(
                maths.Vec3(minpt.x, maxpt.y, maxpt.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                maths.Vec3(minpt.x, maxpt.y, minpt.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                minpt,
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                maths.Vec3(minpt.x, minpt.y, maxpt.z),
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            # bottom
            Vertex(
                minpt,
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                maths.Vec3(maxpt.x, minpt.y, minpt.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                maths.Vec3(maxpt.x, minpt.y, maxpt.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                maths.Vec3(minpt.x, minpt.y, maxpt.z),
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            # back
            Vertex(
                maths.Vec3(maxpt.x, minpt.y, minpt.z),
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                minpt,
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                maths.Vec3(minpt.x, maxpt.y, minpt.z),
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                maths.Vec3(maxpt.x, maxpt.y, minpt.z),
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

# ---



class FrustumMesh(Mesh):

    
    def __init__(self, arr: list[maths.Vec4]) -> None:
        if len(arr) != 8:
            raise MeshError('len of arr was not 8')

        nbl= arr[0].xyz()
        nbr= arr[1].xyz()
        ntl= arr[2].xyz()
        ntr= arr[3].xyz()

        fbl= arr[4].xyz()
        fbr= arr[5].xyz()
        ftl= arr[6].xyz()
        ftr= arr[7].xyz()

        vertices: list[Vertex]= [
            # front
            Vertex(
                ftr,
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                ftl,
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                fbl,
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            Vertex(
                fbr,
                maths.Vec3(0.0, 0.0, 1.0),
                maths.Vec3(1.0, 0.5, 0.5)
            ),
            # right
            Vertex(
                ftr,
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                fbr,
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                nbr,
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            Vertex(
                ntr,
                maths.Vec3(1.0, 0.0, 0.0),
                maths.Vec3(0.5, 0.0, 0.0)
            ),
            # top
            Vertex(
                ftr,
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                ntr,
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                ntl,
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            Vertex(
                ftl,
                maths.Vec3(0.0, 1.0, 0.0),
                maths.Vec3(0.5, 1.0, 0.5)
            ),
            # left
            Vertex(
                ftl,
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                ntl,
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                nbl,
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            Vertex(
                fbl,
                maths.Vec3(-1.0, 0.0, 0.0),
                maths.Vec3(0.0, 0.5, 0.0)
            ),
            # bottom
            Vertex(
                nbl,
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                nbr,
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                fbr,
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            Vertex(
                fbl,
                maths.Vec3(0.0, -1.0, 0.0),
                maths.Vec3(0.5, 0.5, 1.0)
            ),
            # back
            Vertex(
                nbr,
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                nbl,
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                ntl,
                maths.Vec3(0.0, 0.0, -1.0),
                maths.Vec3(0.0, 0.0, 0.5)
            ),
            Vertex(
                ntr,
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

