"""Model
"""
from uuid import uuid4

from py_opengl import maths
from py_opengl import geometry
from py_opengl import mesh
from py_opengl import transform
from py_opengl import shader


# ---


class Model:

    __slots__= ('_transform', '_mesh', 'ID')

    def __init__(self, mesh_data: mesh.Mesh) -> None:
        self._mesh= mesh_data
        self._transform= transform.Transform()
        self.ID:str= uuid4().hex

    def __hash__(self) -> int:
        return hash(self.ID)

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            if self.ID == other.ID:
                return True
        return False

    def __str__(self) -> str:
        return f'[NAME: {self.__class__.__name__}, ID: {self.ID}]'

    def set_position(self, v3: maths.Vec3) -> None:
        """Hard set position
        """
        self._transform.set_position(v3)

    def set_scale(self, v3: maths.Vec3) -> None:
        """Hard set scale
        """
        self._transform.set_scale(v3)

    def get_position(self) -> maths.Vec3:
        """Return current position
        """
        return self._transform.position

    def translate(self, v3: maths.Vec3) -> None:
        """Translate model by 'xyz'
        """
        self._transform.translate(v3)

    def rotate(self, euler: maths.Vec3) -> None:
        """Rotate model by 'xyz'
        """
        self._transform.rotate_euler(euler)

    def get_furthest_pt(self, dir: maths.Vec3) -> maths.Vec3:
        """Return furthest pt
        """
        result= self._mesh.get_furthest_pt(dir)
        result.set_from(result + self._transform.position)
        return result
        
    def compute_aabb(self) -> geometry.AABB3:
        """Compute AABB3
        """
        result= self._mesh.compute_aabb()
        result.transform(self._transform.get_transform_matrix())
        return result

    def draw(
        self,
        _shader: shader.Shader,
        view: maths.Mat4,
        projection: maths.Mat4,
        debug:bool= False
    ) -> None:
        """Draw
        """
        _shader.use()
        _shader.set_mat4('m_matrix', self._transform.get_transform_matrix())
        _shader.set_mat4('v_matrix', view)
        _shader.set_mat4('p_matrix', projection)
        self._mesh.render(debug)

    def delete(self) -> None:
        """Delete
        """
        self._mesh.delete()


# ---


class CubeModel(Model):

    __slots__= ('extents',)

    def __init__(self, extents: maths.Vec3) -> None:
        self.extents: float= extents

        super().__init__(
            mesh.CubeMesh(self.extents)
        )

    def compute_aabb(self) -> geometry.AABB3:
        """OVERRIDE:: Compute AABB3
        """
        c= self._transform.get_transform_matrix().get_translation()
        e= self.extents
        return geometry.AABB3(c, e)


# ---


class LineModel(Model):
    
    __slots__= ('start', 'end')
    
    def __init__(self, start: maths.Vec3, end: maths.Vec3) -> None:
        self.start= start
        self.end= end

        super().__init__(
            mesh.LineMesh(self.start, self.end)
        )


# ---


class TriModel(Model):
    
    __slots__= ('tri')
    
    def __init__(self, tri: geometry.Triangle3) -> None:
        self.tri:geometry.Triangle3= tri

        super().__init__(
            mesh.Traingle(self.tri)
        )


# ---


class PyramidModel(Model):

    __slots__= ('scale',)

    def __init__(self, scale) -> None:
        self.scale: float= scale

        super().__init__(
            mesh.PyramidMesh(self.scale)
        )


# ---


class SphereModel(Model):

    __slots__= ('radius',)

    def __init__(self, radius: float= 1.0) -> None:
        self.radius: float= radius

        super().__init__(
            mesh.SphereMesh(self.radius)
        )



# ---


class FrustumModel(Model):

    def __init__(self, arr: list[maths.Vec3]) -> None:

        super().__init__(
            mesh.FrustumMesh(arr)
        )
