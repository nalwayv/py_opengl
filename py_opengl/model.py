"""Model
"""
from cgitb import reset
from uuid import uuid4

from py_opengl import maths
from py_opengl import geometry
from py_opengl import mesh
from py_opengl import transform
from py_opengl import camera
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

    def set_position(self, v3: maths.Vec3) -> None:
        """Hard set position
        """
        self._transform.origin.set_from(v3)

    def get_position(self) -> maths.Vec3:
        """Return current position
        """
        return self._transform.origin

    def translate(self, v3: maths.Vec3) -> None:
        """Translate model by 'xyz'
        """
        self._transform.translated(v3)

    def rotate(self, v3: maths.Vec3) -> None:
        """Rotate model by 'xyz'
        """
        self._transform.rotated_xyz(v3)

    def get_furthest_pt(self, dir: maths.Vec3) -> maths.Vec3:
        """Return furthest pt
        """
        result= self._mesh.get_furthest_pt(dir)
        result.set_from(result + self._transform.origin)
        return result
        
    def compute_aabb(self) -> geometry.AABB3:
        """Compute AABB3
        """
        result= self._mesh.compute_aabb()
        result.translated(self._transform.origin)
        return result

    def draw(self, _shader: shader.Shader, cam: camera.Camera) -> None:
        """Draw
        """
        _shader.use()
        _shader.set_mat4('m_matrix', self._transform.get_transform_matrix())
        _shader.set_mat4('v_matrix', cam.get_view_matrix())
        _shader.set_mat4('p_matrix', cam.get_projection_matrix())
        self._mesh.render()

    def delete(self) -> None:
        """Delete
        """
        self._mesh.delete()


# ---


class CubeModel(Model):

    __slots__= ('scale',)

    def __init__(self, scale: float= 1.0) -> None:
        self.scale: float= scale

        super().__init__(
            mesh.CubeMesh(maths.Vec3.create_from_value(scale))
        )


# ---


class SphereModel(Model):

    __slots__= ('radius',)

    def __init__(self, radius: float= 1.0) -> None:
        self.radius: float= radius

        super().__init__(
            mesh.SphereMesh(radius)
        )
