"""Model
"""
from abc import ABC, abstractmethod
from uuid import uuid4

from py_opengl import maths
from py_opengl import geometry
from py_opengl import mesh
from py_opengl import transform
from py_opengl import camera
from py_opengl import shader


# ---


class IModel(ABC):

    @abstractmethod
    def compute(self) -> geometry.AABB3:
        pass
    
    @abstractmethod
    def draw(self, _shader: shader.Shader, cam: camera.Camera) -> None:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass


# ---


class CubeModel(IModel):

    __slots__= ('_mesh', '_transform')

    def __init__(self, scale: float) -> None:
        self._mesh= mesh.CubeMesh(maths.Vec3.create_from_value(scale))
        self._transform= transform.Transform()
        self.ID:str = f'CUBE_{uuid4().hex}'

    def __hash__(self) -> int:
        return hash(self.ID)

    def __eq__(self, other: 'CubeModel') -> bool:
        if isinstance(other, self.__class__):
            if self.ID == other.ID:
                return True
        return False

    def set_position(self, v3: maths.Vec3) -> None:
        self._transform.origin.set_from(v3)

    def translate(self, v3: maths.Vec3) -> None:
        self._transform.translated(v3)

    def rotate(self, v3: maths.Vec3) -> None:
        self._transform.rotated_xyz(v3)

    def get_position(self) -> maths.Vec3:
        self._transform.origin

    def compute(self) -> geometry.AABB3:
        return self._mesh.compute_aabb(self._transform)

    def draw(self, _shader: shader.Shader, cam: camera.Camera) -> None:
        _shader.use()
        _shader.set_mat4('m_matrix', self._transform.model_matrix())
        _shader.set_mat4('v_matrix', cam.view_matrix())
        _shader.set_mat4('p_matrix', cam.projection_matrix())
        self._mesh.render()

    def delete(self) -> None:
        self._mesh.delete()
