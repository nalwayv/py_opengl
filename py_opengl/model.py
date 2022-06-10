"""Model
"""
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

    def __str__(self) -> str:
        return f'{self.__class__.__name__}:: {self.ID}'

    def set_position(self, v3: maths.Vec3) -> None:
        """Hard set position
        """
        self._transform.position.set_from(v3)

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
        result.translate(self._transform.position)
        return result

    def draw(self, _shader: shader.Shader, cam: camera.Camera, debug:bool=False) -> None:
        """Draw
        """
        _shader.use()
        _shader.set_mat4('m_matrix', self._transform.get_transform_matrix())
        _shader.set_mat4('v_matrix', cam.get_view_matrix())
        _shader.set_mat4('p_matrix', cam.get_projection_matrix())
        self._mesh.render(debug)

    def delete(self) -> None:
        """Delete
        """
        self._mesh.delete()

# ---

class CubeModel(Model):

    __slots__= ('size',)

    def __init__(self, size: maths.Vec3) -> None:
        self.size: float= size

        super().__init__(
            mesh.CubeMesh(self.size)
        )

    def compute_aabb(self) -> geometry.AABB3:
        """OVERRIDE:: Compute AABB3
        """
        return geometry.AABB3(self._transform.position, self.size)


class CubeModelAABB(Model):

    __slots__= ('bounds',)

    def __init__(self, bounds: geometry.AABB3) -> None:
        self.bounds: geometry.AABB3= bounds

        super().__init__(
            mesh.CubeMeshAABB(self.bounds)
        )

    def compute_aabb(self) -> geometry.AABB3:
        """OVERRIDE:: Compute AABB3
        """
        result: geometry.AABB3= self.bounds.copy()
        result.translate(self._transform.position)
        return result


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
        self.tri= tri

        super().__init__(
            mesh.Traingle(self.tri)
        )

    def compute_aabb(self) -> geometry.AABB3:
        """OVERRIDE:: Compute AABB3
        """
        p0: maths.Vec3= self._transform.rotation.multiply_v3(self.p0) + self._transform.position
        p1: maths.Vec3= self._transform.rotation.multiply_v3(self.p1) + self._transform.position
        p2: maths.Vec3= self._transform.rotation.multiply_v3(self.p2) + self._transform.position

        v0= maths.Vec3(p0.x, p1.x, p2.x)
        v1= maths.Vec3(p0.y, p1.y, p2.y)
        v2= maths.Vec3(p0.z, p1.z, p2.z)

        return geometry.AABB3.create_from_min_max(
            maths.Vec3(v0.get_min_value(), v1.get_min_value(), v2.get_min_value()),
            maths.Vec3(v0.get_max_value(), v1.get_max_value(), v2.get_max_value()),
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

    def compute_aabb(self) -> geometry.AABB3:
        """OVERRIDE:: Compute AABB3
        """
        return geometry.AABB3.create_from_min_max(
            self._transform.position - maths.Vec3.create_from_value(self.radius),
            self._transform.position + maths.Vec3.create_from_value(self.radius)
        )

# ---


class FrustumModel(Model):

    __slots__= ('frustum',)

    def __init__(self, frustum: geometry.Frustum) -> None:
        self.frustum= frustum

        super().__init__(
            mesh.FrustumMesh(self.frustum)
        )
