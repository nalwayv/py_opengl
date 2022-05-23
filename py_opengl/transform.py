"""Transform
"""
from py_opengl import maths


# ---


class Transform:

    __slots__= ('origin', 'rotation')

    def __init__(self) -> None:
        self.origin: maths.Vec3= maths.Vec3.zero()
        self.rotation: maths.Quaternion= maths.Quaternion(w=1)

    def set_position(self, v3: maths.Vec3) -> None:
        """Set position
        """
        self.origin.set_from(v3)

    def set_rotation(self, q: maths.Quaternion) -> None:
        """Set rotation
        """
        self.rotation.set_from(q)

    def translated(self, v3: maths.Vec3) -> None:
        """Translate
        """
        t0: maths.Vec3= self.origin + v3
        self.origin.set_from(t0)

    def rotated(self, angle_deg: float, unit_axis: maths.Vec3) -> None:
        """Rotate based on axis rotation
        """
        q0: maths.Quaternion= maths.Quaternion.create_from_axis(angle_deg, unit_axis)
        q1: maths.Quaternion= q0 * self.rotation
        self.rotation.set_from(q1)

    def rotated_xyz(self, v3: maths.Vec3) -> None:
        """Rotate based on euler rotation
        """
        q0: maths.Quaternion= maths.Quaternion.create_from_euler(v3)
        q1: maths.Quaternion= q0 * self.rotation
        self.rotation.set_from(q1)

    def get_translation(self) -> maths.Vec3:
        t: maths.Mat4= maths.Mat4.create_translation(self.origin)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t).get_translation()

    def get_transform_matrix(self) -> maths.Mat4:
        """Return model matrix
        """
        t: maths.Mat4= maths.Mat4.create_translation(self.origin)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t)

    def get_transformed_matrix(self, v3: maths.Vec3) -> maths.Mat4:
        """
        """
        t: maths.Mat4= maths.Mat4.create_translation(self.origin + v3)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t)
