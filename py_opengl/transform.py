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
        return self.rotation.set_from(q1)

    def translated(self, v3: maths.Vec3) -> None:
        """Translate
        """
        t0: maths.Vec3= self.origin + v3
        self.origin.set_from(t0)

    def model_matrix(self) -> maths.Mat4:
        """Return model matrix
        """
        t: maths.Mat4= maths.Mat4.create_translation(self.origin)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t)

    def inverse(self) -> maths.Mat4:
        """Return inverse model matrix
        """
        t: maths.Mat4= maths.Mat4.create_translation(self.origin)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t).inverse()

    def get_transformed(self, v3: maths.Vec3) -> maths.Vec3:
        """Return v3 transformed
        """
        t: maths.Mat4= maths.Mat4.create_translation(v3 - self.origin)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t).get_transform()

    def get_inverse_transformed(self, v3: maths.Vec3) -> maths.Vec3:
        """Return v3 inverse transformed
        """
        t: maths.Mat4= maths.Mat4.create_translation(v3 - self.origin)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t).inverse().get_transform()

    def get_transformed_rotation(self, v3: maths.Vec3) -> maths.Vec3:
        """Return v3 transformed by rotation only
        """
        t: maths.Mat4= maths.Mat4.create_translation(v3)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t).get_transform()

    def get_inverse_transformed_rotation(self, v3: maths.Vec3) -> maths.Vec3:
        """Return v3 inverse transformed by rotation only
        """
        t: maths.Mat4= maths.Mat4.create_translation(v3)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t).inverse().get_transform()
