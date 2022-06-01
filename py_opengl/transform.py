"""Transform
"""
from py_opengl import maths


# ---


class Transform:

    __slots__= ('translation', 'rotation')

    def __init__(self) -> None:
        self.translation: maths.Vec3= maths.Vec3.zero()
        self.rotation: maths.Quaternion= maths.Quaternion(w=1)

    def translate(self, v3: maths.Vec3) -> None:
        """Translate
        """
        self.translation.set_from(self.translation + v3)

    def set_translate(self, v3: maths.Vec3) -> None:
        """Set position
        """
        self.translation.set_from(v3)

    def get_translate(self) -> maths.Vec3:
        return self.translation

    def rotate(self, angle_deg: float, unit_axis: maths.Vec3) -> None:
        """Rotate based on axis rotation
        """
        q0: maths.Quaternion= maths.Quaternion.create_from_axis(angle_deg, unit_axis)
        q1: maths.Quaternion= q0 * self.rotation
        self.rotation.set_from(q1)

    def set_rotate(self, q: maths.Quaternion) -> None:
        """Set rotation
        """
        self.rotation.set_from(q)

    def rotate_xyz(self, v3: maths.Vec3) -> None:
        """Rotate based on euler rotation
        """
        q0: maths.Quaternion= maths.Quaternion.create_from_euler(v3)
        q1: maths.Quaternion= q0 * self.rotation
        self.rotation.set_from(q1)

    def get_transform_matrix(self) -> maths.Mat4:
        """Return model matrix
        """
        t: maths.Mat4= maths.Mat4.create_translation(self.translation)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t)

    def get_transformed_matrix(self, v3: maths.Vec3) -> maths.Mat4:
        """
        """
        t: maths.Mat4= maths.Mat4.create_translation(self.translation + v3)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t)
