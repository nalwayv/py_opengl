"""Transform
"""
from py_opengl import maths


# ---


class Transform:

    __slots__= ('position', 'rotation')

    def __init__(self) -> None:
        self.position: maths.Vec3= maths.Vec3.zero()
        self.rotation: maths.Quaternion= maths.Quaternion(w=1)

    def set_position(self, v3: maths.Vec3) -> None:
        """Set position
        """
        self.position.set_from(v3)

    def set_rotate(self, q: maths.Quaternion) -> None:
        """Set rotation
        """
        self.rotation.set_from(q)

    def rotate_euler(self, v3: maths.Vec3) -> None:
        """Rotate based on euler rotation
        """
        q0: maths.Quaternion= maths.Quaternion.create_from_euler(v3)
        q1: maths.Quaternion= q0 * self.rotation
        self.rotation.set_from(q1)

    def translate(self, v3: maths.Vec3) -> None:
        """Translate
        """
        self.position.set_from(self.position + v3)

    def get_transform_matrix(self) -> maths.Mat4:
        """Return model matrix
        """
        t: maths.Mat4= maths.Mat4.create_translation(self.position)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t)

    def get_transformed_matrix(self, v3: maths.Vec3) -> maths.Mat4:
        """
        """
        t: maths.Mat4= maths.Mat4.create_translation(self.position + v3)
        r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
        return (r * t)
