"""Transform
"""
from dataclasses import dataclass
from py_opengl import maths


# ---


@dataclass(eq= False, repr= False, slots= True)
class Transform:
    origin: maths.Vec3= maths.Vec3()
    rotation: maths.Quaternion= maths.Quaternion(w=1)

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
