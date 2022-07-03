"""Transform
"""
from py_opengl import maths


# ---


class Transform3:

    __slots__= ('basis', 'position')

    def __init__(self) -> None:
        self.basis= maths.Mat3.identity()
        self.position= maths.Vec3()

    def set_position(self, v3: maths.Vec3) -> None:
        """
        """
        self.position.set_from(v3)

    def translate(self, v3: maths.Vec3) -> None:
        """
        """
        self.position.set_from(self.position + v3.transform_m3(self.basis))

    def rotate(self, angle_rad: float, unit_axis: maths.Vec3) -> None:
        """
        """
        self.basis.set_from(
            self.basis * maths.Mat3.create_from_axis(angle_rad, unit_axis)
        )

    def scale(self, by: maths.Vec3) -> None:
        """
        """
        self.basis.scale(by)
        self.position.scale(by)

    def look_at(self, eye: maths.Vec3, target: maths.Vec3, up: maths.Vec3) -> None:
        self.basis.set_from(maths.Mat3.create_lookat(target - eye, up))
        self.position= eye

    def looking_at(self, target, up) -> 'Transform3':
        """
        """
        t= Transform3()
        t.basis= maths.Mat3.create_lookat(target - self.position, up)
        t.position= self.position()
        return t

    def orthonormalize(self) -> None:
        """
        """
        self.basis.orthonormalize()
    
    def get_transform_matrix(self) -> maths.Mat4:
        """Return model matrix
        """
        return maths.Mat4(
            maths.Vec4.create_from_v3(self.basis.row0),
            maths.Vec4.create_from_v3(self.basis.row1),
            maths.Vec4.create_from_v3(self.basis.row2),
            maths.Vec4.create_from_v3(self.position, 1.0),
        )
