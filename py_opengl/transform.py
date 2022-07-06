"""Transform
"""
from py_opengl import maths


# ---


class Transform3:

    __slots__= ('basis', 'position')

    def __init__(self) -> None:
        self.basis: maths.Mat3= maths.Mat3.identity()
        self.origin: maths.Vec3= maths.Vec3()

    def set_position(self, v3: maths.Vec3) -> None:
        """
        """
        self.origin.set_from(v3)

    def translate(self, v3: maths.Vec3) -> None:
        """
        """
        self.origin.set_from(self.origin + v3.transform_m3(self.basis))

    def rotate(self, angle_rad: float, unit_axis: maths.Vec3) -> None:
        """
        """
        self.basis.rotate(angle_rad, unit_axis)

    def scale(self, by: maths.Vec3) -> None:
        """
        """
        self.basis.scale(by)
        print(self.basis.get_scale())

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
            maths.Vec4.create_from_v3(self.origin, 1.0),
        )
