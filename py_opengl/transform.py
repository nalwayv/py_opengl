"""Transform
"""
from py_opengl import maths


# ---


# class Transform:

#     __slots__= ('position', 'rotation', 'scale')

#     def __init__(self) -> None:
#         self.position: maths.Vec3= maths.Vec3.zero()
#         self.rotation: maths.Quaternion= maths.Quaternion(w=1)
#         self.scale: maths.Vec3= maths.Vec3.one()

#     def set_position(self, v3: maths.Vec3) -> None:
#         """Set position
#         """
#         self.position.set_from(v3)

#     def set_rotate(self, q: maths.Quaternion) -> None:
#         """Set rotation
#         """
#         self.rotation.set_from(q)

#     def set_scale(self, v3: maths.Vec3) -> None:
#         """Set rotation
#         """
#         self.scale.set_from(v3)

#     def rotate(self, v3: maths.Vec3) -> None:
#         """Rotate based on euler rotation
#         """
#         q0: maths.Quaternion= maths.Quaternion.create_from_euler(v3)
#         q1: maths.Quaternion= q0 * self.rotation
#         self.rotation.set_from(q1)

#     def translate(self, v3: maths.Vec3) -> None:
#         """Translate
#         """
#         self.position.set_from(self.position + v3)

#     def get_transform_matrix(self) -> maths.Mat4:
#         """Return model matrix
#         """
#         t: maths.Mat4= maths.Mat4.create_translation(self.position)
#         s: maths.Mat4= maths.Mat4.create_scaler(self.scale)
#         r: maths.Mat4= maths.Mat4.create_from_quaternion(self.rotation)
#         return (s * r * t)



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
        # self.position.x += self.basis.row0.dot(v3)
        # self.position.y += self.basis.row1.dot(v3)
        # self.position.z += self.basis.row2.dot(v3)
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

    def orthonormalize(self) -> None:
        """
        """
        self.basis.orthonormalize()
    
    def get_transform_matrix(self) -> maths.Mat4:
        """Return model matrix
        """
        return maths.Mat4(
            maths.Vec4.create_from_v3(self.basis.row0, 0.0),
            maths.Vec4.create_from_v3(self.basis.row1, 0.0),
            maths.Vec4.create_from_v3(self.basis.row2, 0.0),
            maths.Vec4.create_from_v3(self.position, 1.0),
        )