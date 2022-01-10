"""Transform
"""
from dataclasses import dataclass
from py_opengl import glm

@dataclass(eq=False, repr=False, slots=True)
class Transform:
    position: glm.Vec3 = glm.Vec3()
    scale: float = 1.0
    rotation: glm.Quaternion = glm.Quaternion(w=1.0)

    def get_matrix(self) -> glm.Mat4:
        """Return transform matrix

        Returns
        ---
        glm.Mat4
        """
        return (
            glm.Mat4.create_translation(self.position) *
            self.rotation.to_mat4() * 
            glm.Mat4.create_scaler(glm.Vec3(self.scale, self.scale, self.scale))
        )