'''
Camera
---
A perspective camera
'''
from dataclasses import dataclass
from py_opengl import glm
from enum import Enum


class CameraError(Exception):
    '''Custom error for camera 4x4'''

    def __init__(self, msg: str):
        super().__init__(msg)


class CameraDir(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    IN = 4
    OUT = 5
    DEFAULT = 6


class CameraRot(Enum):
    YAW = 0
    PITCH = 1
    FOV = 2
    DEFAULT = 3


@dataclass(eq=False, repr=False, slots=True)
class Camera:
    position: glm.Vec3 = glm.Vec3()
    front: glm.Vec3 = glm.Vec3(z=-1.0)
    up: glm.Vec3 = glm.Vec3(y=1.0)
    right: glm.Vec3 = glm.Vec3(x=1.0)
    aspect: float = 1.0

    fovy: float = glm.PIOVER2
    yaw: float = glm.PIOVER2 * -1.0     # -90.0 deg
    pitch: float = 0.0
    znear: float = 0.01
    zfar: float = 1000.0

    # @staticmethod
    # def to_pitch(value: float) -> float:
    #     """Helper func for converting angle values for the cameras pitch field

    #     Parameters
    #     ---
    #     value : float
    #         angle in degreese

    #     Example
    #     ---
    #     ```python
    #     camera = Camera()
    #     camera.pitch += Camera.to_pitch(80.0)
    #     ```

    #     Returns
    #     ---
    #     float
    #         float value in randians
    #     """
    #     return glm.to_radians(glm.clamp(value, -89.0, 89.0))

    # @staticmethod
    # def to_yaw(value: float) -> float:
    #     """Helper func for converting angle values for the camera yaw field

    #     Parameters
    #     ---
    #     value : float
    #         angle in degreese

    #     Example
    #     ---
    #     ```python
    #     camera = Camera()
    #     camera.yaw += Camera.to_yaw(15.0)
    #     ```

    #     Returns
    #     ---
    #     float
    #         float value in randians
    #     """
    #     return glm.to_radians(value)

    # @staticmethod
    # def to_fovy(value: float) -> float:
    #     """Helper func for converting angle values for the cameras fovy field

    #     Parameters
    #     ---
    #     value : float
    #         angle in degreese

    #     Example
    #     ---
    #     ```python
    #     camera = Camera()
    #     camera.fovy += Camera.to_fovy(45.0)
    #     ```

    #     Returns
    #     ---
    #     float
    #         float value in radians
    #     """
    #     return glm.to_radians(glm.clamp(value, 1.0, 45.0))

    def move_by(self, dir: CameraDir, sensativity: float, dt: float) -> None:
        """Move camera

        Parameters
        ----
        dir : CameraDir
            move direction

        sensativity : float
            speed of movement

        dt : float
            delta time to scale move by

        Raises
        ---
        CameraError
            not a valid move direction
        """
        match dir:
            case CameraDir.UP: self.position = self.position - self.up * (sensativity * dt)
            case CameraDir.DOWN: self.position = self.position + self.up * (sensativity * dt)
            case CameraDir.RIGHT: self.position = self.position - self.right * (sensativity * dt)
            case CameraDir.LEFT: self.position = self.position + self.right * (sensativity * dt)
            case CameraDir.OUT: self.position = self.position - self.front * (sensativity * dt)
            case CameraDir.IN: self.position = self.position + self.front * (sensativity * dt)
            
            case _: raise CameraError('unknown camera direction')
    
    def rotate_by(self, dir: CameraRot, value: float, sensativity: float) -> None:
        """Rotate camera by value

        Parameters
        ---
        dir : CameraRot
            rotate direction
        value : float

        sensativity : float
            
        Raises
        ---
        CameraError
            not a valid rotate direction
        """
        match dir:
            case CameraRot.YAW: self.yaw = self.yaw - glm.to_radians(value * sensativity)
            case CameraRot.PITCH: self.pitch = self.pitch + glm.to_radians(glm.clamp(value * sensativity, -89.0, 89.0))
            case CameraRot.FOV: self.fovy = glm.clamp(
                                    self.fovy + glm.to_radians(glm.clamp(value * sensativity, 1.0, 45.0)), 
                                    0.1, 
                                    glm.PI
                                )

            case _: raise CameraError('unknown camera rotation')

    def update(self) -> None:
        """Update camera's up, right and front fields
        """
        self.front.x = glm.cos(self.pitch) * glm.cos(self.yaw)
        self.front.y = glm.sin(self.pitch)
        self.front.z = glm.cos(self.pitch) * glm.sin(self.yaw)

        self.front = self.front.unit()
        self.right = self.front.cross(glm.Vec3(y=1.0)).unit()
        self.up = self.right.cross(self.front)

    def perspective_matrix(self) -> glm.Mat4:
        """Return perspective matrix

        Returns
        ---
        glm.Mat4
            camera perspective matrix
        """
        return glm.Mat4.perspective(self.fovy, self.aspect, self.znear, self.zfar)

    def view_matrix(self) -> glm.Mat4:
        """Return view matrix

        Returns
        ---
        glm.Mat4
            cameras view matrix
        """
        return glm.Mat4.look_at(self.position, self.position + self.front, self.up)