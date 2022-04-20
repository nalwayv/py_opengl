"""
Camera
---
A perspective camera
"""
from dataclasses import dataclass
from py_opengl import maths
from enum import Enum, auto


class CameraError(Exception):
    '''Custom error for camera 4x4'''

    def __init__(self, msg: str):
        super().__init__(msg)


class CameraDirection(Enum):
    UP= auto()
    DOWN= auto()
    LEFT= auto()
    RIGHT= auto()
    IN= auto()
    OUT= auto()
    DEFAULT= auto()


class CameraRotation(Enum):
    YAW= auto()
    PITCH= auto()
    ROLL= auto()
    DEFAULT= auto()


class CameraZoom(Enum):
    IN= auto()
    OUT= auto()
    DEFAULT= auto()


@dataclass(eq= False, repr= False, slots= True)
class Camera:
    position: maths.Vec3= maths.Vec3()
    front: maths.Vec3= maths.Vec3(z=-1.0)
    up: maths.Vec3= maths.Vec3(y=1.0)
    right: maths.Vec3= maths.Vec3(x=1.0)
    aspect: float= 1.0
    fovy: float= maths.PHI
    yaw: float= -maths.PHI
    pitch: float= 0.0
    znear: float= 0.01
    zfar: float= 1000.0

    def move_by(self, dir: CameraDirection, sensativity: float, dt: float) -> None:
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
            case CameraDirection.UP:
                self.position -= self.up * (sensativity * dt)

            case CameraDirection.DOWN:
                self.position += self.up * (sensativity * dt)

            case CameraDirection.RIGHT:
                self.position -= self.right * (sensativity * dt)

            case CameraDirection.LEFT:
                self.position += self.right * (sensativity * dt)

            case CameraDirection.OUT:
                self.position -= self.front * (sensativity * dt)

            case CameraDirection.IN:
                self.position += self.front * (sensativity * dt)

            case _: raise CameraError('unknown camera direction')
    
    def rotate_by(self, dir: CameraRotation, value: float, sensativity: float) -> None:
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
            case CameraRotation.YAW:
                self.yaw -= maths.to_radians(value * sensativity)

            case CameraRotation.PITCH:
                self.pitch += maths.to_radians(maths.clampf(value * sensativity, -89.0, 89.0))

            case CameraRotation.ROLL:
                return

            case _: raise CameraError('unknown camera rotation')

    def zoom_by(self, dir: CameraZoom, value: float, sensativity: float) -> None:
        """Zoom camera by value

        Parameters
        ---
        dir : CameraZoom

        value : float

        sensativity : float
            
        Raises
        ---
        CameraError
            not a valid zoom direction
        """
        match dir:
            case CameraZoom.OUT: 
                self.fovy= maths.clampf(
                    self.fovy + maths.to_radians(maths.clampf(value * sensativity, 1.0, 45.0)), 
                    0.1, 
                    maths.PI
                )
            case CameraZoom.IN: 
                self.fovy= maths.clampf(
                    self.fovy - maths.to_radians(maths.clampf(value * sensativity, 1.0, 45.0)), 
                    0.1, 
                    maths.PI
                )
            case _: raise CameraError('unknown camera zoom')

    def update(self) -> None:
        """Update camera's up, right and front fields
        """
        self.front.x= maths.cos(self.pitch) * maths.cos(self.yaw)
        self.front.y= maths.sin(self.pitch)
        self.front.z= maths.cos(self.pitch) * maths.sin(self.yaw)

        self.front.to_unit()
        self.right= self.front.cross(maths.Vec3(y=1))
        self.right.to_unit()
        self.up= self.right.cross(self.front)

    def projection_matrix(self) -> maths.Mat4:
        """Return projection matrix

        Returns
        ---
        glm.Mat4
            camera perspective matrix
        """
        # return glm.Mat4.ortho_projection(-1, 1, 1, -1, 1, 100)
        return maths.Mat4.frustum_projection(self.fovy, self.aspect, self.znear, self.zfar)


    def view_matrix(self) -> maths.Mat4:
        """Return view matrix

        Returns
        ---
        glm.Mat4
            cameras view matrix
        """
        return maths.Mat4.look_at(self.position, self.position + self.front, self.up)