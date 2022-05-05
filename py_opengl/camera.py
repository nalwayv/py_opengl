"""Camera
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
        """
        match dir:
            case CameraDirection.UP:
                u: maths.Vec3= self.right.cross(self.position + self.front)
                if not u.is_unit():
                    u.to_unit()
                u.scaled(sensativity * dt)
                self.position.added(u)

            case CameraDirection.DOWN:
                d: maths.Vec3= (self.position + self.front).cross(self.right)
                if not d.is_unit():
                    d.to_unit()
                d.scaled(sensativity * dt)
                self.position.added(d)

            case CameraDirection.RIGHT:
                r: maths.Vec3= (self.position + self.front).cross(self.up)
                if not r.is_unit():
                    r.to_unit()
                r.scaled(sensativity * dt)
                self.position.added(r)

            case CameraDirection.LEFT:
                l: maths.Vec3= self.up.cross(self.position + self.front)
                if not l.is_unit():
                    l.to_unit()

                l.scaled(sensativity * dt)
                self.position.added(l)

            case CameraDirection.OUT:
                self.position.subbed(self.front * (sensativity * dt))

            case CameraDirection.IN:
                self.position.added(self.front * (sensativity * dt))

        self._update()
    
    def rotate_by(self, dir: CameraRotation, value: float, sensativity: float) -> None:
        """Rotate camera by value

        Parameters
        ---
        dir : CameraRot

        value : float

        sensativity : float
            
        Raises
        ---
        CameraError
        """
        match dir:
            case CameraRotation.YAW:
                self.yaw -= maths.to_radians(value * sensativity)

            case CameraRotation.PITCH:
                self.pitch += maths.to_radians(maths.clampf(value * sensativity, -89.0, 89.0))

            case CameraRotation.ROLL:
                return

        self._update()

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

        self._update()

    def _update(self) -> None:
        """Update camera's up, right and front fields
        """
        self.front.x= maths.cos(self.pitch) * maths.cos(self.yaw)
        self.front.y= maths.sin(self.pitch)
        self.front.z= maths.cos(self.pitch) * maths.sin(self.yaw)

        if not self.front.is_unit():
            self.front.to_unit()

        self.right= self.front.cross(maths.Vec3(y=1))
        
        if not self.right.is_unit():
            self.right.to_unit()
        
        self.up= self.right.cross(self.front)

    def projection_matrix(self) -> maths.Mat4:
        """Return projection matrix

        Returns
        ---
        maths.Mat4
        """
        return maths.Mat4.create_perspective(self.fovy, self.aspect, self.znear, self.zfar)

    def view_matrix(self) -> maths.Mat4:
        """Return view matrix

        Returns
        ---
        maths.Mat4
        """
        return maths.Mat4.create_look_at(self.position, self.position + self.front, self.up)
