"""Camera
"""
from enum import Enum, auto

from py_opengl import maths
from py_opengl import geometry

# ---


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


class Camera:

    __slots__= (
        'position',
        'front',
        'up',
        'right',
        'aspect',
        'fovy',
        'yaw',
        'pitch',
        'znear',
        'zfar',
        'sensativity',
        'rsensativity',
        'zsensativity'
    )

    def __init__(self, pos: maths.Vec3, aspect: float= 1.0) -> None:
        self.position: maths.Vec3= pos
        self.front: maths.Vec3= maths.Vec3(z= -1.0)
        self.up: maths.Vec3= maths.Vec3(y= 1.0)
        self.right: maths.Vec3= maths.Vec3(x= 1.0)
        self.aspect: float= aspect
        self.fovy: float= maths.PHI
        self.yaw: float= -maths.PHI
        self.pitch: float= 0.0
        self.znear: float= 0.1
        self.zfar: float= 110.0
        self.sensativity: float= 3.2
        self.rsensativity: float = 18.2
        self.zsensativity: float = 0.2

    def translate(self, dir: CameraDirection, dt: float) -> None:
        """Move camera
        """
        match dir:
            case CameraDirection.UP:
                up: maths.Vec3= self.right.cross(self.position + self.front)
                if not up.is_normalized():
                    up.normalize()
                up= up * (self.sensativity * dt)
                self.position.set_from(self.position + up)

            case CameraDirection.DOWN:
                down: maths.Vec3= (self.position + self.front).cross(self.right)
                if not down.is_normalized():
                    down.normalize()
                down= down * (self.sensativity * dt)
                self.position.set_from(self.position + down)

            case CameraDirection.RIGHT:
                right: maths.Vec3= (self.position + self.front).cross(self.up)
                if not right.is_normalized():
                    right.normalize()
                right= right * (self.sensativity * dt)
                self.position.set_from(self.position + right)

            case CameraDirection.LEFT:
                left: maths.Vec3= self.up.cross(self.position + self.front)
                if not left.is_normalized():
                    left.normalize()
                left= left * (self.sensativity * dt)
                self.position.set_from(self.position + left)

            case CameraDirection.OUT:
                self.position.set_from(self.position - (self.front * (self.sensativity * dt)))

            case CameraDirection.IN:
                self.position.set_from(self.position + (self.front * (self.sensativity * dt)))

        self._update()

    def rotate(self, dir: CameraRotation, value: float, dt: float) -> None:
        """Rotate camera by value
        """
        match dir:
            case CameraRotation.YAW:
                by= maths.to_rad(value * self.rsensativity * dt)
                self.yaw -= by

            case CameraRotation.PITCH:
                by= maths.to_rad(maths.clampf(value * self.rsensativity * dt, -89.0, 89.0))
                self.pitch += by

            case CameraRotation.ROLL:
                return

        self._update()

    def _update(self) -> None:
        """Update camera's up, right and front fields
        """
        self.front.x= maths.cos(self.pitch) * maths.cos(self.yaw)
        self.front.y= maths.sin(self.pitch)
        self.front.z= maths.cos(self.pitch) * maths.sin(self.yaw)

        if not self.front.is_normalized():
            self.front.normalize()

        self.right= self.front.cross(maths.Vec3(y= 1.0))

        if not self.right.is_normalized():
            self.right.normalize()

        self.up= self.right.cross(self.front)

    def get_projection_matrix(self) -> maths.Mat4:
        """Return projection matrix
        """
        return maths.Mat4.create_perspective_fov(self.fovy, self.aspect, self.znear, self.zfar)

    def get_view_matrix(self) -> maths.Mat4:
        """Return view matrix
        """
        return maths.Mat4.create_lookat(self.position, self.position + self.front, self.up)

    def get_frustum(self) -> geometry.Frustum:
        """Return list of frustum planes

        [ near, far, left, right, top, bottom ]
        """
        v: maths.Mat4= self.get_view_matrix()
        p: maths.Mat4= self.get_projection_matrix()
        vp: maths.Mat4= v * p

        return geometry.Frustum.create_from_matrix(vp)

    def get_frustum_corners(self, normalize: bool=False) -> list[maths.Vec3]:
        """Return corners of camera frustum

        [ nbl, nbr, ntl, ntr, fbl, fbr, ftl, ftr ]
        """
        fr= self.get_frustum()
        return fr.get_corners(normalize)
