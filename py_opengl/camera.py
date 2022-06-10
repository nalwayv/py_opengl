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
        self.front: maths.Vec3= maths.Vec3(z=-1.0)
        self.up: maths.Vec3= maths.Vec3(y=1.0)
        self.right: maths.Vec3= maths.Vec3(x=1.0)
        self.aspect: float= aspect
        self.fovy: float= maths.PHI
        self.yaw: float= -maths.PHI
        self.pitch: float= 0.0
        self.znear: float= 0.01
        self.zfar: float= 1000.0
        self.sensativity: float= 3.2
        self.rsensativity: float = 18.2
        self.zsensativity: float = 0.2

    def translate(self, dir: CameraDirection, dt: float) -> None:
        """Move camera

        Raises
        ---
        CameraError
        """
        match dir:
            case CameraDirection.UP:
                u: maths.Vec3= self.right.cross(self.position + self.front)
                if not u.is_unit():
                    u.to_unit()
                u.scale(self.sensativity * dt)
                self.position.add(u)

            case CameraDirection.DOWN:
                d: maths.Vec3= (self.position + self.front).cross(self.right)
                if not d.is_unit():
                    d.to_unit()
                d.scale(self.sensativity * dt)
                self.position.add(d)

            case CameraDirection.RIGHT:
                r: maths.Vec3= (self.position + self.front).cross(self.up)
                if not r.is_unit():
                    r.to_unit()
                r.scale(self.sensativity * dt)
                self.position.add(r)

            case CameraDirection.LEFT:
                l: maths.Vec3= self.up.cross(self.position + self.front)
                if not l.is_unit():
                    l.to_unit()

                l.scale(self.sensativity * dt)
                self.position.add(l)

            case CameraDirection.OUT:
                self.position.subtract(self.front * (self.sensativity * dt))

            case CameraDirection.IN:
                self.position.add(self.front * (self.sensativity * dt))

        self._update()
    
    def rotate(self, dir: CameraRotation, value: float, dt: float) -> None:
        """Rotate camera by value

        Raises
        ---
        CameraError
        """
        match dir:
            case CameraRotation.YAW:
                self.yaw -= maths.to_rad(value * self.rsensativity * dt)

            case CameraRotation.PITCH:
                self.pitch += maths.to_rad(maths.clampf(value * self.rsensativity * dt, -89.0, 89.0))

            case CameraRotation.ROLL:
                return

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

    def get_projection_matrix(self) -> maths.Mat4:
        """Return projection matrix
        """
        return maths.Mat4.create_projection(self.fovy, self.aspect, self.znear, self.zfar)

    def get_view_matrix(self) -> maths.Mat4:
        """Return view matrix
        """
        return maths.Mat4.create_look_at(self.position, self.position + self.front, self.up)

    def get_frustum(self) -> geometry.Frustum:
        result: geometry.Frustum= geometry.Frustum()
        vp: maths.Mat4= self.get_view_matrix() * self.get_projection_matrix()
       
        p0= vp.col0().xyz()
        p1= vp.col1().xyz()
        p2= vp.col2().xyz()
        p3= vp.col3().xyz()

        result.left.normal= p3 + p0
        result.right.normal= p3 - p0
        result.bottom.normal= p3 + p1
        result.top.normal= p3 - p1
        result.near.normal= p2
        result.far.normal= p3 - p2

        result.left.direction = vp.get_at(3, 3) + vp.get_at(3, 0)
        result.right.direction = vp.get_at(3, 3) - vp.get_at(3, 0)
        result.bottom.direction = vp.get_at(3, 3) + vp.get_at(3, 1)
        result.top.direction = vp.get_at(3, 3) - vp.get_at(3, 1)
        result.near.direction = vp.get_at(3, 2)
        result.far.direction = vp.get_at(3, 3) - vp.get_at(3, 3)

        result.left.to_unit()
        result.right.to_unit()
        result.top.to_unit()
        result.bottom.to_unit()
        result.near.to_unit()
        result.far.to_unit()

        return result
