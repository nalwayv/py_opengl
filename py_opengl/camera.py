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
        self.zfar: float= 110.0
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
        return maths.Mat4.create_projection_rh(self.fovy, self.aspect, self.znear, self.zfar)

    def get_view_matrix(self) -> maths.Mat4:
        """Return view matrix
        """
        return maths.Mat4.create_lookat_rh(self.position, self.position + self.front, self.up)

    def get_frustum_corners(self, to_unit: bool= False) -> list[maths.Vec3]:
        """Return corners of camera frustum

        [
            Near bottom left,

            Near bottom right,

            Near top left,

            Near top right,
            
            Far bottom left,
            
            Far bottom right,
            
            Far top left,
            
            Far top right
        ]
        """
        corners: list[maths.Vec4]= [
            maths.Vec4(-1.0, -1.0, -1.0, 1.0), # n bl
            maths.Vec4( 1.0, -1.0, -1.0, 1.0), # n br
            maths.Vec4(-1.0,  1.0, -1.0, 1.0), # n tl
            maths.Vec4( 1.0,  1.0, -1.0, 1.0), # n tr
            maths.Vec4(-1.0, -1.0,  1.0, 1.0), # f bl
            maths.Vec4( 1.0, -1.0,  1.0, 1.0), # f br
            maths.Vec4(-1.0,  1.0,  1.0, 1.0), # f tl
            maths.Vec4( 1.0,  1.0,  1.0, 1.0), # f tr
        ]

        iv: maths.Mat4= self.get_view_matrix().inverse()
        ip: maths.Mat4= self.get_projection_matrix().inverse()
        inv_vp: maths.Mat4= ip * iv

        for corner in corners:
            corner.set_from(inv_vp.multiply_v4(corner))
            corner.set_from(corner * (1.0 / corner.w))
            if to_unit:
                corner.to_unit()

        return corners

    def get_frustum_planes(self, to_unit: bool= False) -> list[geometry.Plane3]:
        """Return list of frustum planes

        [ Left, Right, Top, Bottom, Near, Far ]
        """
        v: maths.Mat4= self.get_view_matrix()
        p: maths.Mat4= self.get_projection_matrix()
        vp: maths.Mat4= p * v

        l= geometry.Plane3.create_from_v4(vp.row3 + vp.row0)
        r= geometry.Plane3.create_from_v4(vp.row3 - vp.row0)
        t= geometry.Plane3.create_from_v4(vp.row3 - vp.row1)
        b= geometry.Plane3.create_from_v4(vp.row3 + vp.row1)
        n= geometry.Plane3.create_from_v4(vp.row3 + vp.row2)
        f= geometry.Plane3.create_from_v4(vp.row3 - vp.row2)

        if to_unit:
            l.to_unit()
            r.to_unit()
            t.to_unit()
            b.to_unit()
            n.to_unit()
            f.to_unit()

        return [l, r, t, b, n, f]
