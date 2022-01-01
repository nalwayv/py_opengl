'''GL Math Help
'''
import math
from dataclasses import dataclass

PI: float = 3.14159265358979323846
PIOVER2: float = 1.57079632679489661923
TAU: float = 6.28318530717958647693
EPSILON: float = 0.00000000000000022204
INFINITY: float = float('inf')
NEGATIVE_INFINITY: float = float('-inf')


def is_zero(val: float) -> bool:
    '''Is float value zero'''
    return abs(val) <= EPSILON


def is_one(val: float) -> bool:
    '''Is float value one'''
    return is_zero(val - 1.0)


def is_equil(x: float, y: float) -> bool:
    '''Are float values similar'''
    return abs(x - y) <= EPSILON


def to_radians(val: float) -> float:
    '''Convert to radians'''
    return val * 0.01745329251994329577


def to_degreese(val: float) -> float:
    '''Convert to degreese'''
    return val * 57.2957795130823208768


def sqr(val: float) -> float:
    '''Sqr float'''
    return val * val


def sqrt(val: float) -> float:
    '''Sqrt float'''
    return math.sqrt(val)


def inv_sqrt(val: float) -> float:
    '''Inverse sqrt'''
    return 1.0 / sqrt(val)


def clamp(
        val: float | int,
        low: float | int,
        high: float | int) -> float | int:
    '''Clamp value between low and high'''
    return max(low, min(val, high))


def lerp(start: float, end: float, weight: float) -> float:
    '''Lerp value between start and end'''
    return start + clamp(weight, 0, 1) * (end - start)


def normalize(val: float, low: float, high: float) -> float:
    '''Normalize value between low and high'''
    return (val - low) / (high - low)


def tan(val: float) -> float:
    '''Return tan value'''
    return math.tan(val)


def sin(val: float) -> float:
    '''Return sin value'''
    return math.sin(val)


def cos(val: float) -> float:
    '''Return cos value'''
    return math.cos(val)


def arccos(val: float) -> float:
    '''Return arccos value
    value has to be between -1 and 1
    '''
    return math.acos(val)


def arcsin(val: float) -> float:
    '''Return arcsin value
    value has to be between -1 and 1
    '''
    return math.asin(val)


def arctan2(y: float, x: float) -> float:
    '''Return arctan2 angle'''
    return math.atan2(y, x)

# --- VECTOR2(X, Y)


class Vec2Error(Exception):
    '''Custom error for Vec2'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __getitem__(self, idx):
        match clamp(idx, 0, 2):
            case 0: return self.x
            case 1: return self.y
            case _: raise Vec3Error('out of range')

    def __add__(self, other):
        if not isinstance(other, Vec2):
            raise Vec2Error('not of type Vec2')
        x: float = self.x + other.x
        y: float = self.y + other.y
        return Vec2(x, y)

    def __sub__(self, other):
        if not isinstance(other, Vec2):
            raise Vec2Error('not of type Vec2')
        x: float = self.x - other.x
        y: float = self.y - other.y
        return Vec2(x, y)

    def __mul__(self, other):
        if not isinstance(other, Vec2):
            raise Vec2Error('not of type Vec2')
        x: float = self.x * other.x
        y: float = self.y * other.y
        return Vec2(x, y)


def v2_copy(v2: Vec2) -> Vec2:
    '''Return a copy of passed in V2'''
    return Vec2(v2.x, v2.y)


def v2_one() -> Vec2:
    '''Return a v3 with all values set to one'''
    return Vec2(1.0, 1.0)


def v2_scale(v2: Vec2, by: float) -> Vec2:
    '''Scale v2 by value'''
    x: float = v2.x * by
    y: float = v2.y * by
    return Vec2(x, y)


def v2_cross(a: Vec2, b: Vec2) -> float:
    '''Get the cross product of two v2s'''
    return (a.x * b.y) - (a.y * b.x)


def v2_length_sq(v2: Vec2) -> float:
    '''Get the length sqr of this v2'''
    x2: float = sqr(v2.x)
    y2: float = sqr(v2.y)
    return x2 + y2


def v2_length(v2: Vec2) -> float:
    '''Get the length sqrt of this v2'''
    x2: float = sqr(v2.x)
    y2: float = sqr(v2.y)
    return sqrt(x2 + y2)


def v2_unit(v2: Vec2) -> Vec2:
    '''Return a cpy of this v2 with a unit length'''
    lsq = v2_length_sq(v2)
    if is_zero(lsq):
        return v2_copy(v2)
    return v2_scale(v2, inv_sqrt(lsq))


def v2_dot(a: Vec2, b: Vec2) -> float:
    '''Get the dot product of two v3s'''
    x: float = a.x * b.x
    y: float = a.y * b.y
    return x + y


def v2_is_unit(v2: Vec2) -> bool:
    '''Check if this v2 has a unit length'''
    return is_one(v2_length_sq(v2))


def v2_is_zero(v2: Vec2) -> bool:
    '''Check if this v2 has a length of zero'''
    return (
        is_zero(v2.x) and
        is_zero(v2.y))


def v2_is_equil(a: Vec2, b: Vec2) -> bool:
    '''Check if v2 a is equil to v2 b'''
    return (is_equil(a.x, b.x) and is_equil(a.y, b.y))


# --- VECTOR3(X, Y, Z)


class Vec3Error(Exception):
    '''Custom error for Vec3'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, slots=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __getitem__(self, idx):
        match clamp(idx, 0, 3):
            case 0: return self.x
            case 1: return self.y
            case 2: return self.z
            case _: raise Vec3Error('out of range')

    def __add__(self, other):
        if not isinstance(other, Vec3):
            raise Vec3Error('not of type Vec3')
        x: float = self.x + other.x
        y: float = self.y + other.y
        z: float = self.z + other.z
        return Vec3(x, y, z)

    def __sub__(self, other):
        if not isinstance(other, Vec3):
            raise Vec3Error('not of type Vec3')
        x: float = self.x - other.x
        y: float = self.y - other.y
        z: float = self.z - other.z
        return Vec3(x, y, z)

    def __mul__(self, other):
        if not isinstance(other, Vec3):
            raise Vec3Error('not of type Vec3')
        x: float = self.x * other.x
        y: float = self.y * other.y
        z: float = self.z * other.z
        return Vec3(x, y, z)


def v3_copy(v3: Vec3) -> Vec3:
    '''Return a copy of passed in V3'''
    return Vec3(v3.x, v3.y, v3.z)


def v3_one() -> Vec3:
    '''Return a v3 with all values set to one'''
    return Vec3(1.0, 1.0, 1.0)


def v3_scale(v3: Vec3, by: float) -> Vec3:
    '''Scale v3 by value'''
    x: float = v3.x * by
    y: float = v3.y * by
    z: float = v3.z * by
    return Vec3(x, y, z)


def v3_cross(a: Vec3, b: Vec3) -> Vec3:
    '''Get the cross product of two v3s'''
    x: float = (a.y * b.z) - (a.z * b.y)
    y: float = (a.z * b.x) - (a.x * b.z)
    z: float = (a.x * b.y) - (a.y * b.x)
    return Vec3(x, y, z)


def v3_length_sq(v3: Vec3) -> float:
    '''Get the length sqr of this v3'''
    x2: float = sqr(v3.x)
    y2: float = sqr(v3.y)
    z2: float = sqr(v3.z)
    return x2 + y2 + z2


def v3_length(v3: Vec3) -> float:
    '''Get the length sqrt of this v3'''
    x2: float = sqr(v3.x)
    y2: float = sqr(v3.y)
    z2: float = sqr(v3.z)
    return sqrt(x2 + y2 + z2)


def v3_unit(v3: Vec3) -> Vec3:
    '''Return a cpy of this v3 with a unit length'''
    lsq = v3_length_sq(v3)
    if is_zero(lsq):
        return v3_copy(v3)
    return v3_scale(v3, inv_sqrt(lsq))


def v3_dot(a: Vec3, b: Vec3) -> float:
    '''Get the dot product of two v3s'''
    x: float = a.x * b.x
    y: float = a.y * b.y
    z: float = a.z * b.z
    return x + y + z


def v3_is_unit(v3: Vec3) -> bool:
    '''Check if this v3 has a unit length'''
    return is_one(v3_length_sq(v3))


def v3_is_zero(v3: Vec3) -> bool:
    '''Check if this v3 has a length of zero'''
    return (
        is_zero(v3.x) and
        is_zero(v3.y) and
        is_zero(v3.z))


def v3_project(a: Vec3, b: Vec3) -> Vec3:
    '''Return a vec3 project'''
    by: float = v3_dot(a, b) / v3_length_sq(b)
    return v3_scale(b, by)


def v3_reject(a: Vec3, b: Vec3) -> Vec3:
    '''Return a vec3 reject'''
    proj = v3_project(a, b)
    return a - proj


def v3_is_equil(a: Vec3, b: Vec3) -> bool:
    '''Check if v3 a is equil to v3 b'''
    return (
        is_equil(a.x, b.x) and
        is_equil(a.y, b.y) and
        is_equil(a.z, b.z))


# --- Matrix 4x4


class Mat4Error(Exception):
    '''Custom error for matrix 4x4'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=True, slots=True)
class Mat4:
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    aw: float = 0.0
    bx: float = 0.0
    by: float = 0.0
    bz: float = 0.0
    bw: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    cz: float = 0.0
    cw: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    dw: float = 0.0

    def __getitem__(self, idx):
        match clamp(idx, 0, 15):
            case 0: return self.ax
            case 1: return self.ay
            case 2: return self.az
            case 3: return self.aw
            case 4: return self.bx
            case 5: return self.by
            case 6: return self.bz
            case 7: return self.bw
            case 8: return self.cx
            case 9: return self.cy
            case 10: return self.cz
            case 11: return self.cw
            case 12: return self.dx
            case 13: return self.dy
            case 14: return self.dz
            case 15: return self.dw
            case _: raise Mat4Error('out of range')

    def __add__(self, other):
        if not isinstance(other, Mat4):
            raise Mat4Error('not of type mat4')

        ax: float = self.ax + other.ax
        ay: float = self.ay + other.ay
        az: float = self.az + other.az
        aw: float = self.aw + other.aw
        bx: float = self.bx + other.bx
        by: float = self.by + other.by
        bz: float = self.bz + other.bz
        bw: float = self.bw + other.bw
        cx: float = self.cx + other.cx
        cy: float = self.cy + other.cy
        cz: float = self.cz + other.cz
        cw: float = self.cw + other.cw
        dx: float = self.dx + other.dx
        dy: float = self.dy + other.dy
        dz: float = self.dz + other.dz
        dw: float = self.dw + other.dw

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw)

    def __sub__(self, other):
        if not isinstance(other, Mat4):
            raise Mat4Error('not of type mat4')

        ax: float = self.ax - other.ax
        ay: float = self.ay - other.ay
        az: float = self.az - other.az
        aw: float = self.aw - other.aw
        bx: float = self.bx - other.bx
        by: float = self.by - other.by
        bz: float = self.bz - other.bz
        bw: float = self.bw - other.bw
        cx: float = self.cx - other.cx
        cy: float = self.cy - other.cy
        cz: float = self.cz - other.cz
        cw: float = self.cw - other.cw
        dx: float = self.dx - other.dx
        dy: float = self.dy - other.dy
        dz: float = self.dz - other.dz
        dw: float = self.dw - other.dw

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw)

    def __mul__(self, other):
        if not isinstance(other, Mat4):
            raise Mat4Error('not of type Mat4')
        ax = ((self.ax * other.ax) +
              (self.ay * other.bx) +
              (self.az * other.cx) +
              (self.aw * other.dx))

        ay = ((self.ax * other.ay) +
              (self.ay * other.by) +
              (self.az * other.cy) +
              (self.aw * other.dy))

        az = ((self.ax * other.az) +
              (self.ay * other.bz) +
              (self.az * other.cz) +
              (self.aw * other.dz))

        aw = ((self.ax * other.aw) +
              (self.ay * other.bw) +
              (self.az * other.cw) +
              (self.aw * other.dw))

        bx = ((self.bx * other.ax) +
              (self.by * other.bx) +
              (self.bz * other.cx) +
              (self.bw * other.dx))

        by = ((self.bx * other.ay) +
              (self.by * other.by) +
              (self.bz * other.cy) +
              (self.bw * other.dy))

        bz = ((self.bx * other.az) +
              (self.by * other.bz) +
              (self.bz * other.cz) +
              (self.bw * other.dz))

        bw = ((self.bx * other.aw) +
              (self.by * other.bw) +
              (self.bz * other.cw) +
              (self.bw * other.dw))

        cx = ((self.cx * other.ax) +
              (self.cy * other.bx) +
              (self.cz * other.cx) +
              (self.cw * other.dx))

        cy = ((self.cx * other.ay) +
              (self.cy * other.by) +
              (self.cz * other.cy) +
              (self.cw * other.dy))

        cz = ((self.cx * other.az) +
              (self.cy * other.bz) +
              (self.cz * other.cz) +
              (self.cw * other.dz))

        cw = ((self.cx * other.aw) +
              (self.cy * other.bw) +
              (self.cz * other.cw) +
              (self.cw * other.dw))

        dx = ((self.dx * other.ax) +
              (self.dy * other.bx) +
              (self.dz * other.cx) +
              (self.dw * other.dx))

        dy = ((self.dx * other.ay) +
              (self.dy * other.by) +
              (self.dz * other.cy) +
              (self.dw * other.dy))

        dz = ((self.dx * other.az) +
              (self.dy * other.bz) +
              (self.dz * other.cz) +
              (self.dw * other.dz))

        dw = ((self.dx * other.aw) +
              (self.dy * other.bw) +
              (self.dz * other.cw) +
              (self.dw * other.dw))

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw)


def m4_copy(m4: Mat4) -> Mat4:
    return Mat4(
            m4.ax, m4.ay, m4.az, m4.aw,
            m4.bx, m4.by, m4.bz, m4.bw,
            m4.cx, m4.cy, m4.cz, m4.cw,
            m4.dx, m4.dy, m4.dz, m4.dw)


def m4_row0(m4: Mat4) -> tuple[float, float, float, float]:
    return (m4.ax, m4.ay, m4.az, m4.aw)


def m4_row1(m4: Mat4) -> tuple[float, float, float, float]:
    return (m4.bx, m4.by, m4.bz, m4.bw)


def m4_row2(m4: Mat4) -> tuple[float, float, float, float]:
    return (m4.cx, m4.cy, m4.cz, m4.cw)


def m4_row3(m4: Mat4) -> tuple[float, float, float, float]:
    return (m4.dx, m4.dy, m4.dz, m4.dw)


def m4_col0(m4: Mat4) -> tuple[float, float, float, float]:
    return (m4.ax, m4.bx, m4.cx, m4.dx)


def m4_col1(m4: Mat4) -> tuple[float, float, float, float]:
    return (m4.ay, m4.by, m4.cy, m4.dy)


def m4_col2(m4: Mat4) -> tuple[float, float, float, float]:
    return (m4.az, m4.bz, m4.cz, m4.dz)


def m4_col3(m4: Mat4) -> tuple[float, float, float, float]:
    return (m4.aw, m4.bw, m4.cw, m4.dw)


def m4_create_scaler(v3: Vec3) -> Mat4:
    '''Init a matrix 4x4's values for a scaler matrix'''
    x: float = v3.x
    y: float = v3.y
    z: float = v3.z
    return Mat4(ax=x, by=y, cz=z, dw=1.0)


def m4_create_translation(v3: Vec3) -> Mat4:
    '''Init a matrix 4x4's values for a translation matrix'''
    x: float = v3.x
    y: float = v3.y
    z: float = v3.z
    return Mat4(
            ax=1.0,
            by=1.0,
            cz=1.0,
            dx=x, dy=y, dz=z, dw=1.0)


def m4_create_rotation_x(angle_deg: float) -> Mat4:
    '''Return a matrix 4x4's values for a rotation on the x axis'''
    angle_rad: float = to_radians(angle_deg)
    c: float = cos(angle_rad)
    s: float = sin(angle_rad)
    return Mat4(ax=1.0, by=c, bz=-s, cy=s, cz=c, dw=1.0)


def m4_create_rotation_y(angle_deg: float) -> Mat4:
    '''Return a matrix 4x4's values for a rotation on the y axis'''
    angle_rad: float = to_radians(angle_deg)
    c: float = cos(angle_rad)
    s: float = sin(angle_rad)
    return Mat4(ax=c, az=s, by=1.0, cx=-s, cz=c, dw=1.0)


def m4_create_rotation_z(angle_deg: float) -> Mat4:
    '''Return a matrix 4x4's values for a rotation on the z axis'''
    angle_rad: float = to_radians(angle_deg)
    c: float = cos(angle_rad)
    s: float = sin(angle_rad)
    return Mat4(ax=c, ay=-s, bx=s, by=c, cz=1.0, dw=1.0)


def m4_create_identity() -> Mat4:
    '''Return a identity matrix'''
    return Mat4(ax=1.0, by=1.0, cz=1.0, dw=1.0)


def m4_scale(m4: Mat4, by: float) -> Mat4:
    '''Scale a matrix 4x4 by a float value'''
    ax: float = m4.ax * by
    ay: float = m4.ay * by
    az: float = m4.az * by
    aw: float = m4.aw * by
    bx: float = m4.bx * by
    by: float = m4.by * by
    bz: float = m4.bz * by
    bw: float = m4.bw * by
    cx: float = m4.cx * by
    cy: float = m4.cy * by
    cz: float = m4.cz * by
    cw: float = m4.cw * by
    dx: float = m4.dx * by
    dy: float = m4.dy * by
    dz: float = m4.dz * by
    dw: float = m4.dw * by

    return Mat4(
        ax, ay, az, aw,
        bx, by, bz, bw,
        cx, cy, cz, cw,
        dx, dy, dz, dw)


def m4_transpose(m4: Mat4) -> Mat4:
    ax: float = m4.ax
    ay: float = m4.bx
    az: float = m4.cx
    aw: float = m4.dx
    bx: float = m4.ay
    by: float = m4.by
    bz: float = m4.cy
    bw: float = m4.dy
    cx: float = m4.az
    cy: float = m4.bz
    cz: float = m4.cz
    cw: float = m4.dz
    dx: float = m4.aw
    dy: float = m4.bw
    dz: float = m4.cw
    dw: float = m4.dw

    return Mat4(
        ax, ay, az, aw,
        bx, by, bz, bw,
        cx, cy, cz, cw,
        dx, dy, dz, dw)


def m4_look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    '''Get matrix 4x4 look-at value'''
    z: Vec3 = v3_unit(eye - target)

    if v3_is_zero(z):
        return m4_create_identity()

    x: Vec3 = v3_unit(v3_cross(up, z))
    y: Vec3 = v3_unit(v3_cross(z, x))

    dx: float = -v3_dot(x, eye)
    dy: float = -v3_dot(y, eye)
    dz: float = -v3_dot(z, eye)

    return Mat4(
            x.x, y.x, z.x, 0.0,
            x.y, y.y, z.y, 0.0,
            x.z, y.z, z.z, 0.0,
            dx, dy, dz, 1.0)


def m4_from_axis(angle_deg: float, axis: Vec3) -> Mat4:
    '''Get a matrix 4x4 angle axis value'''
    axis_cpy: Vec3 = v3_copy(axis)

    if not v3_is_unit(axis_cpy):
        inv: float = inv_sqrt(v3_length_sq(axis_cpy))
        axis_cpy = v3_scale(axis_cpy, inv)

    angle_rad: float = to_radians(angle_deg)
    c: float = cos(angle_rad)
    s: float = sin(angle_rad)

    x2: float = sqr(axis_cpy.x)
    y2: float = sqr(axis_cpy.y)
    z2: float = sqr(axis_cpy.z)
    ww: float = 1.0 - c

    ax: float = c + x2 * ww
    ay: float = axis_cpy.x * axis_cpy.y * ww - axis_cpy.z * s
    az: float = axis_cpy.x * axis_cpy.z * ww + axis_cpy.y * s

    bx: float = axis_cpy.y * axis_cpy.x * ww + axis_cpy.z * s
    by: float = c + y2 * ww
    bz: float = axis_cpy.y * axis_cpy.z * ww - axis_cpy.x * s

    cx: float = axis_cpy.z * axis_cpy.x * ww - axis_cpy.y * s
    cy: float = axis_cpy.z * axis_cpy.y * ww + axis_cpy.x * s
    cz: float = c + z2 * ww

    return Mat4(
        ax=ax, ay=ay, az=az,
        bx=bx, by=by, bz=bz,
        cx=cx, cy=cy, cz=cz,
        dw=1.0)


def m4_frustum(
        left: float,
        right: float,
        bottom: float,
        top: float,
        far: float,
        near: float) -> Mat4:
    ''' '''
    rl: float = 1.0 / (right - left)
    tb: float = 1.0 / (top - bottom)
    fn: float = 1.0 / (far - near)

    x: float = 2.0 * near * rl
    y: float = 2.0 * near * tb
    a: float = (right + left) * rl
    b: float = (top + bottom) * tb
    c: float = -(far + near) * fn
    d: float = -(2.0 * far * near) * fn

    return Mat4(
            ax=x,
            by=y,
            cx=a, cy=b, cz=c, cw=-1.0,
            dz=d, dw=0.0)


def m4_ortho(
        left: float,
        right: float,
        bottom: float,
        top: float,
        near: float,
        far: float) -> Mat4:
    ''' '''
    lr: float = 1.0 / (left - right)
    bt: float = 1.0 / (bottom - top)
    nf: float = 1.0 / (near - far)

    x: float = -2.0 * lr
    y: float = -2.0 * bt
    z: float = 2.0 * nf

    a: float = (right + left) * lr
    b: float = (top + bottom) * bt
    c: float = (far + near) * nf

    return Mat4(
            ax=x,
            by=y,
            cz=z,
            dx=a, dy=b, dz=c, dw=1.0)


def m4_projection(
        fov: float,
        aspect: float,
        znear: float,
        zfar: float) -> Mat4:
    #if fov <= 0.0 or fov >= PI:
    #    raise Mat4Error('m4 projection fov out of range')

    #if znear <= 0.0 or zfar <= 0.0 or znear >= zfar:
    #    raise Mat4Error('m4 projection aspect out of range')

    fovy: float = 1.0 / tan(fov * 0.5)
    fovx: float = fovy / aspect
    zrange: float = zfar / (zfar - znear)

    return Mat4(ax=fovx, by=fovy, cz=zrange, cw=-1.0, dz=znear*zrange)


def m4_determinant(m4: Mat4) -> float:
    '''Get determinant if matrix 4x4'''
    a00: float = m4.ax
    a01: float = m4.ay
    a02: float = m4.az
    a03: float = m4.aw
    a10: float = m4.bx
    a11: float = m4.by
    a12: float = m4.bz
    a13: float = m4.bw
    a20: float = m4.cx
    a21: float = m4.cy
    a22: float = m4.cz
    a23: float = m4.cw
    a30: float = m4.dx
    a31: float = m4.dy
    a32: float = m4.dz
    a33: float = m4.dw

    b00: float = a30 * a21 * a12 * a03
    b01: float = a20 * a31 * a12 * a03
    b02: float = a30 * a11 * a22 * a03
    b03: float = a10 * a31 * a22 * a03
    b10: float = a20 * a11 * a32 * a03
    b11: float = a10 * a21 * a32 * a03
    b12: float = a30 * a21 * a02 * a13
    b13: float = a20 * a31 * a02 * a13
    b20: float = a30 * a01 * a22 * a13
    b21: float = a00 * a31 * a22 * a13
    b22: float = a20 * a01 * a32 * a13
    b23: float = a00 * a21 * a32 * a13
    b30: float = a30 * a11 * a02 * a23
    b31: float = a10 * a31 * a02 * a23
    b32: float = a30 * a01 * a12 * a23
    b33: float = a00 * a31 * a12 * a23
    b40: float = a10 * a01 * a32 * a23
    b41: float = a00 * a11 * a32 * a23
    b42: float = a20 * a11 * a02 * a33
    b43: float = a10 * a21 * a02 * a33
    b50: float = a20 * a01 * a12 * a33
    b51: float = a00 * a21 * a12 * a33
    b52: float = a10 * a01 * a22 * a33
    b53: float = a00 * a11 * a22 * a33

    return(
        b00 - b01 - b02 + b03 +
        b10 - b11 - b12 + b13 +
        b20 - b21 - b22 + b23 +
        b30 - b31 - b32 + b33 +
        b40 - b41 - b42 + b43 +
        b50 - b51 - b52 + b53)


def m4_inverse(m4: Mat4) -> Mat4:
    '''Get inverse matrix 4x4'''
    a00: float = m4.ax
    a01: float = m4.ay
    a02: float = m4.az
    a03: float = m4.aw
    a10: float = m4.bx
    a11: float = m4.by
    a12: float = m4.bz
    a13: float = m4.bw
    a20: float = m4.cx
    a21: float = m4.cy
    a22: float = m4.cz
    a23: float = m4.cw
    a30: float = m4.dx
    a31: float = m4.dy
    a32: float = m4.dz
    a33: float = m4.dw

    b00: float = a00 * a11 - a01 * a10
    b01: float = a00 * a12 - a02 * a10
    b02: float = a00 * a13 - a03 * a10
    b03: float = a01 * a12 - a02 * a11
    b04: float = a01 * a13 - a03 * a11
    b05: float = a02 * a13 - a03 * a12
    b06: float = a20 * a31 - a21 * a30
    b07: float = a20 * a32 - a22 * a30
    b08: float = a20 * a33 - a23 * a30
    b09: float = a21 * a32 - a22 * a31
    b10: float = a21 * a33 - a23 * a31
    b11: float = a22 * a33 - a23 * a32

    det: float = (
            b00 * b11 - b01 *
            b10 + b02 * b09 +
            b03 * b08 - b04 *
            b07 + b05 * b06)

    inv: float = 1.0 / det

    ax: float = (a11 * b11 - a12 * b10 + a13 * b09) * inv
    ay: float = (-a01 * b11 + a02 * b10 - a03 * b09) * inv
    az: float = (a31 * b05 - a32 * b04 + a33 * b03) * inv
    aw: float = (-a21 * b05 + a22 * b04 - a23 * b03) * inv
    bx: float = (-a10 * b11 + a12 * b08 - a13 * b07) * inv
    by: float = (a00 * b11 - a02 * b08 + a03 * b07) * inv
    bz: float = (-a30 * b05 + a32 * b02 - a33 * b01) * inv
    bw: float = (a20 * b05 - a22 * b02 + a23 * b01) * inv
    cx: float = (a10 * b10 - a11 * b08 + a13 * b06) * inv
    cy: float = (-a00 * b10 + a01 * b08 - a03 * b06) * inv
    cz: float = (a30 * b04 - a31 * b02 + a33 * b00) * inv
    cw: float = (-a20 * b04 + a21 * b02 - a23 * b00) * inv
    dx: float = (-a10 * b09 + a11 * b07 - a12 * b06) * inv
    dy: float = (a00 * b09 - a01 * b07 + a02 * b06) * inv
    dz: float = (-a30 * b03 + a31 * b01 - a32 * b00) * inv
    dw: float = (a20 * b03 - a21 * b01 + a22 * b00) * inv

    return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw)


def m4_unit(m4: Mat4) -> Mat4:
    '''Mat4x4 unit'''
    det: float = m4_determinant(m4)

    if is_zero(det):
        return m4_copy(m4)

    return m4_scale(m4, 1.0 / det)


def m4_array(m4: Mat4) -> list[float]:
    '''Mat4x4 to list[float]'''
    return [
            m4.ax, m4.ay, m4.az, m4.aw,
            m4.bx, m4.by, m4.bz, m4.bw,
            m4.cx, m4.cy, m4.cz, m4.cw,
            m4.dx, m4.dy, m4.dz, m4.dw]


def m4_multi_array(m4: Mat4) -> list[list[float]]:
    '''Mat4x4 to list[list[float]]'''
    return [
            [m4.ax, m4.ay, m4.az, m4.aw],
            [m4.bx, m4.by, m4.bz, m4.bw],
            [m4.cx, m4.cy, m4.cz, m4.cw],
            [m4.dx, m4.dy, m4.dz, m4.dw]]


# --- Quaternion


class QuatError(Exception):
    '''Custom error for quaternion'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, slots=True)
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0

    def __getitem__(self, idx):
        match clamp(idx, 0, 4):
            case 0: return self.x
            case 1: return self.y
            case 2: return self.z
            case 3: return self.w
            case _: raise QuatError('out of range')

    def __add_(self, other):
        if not isinstance(other, Quaternion):
            raise QuatError('not of type Quaternion')
        x: float = self.x + other.x
        y: float = self.y + other.y
        z: float = self.z + other.z
        w: float = self.w + other.w
        return Quaternion(x, y, z, w)

    def __sub_(self, other):
        if not isinstance(other, Quaternion):
            raise QuatError('not of type Quaternion')
        x: float = self.x - other.x
        y: float = self.y - other.y
        z: float = self.z - other.z
        w: float = self.w - other.w
        return Quaternion(x, y, z, w)

    def __mul__(self, other):
        if not isinstance(other, Quaternion):
            raise QuatError('not of type Quaternion')

        x1: float = self.x
        y1: float = self.y
        z1: float = self.z
        w1: float = self.w

        x2: float = other.x
        y2: float = other.y
        z2: float = other.z
        w2: float = other.w

        cx: float = y1 * z2 - z1 * y2
        cy: float = z1 * x2 - x1 * z2
        cz: float = x1 * y2 - y1 * x2

        dt: float = x1 * x2 + y1 * y2 + z1 * z2

        x: float = x1 * w2 + x2 * w1 + cx
        y: float = y1 * w2 + y2 * w1 + cy
        z: float = z1 * w2 + z2 * w1 + cz
        w: float = w1 * w2 - dt

        return Quaternion(x, y, z, w)


def qt_copy(qt: Quaternion) -> Quaternion:
    '''Return a copy of this quaternion'''
    return Quaternion(qt.x, qt.y, qt.z, qt.z)


def qt_length_sq(qt: Quaternion) -> float:
    '''Return quaternion length squared'''
    x2: float = sqr(qt.x)
    y2: float = sqr(qt.y)
    z2: float = sqr(qt.z)
    w2: float = sqr(qt.w)
    return x2 + y2 + z2 + w2


def qt_length(qt: Quaternion) -> float:
    '''Return quaternion length'''
    x2: float = sqr(qt.x)
    y2: float = sqr(qt.y)
    z2: float = sqr(qt.z)
    w2: float = sqr(qt.w)
    return sqrt(x2 + y2 + z2 + w2)


def qt_conjugate(qt: Quaternion) -> Quaternion:
    '''Return a conjugate of this quaternion'''
    return Quaternion(-qt.x, -qt.y, -qt.z, qt.w)


def qt_inverse(qt: Quaternion) -> Quaternion:
    '''Return the inverse of this quaternion'''
    inv: float = 1.0 / qt_length_sq(qt)
    x: float = -qt.x * inv
    y: float = -qt.y * inv
    z: float = -qt.z * inv
    w: float = qt.w * inv
    return Quaternion(x, y, z, w)


def qt_scale(qt: Quaternion, by: float) -> Quaternion:
    '''Return a scaled quaternion by float value'''
    x = qt.x * by
    y = qt.y * by
    z = qt.z * by
    w = qt.w * by
    return Quaternion(x, y, z, w)


def qt_dot(a: Quaternion, b: Quaternion) -> float:
    '''Return quaternion dot product'''
    return (
        (a.x * b.x) +
        (a.y * b.y) +
        (a.z * b.z) +
        (a.w * b.w))


def qt_get_angle(qt: Quaternion) -> float:
    return 2.0 * arccos(qt.w)


def qt_is_equil(a: Quaternion, b: Quaternion) -> bool:
    '''Check if Quaternion a is equil to Quaternion b'''
    return (
        is_equil(a.x, b.x) and
        is_equil(a.y, b.y) and
        is_equil(a.z, b.z) and
        is_equil(a.w, b.w))


def qt_unit(qt: Quaternion) -> Quaternion:
    '''Return a copy of this qt with a unit length'''
    lsq: float = qt_length_sq(qt)
    if is_zero(lsq):
        return qt_copy(qt)
    return qt_scale(qt, inv_sqrt(lsq))


def qt_is_unit(qt: Quaternion) -> bool:
    '''Check if this quaternion is of unit length'''
    return is_one(qt_length_sq(qt))


def qt_from_euler(angles: Vec3) -> Quaternion:
    '''Create Quaternion from euler angles'''
    rx = to_radians(angles.x)
    ry = to_radians(angles.y)
    rz = to_radians(angles.z)

    c1 = cos(rx * 0.5)
    c2 = cos(ry * 0.5)
    c3 = cos(rz * 0.5)
    s1 = sin(rx * 0.5)
    s2 = sin(ry * 0.5)
    s3 = sin(rz * 0.5)

    x = (s1 * c2 * c3) + (c1 * s2 * s3)
    y = (c1 * s2 * c3) - (s1 * c2 * s3)
    z = (c1 * c2 * s3) + (s1 * s2 * c3)
    w = (c1 * c2 * c3) - (s1 * s2 * s3)

    return Quaternion(x, y, z, w)


def qt_from_axis(angle_deg: float, axis: Vec3) -> Quaternion:
    '''Create Quaternion from axis angles'''
    lsq: float = v3_length_sq(axis)
    if is_zero(lsq):
        return Quaternion(w=1.0)

    ax: float = axis.x
    ay: float = axis.y
    az: float = axis.z

    if not v3_is_unit(axis):
        inv = inv_sqrt(v3_length_sq(axis))
        ax *= inv
        ay *= inv
        az *= inv

    angle_rad: float = to_radians(angle_deg * 0.5)
    c: float = cos(angle_rad)
    s: float = sin(angle_rad)

    x: float = ax * s
    y: float = ay * s
    z: float = az * s
    w: float = c

    return Quaternion(x, y, z, w)


def qt_lerp(start: Quaternion, end: Quaternion, weight: float) -> Quaternion:
    '''Return a lerped quaternion'''

    t: float = weight
    t1: float = 1.0 - t

    dot = qt_dot(start, end)

    x, y, z, w = 0.0, 0.0, 0.0, 0.0
    if is_zero(dot):
        x = t1 * start.x + t * end.x
        y = t1 * start.y + t * end.y
        z = t1 * start.z + t * end.z
        w = t1 * start.w + t * end.w
    else:
        x = t1 * start.x - t * end.x
        y = t1 * start.y - t * end.y
        z = t1 * start.z - t * end.z
        w = t1 * start.w - t * end.w

    return qt_unit(Quaternion(x, y, z, w))


def qt_slerp(start: Quaternion, end: Quaternion, weight: float) -> Quaternion:
    '''Return a slerp quaternion'''

    t = weight
    dot: float = qt_dot(start, end)
    flip: bool = False

    if dot < 0.0:
        flip = True
        dot *= -1.0

    s1, s2 = 0.0, 0.0
    if dot > (1.0 - EPSILON):
        s1 = 1.0 - t
        s2 = -t if flip else t
    else:
        o: float = arccos(dot)
        inv: float = 1.0 / sin(o)
        s1 = sin((1.0 - t) * o) * inv
        s2 = -sin(t * o) * inv if flip else sin(t * o) * inv

    x: float = s1 * start.x + s2 * end.x
    y: float = s1 * start.y + s2 * end.y
    z: float = s1 * start.z + s2 * end.z
    w: float = s1 * start.w + s2 * end.w

    return Quaternion(x, y, z, w)


def qt_to_euler(qt: Quaternion) -> Vec3:
    threshold: float = 0.4999995

    w2: float = sqr(qt.w)
    x2: float = sqr(qt.x)
    y2: float = sqr(qt.y)
    z2: float = sqr(qt.z)

    len_sq: float = x2 + y2 + z2 + w2
    test: float = (qt.x * qt.z) + (qt.w * qt.y)

    if test > threshold * len_sq:
        return Vec3(0.0, PIOVER2, 2.0 * arctan2(qt.x, qt.y))

    if test < -threshold * len_sq:
        return Vec3(0.0, -PIOVER2, -2.0 * arctan2(qt.x, qt.w))

    x: float = arctan2(
            2.0 * ((qt.w * qt.x) - (qt.y * qt.z)),
            w2 - x2 - y2 + z2)
    y: float = arcsin(2.0 * test / len_sq)
    z: float = arctan2(
            2.0 * ((qt.w * qt.z) - (qt.z * qt.y)),
            w2 + x2 - y2 - z2)

    return Vec3(x, y, z)


def qt_to_axis(qt: Quaternion) -> Vec3:
    qt_cpy = qt_copy(qt)

    if qt_cpy.w > 1.0:
        qt_cpy = qt_unit(qt_cpy)

    scl = sqrt(1.0 - sqr(qt_cpy.w))

    if is_zero(scl):
        return Vec3(qt_cpy.x, qt_cpy.y, qt_cpy.z)

    d = 1.0 / scl
    x = qt.x * d
    y = qt.y * d
    z = qt.z * d

    return Vec3(x, y, z)


def qt_to_mat4(qt: Quaternion) -> Mat4:
    '''Convert a quaternion into a matrix 4x4'''
    x2: float = sqr(qt.x)
    y2: float = sqr(qt.y)
    z2: float = sqr(qt.z)
    w2: float = sqr(qt.w)

    xy: float = qt.x * qt.y
    xz: float = qt.x * qt.z
    xw: float = qt.x * qt.w

    yz: float = qt.y * qt.z
    yw: float = qt.y * qt.w

    zw: float = qt.z * qt.w

    s2: float = 2.0 / (x2 + y2 + z2 + w2)

    ax: float = 1.0 - (s2 * (y2 + z2))
    ay: float = s2 * (xy + zw)
    az: float = s2 * (xz - yw)
    bx: float = s2 * (xy - zw)
    by: float = 1.0 - (s2 * (x2 + z2))
    bz: float = s2 * (yz + xw)
    cx: float = s2 * (xz + yw)
    cy: float = s2 * (yz - xw)
    cz: float = 1.0 - (s2 * (x2 + y2))

    return Mat4(
        ax, ay, az, 0.0,
        bx, by, bz, 0.0,
        cx, cy, cz, 0.0,
        0.0, 0.0, 0.0, 1.0)
