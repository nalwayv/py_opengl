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


def lerp(start: float, end: float, by: float) -> float:
    '''Lerp value between start and end'''
    return start + clamp(by, 0, 1) * (end - start)


def tan(val: float) -> float:
    return math.tan(val)


def sin(val: float) -> float:
    return math.sin(val)


def cos(val: float) -> float:
    return math.cos(val)


# --- VECTOR3(X, Y, Z)

@dataclass(eq=False, slots=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def v3_copy(v3: Vec3) -> Vec3:
    '''Return a copy of passed in V3'''
    return Vec3(v3.x, v3.y, v3.z)


def v3_add(a: Vec3, b: Vec3) -> Vec3:
    '''Add v3s'''
    vx: float = a.x + b.x
    vy: float = a.y + b.y
    vz: float = a.z + b.z
    return Vec3(vx, vy, vz)


def v3_sub(a: Vec3, b: Vec3) -> Vec3:
    '''Sub v3s'''
    vx: float = a.x - b.x
    vy: float = a.y - b.y
    vz: float = a.z - b.z
    return Vec3(vx, vy, vz)


def v3_scale(v3: Vec3, by: float) -> Vec3:
    '''Scale v3 by value'''
    vx: float = v3.x * by
    vy: float = v3.y * by
    vz: float = v3.z * by
    return Vec3(vx, vy, vz)


def v3_cross(a: Vec3, b: Vec3) -> Vec3:
    '''Get the cross product of two v3s'''
    vx: float = (a.y * b.z) - (a.z * b.y)
    vy: float = (a.z * b.x) - (a.x * b.z)
    vz: float = (a.x * b.y) - (a.y * b.x)
    return Vec3(vx, vy, vz)


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
    '''Return a copy of this v3 with a unit length'''
    inv: float = inv_sqrt(v3_length(v3))
    return Vec3(v3.x*inv, v3.y*inv, v3.z*inv)


def v3_dot(a: Vec3, b: Vec3) -> float:
    '''Get the dot product of two v3s'''
    return ((a.x * b.x) +
            (a.y * b.y) +
            (a.z * b.z))


def v3_is_unit(v3: Vec3) -> bool:
    '''Check if this v3 has a unit length'''
    return is_one(v3_length_sq(v3))


def v3_is_zero(v3: Vec3) -> bool:
    '''Check if this v3 has a length of zero'''
    return is_zero(v3.x) and is_zero(v3.y) and is_zero(v3.z)


def v3_is_equil(a: Vec3, b: Vec3) -> bool:
    '''Check if v3 a is equil to v3 b'''
    return (
        is_equil(a.x, b.x) and
        is_equil(a.y, b.y) and
        is_equil(a.z, b.z))


# --- Matrix 4x4

class Mat4Err(Exception):
    '''Custom error for matrix 4x4'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, slots=True)
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


def m4_copy(m4: Mat4) -> Mat4:
    return Mat4(
            m4.ax, m4.ay, m4.az, m4.aw,
            m4.bx, m4.by, m4.bz, m4.bw,
            m4.cx, m4.cy, m4.cz, m4.cw,
            m4.dx, m4.dy, m4.dz, m4.dw)


def m4_init_scaler(v3: Vec3) -> Mat4:
    '''Init a matrix 4x4's values for a scaler matrix'''
    return Mat4(ax=v3.x, by=v3.y, cz=v3.z, dw=1.0)


def m4_init_translate(v3: Vec3) -> Mat4:
    '''Init a matrix 4x4's values for a translation matrix'''
    return Mat4(dx=v3.x, dy=v3.y, dz=v3.z, dw=1.0)


def m4_init_rotation_x(angle_deg: float) -> Mat4:
    '''Init a matrix 4x4's values for a rotation on the x axis'''
    angle_rad = to_radians(angle_deg)
    c = cos(angle_rad)
    s = sin(angle_rad)

    return Mat4(ax=1.0, by=c, bz=-s, cy=s, cz=c, dw=1.0)


def m4_init_rotation_y(angle_deg: float) -> Mat4:
    '''Init a matrix 4x4's values for a rotation on the y axis'''
    angle_rad = to_radians(angle_deg)
    c = cos(angle_rad)
    s = sin(angle_rad)

    return Mat4(ax=c, az=s, by=1.0, cx=-s, cz=c, dw=1.0)


def m4_init_rotation_z(angle_deg: float) -> Mat4:
    '''Init a matrix 4x4's values for a rotation on the z axis'''
    angle_rad = to_radians(angle_deg)
    c = cos(angle_rad)
    s = sin(angle_rad)

    return Mat4(ax=c, ay=-s, bx=s, by=c, cz=1.0, dw=1.0)


def m4_add(a: Mat4, b: Mat4) -> Mat4:
    '''Add two matrix 4x4's together'''
    ax: float = a.ax + b.ax
    ay: float = a.ay + b.ay
    az: float = a.az + b.az
    aw: float = a.aw + b.aw
    bx: float = a.bx + b.bx
    by: float = a.by + b.by
    bz: float = a.bz + b.bz
    bw: float = a.bw + b.bw
    cx: float = a.cx + b.cx
    cy: float = a.cy + b.cy
    cz: float = a.cz + b.cz
    cw: float = a.cw + b.cw
    dx: float = a.dx + b.dx
    dy: float = a.dy + b.dy
    dz: float = a.dz + b.dz
    dw: float = a.aw + b.dw

    return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw)


def m4_sub(a: Mat4, b: Mat4) -> Mat4:
    '''Subtract two matrix 4x4's'''
    ax: float = a.ax - b.ax
    ay: float = a.ay - b.ay
    az: float = a.az - b.az
    aw: float = a.aw - b.aw
    bx: float = a.bx - b.bx
    by: float = a.by - b.by
    bz: float = a.bz - b.bz
    bw: float = a.bw - b.bw
    cx: float = a.cx - b.cx
    cy: float = a.cy - b.cy
    cz: float = a.cz - b.cz
    cw: float = a.cw - b.cw
    dx: float = a.dx - b.dx
    dy: float = a.dy - b.dy
    dz: float = a.dz - b.dz
    dw: float = a.aw - b.dw

    return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw)


def m4_scale(a: Mat4, by: float) -> Mat4:
    '''Scale a matrix 4x4 by a float value'''
    ax: float = a.ax * by
    ay: float = a.ay * by
    az: float = a.az * by
    aw: float = a.aw * by
    bx: float = a.bx * by
    by: float = a.by * by
    bz: float = a.bz * by
    bw: float = a.bw * by
    cx: float = a.cx * by
    cy: float = a.cy * by
    cz: float = a.cz * by
    cw: float = a.cw * by
    dx: float = a.dx * by
    dy: float = a.dy * by
    dz: float = a.dz * by
    dw: float = a.aw * by

    return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw)


def m4_multiply(a: Mat4, b: Mat4) -> Mat4:
    '''Multiply two matrix 4x4's'''
    ax: float = (a.ax * b.ax) + (a.ay * b.bx) + (a.az * b.cx) + (a.aw * b.dx)
    ay: float = (a.ax * b.ay) + (a.ay * b.by) + (a.az * b.cy) + (a.aw * b.dy)
    az: float = (a.ax * b.az) + (a.ay * b.bz) + (a.az * b.cz) + (a.aw * b.dz)
    aw: float = (a.ax * b.aw) + (a.ay * b.bw) + (a.az * b.cw) + (a.aw * b.dw)
    bx: float = (a.bx * b.ax) + (a.by * b.bx) + (a.bz * b.cx) + (a.bw * b.dx)
    by: float = (a.bx * b.ay) + (a.by * b.by) + (a.bz * b.cy) + (a.bw * b.dy)
    bz: float = (a.bx * b.az) + (a.by * b.bz) + (a.bz * b.cz) + (a.bw * b.dz)
    bw: float = (a.bx * b.aw) + (a.by * b.bw) + (a.bz * b.cw) + (a.bw * b.dw)
    cx: float = (a.cx * b.ax) + (a.cy * b.bx) + (a.cz * b.cx) + (a.cw * b.dx)
    cy: float = (a.cx * b.ay) + (a.cy * b.by) + (a.cz * b.cy) + (a.cw * b.dy)
    cz: float = (a.cx * b.az) + (a.cy * b.bz) + (a.cz * b.cz) + (a.cw * b.dz)
    cw: float = (a.cx * b.aw) + (a.cy * b.bw) + (a.cz * b.cw) + (a.cw * b.dw)
    dx: float = (a.dx * b.ax) + (a.dy * b.bx) + (a.dz * b.cx) + (a.dw * b.dx)
    dy: float = (a.dx * b.ay) + (a.dy * b.by) + (a.dz * b.cy) + (a.dw * b.dy)
    dz: float = (a.dx * b.az) + (a.dy * b.bz) + (a.dz * b.cz) + (a.dw * b.dz)
    dw: float = (a.dx * b.aw) + (a.dy * b.bw) + (a.dz * b.cw) + (a.dw * b.dw)

    return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw)


def m4_multiply_m4s(*args) -> Mat4:
    if not len(args):
        raise Mat4Err('cant multipy no m4s!!')

    from functools import reduce
    return reduce(m4_multiply, args)


def m4_look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    '''Get matrix 4x4 look-at value'''
    z: Vec3 = v3_unit(v3_sub(eye, target))

    if v3_is_zero(z):
        return Mat4(ax=1.0, by=1.0, cz=1.0, dw=1.0)

    x: Vec3 = v3_unit(v3_cross(up, z))
    y: Vec3 = v3_unit(v3_cross(z, x))

    dx: float = v3_dot(x, eye) * -1.0
    dy: float = v3_dot(y, eye) * -1.0
    dz: float = v3_dot(z, eye) * -1.0

    return Mat4(
            x.x, y.x, z.x, 0.0,
            x.y, y.y, z.y, 0.0,
            x.z, y.z, z.z, 0.0,
            dx, dy, dz, 1.0)


def m4_from_axis(angle_deg: float, axis: Vec3) -> Mat4:
    '''Get a matrix 4x4 angle axis value'''
    x: float = axis.x
    y: float = axis.y
    z: float = axis.z

    if not v3_is_unit(axis):
        inv: float = 1.0 / v3_length(axis)
        x *= inv
        y *= inv
        z *= inv

    angle_rad = to_radians(angle_deg)
    c: float = cos(angle_rad)
    s: float = sin(angle_rad)

    x2: float = sqr(x)
    y2: float = sqr(y)
    z2: float = sqr(z)
    ww: float = 1.0 - c

    ax: float = c + x2 * ww
    ay: float = x * y * ww - z * s
    az: float = x * z * ww + y * s

    bx: float = y * x * ww + z * s
    by: float = c + y2 * ww
    bz: float = y * z * ww - x * s

    cx: float = z * x * ww - y * s
    cy: float = z * y * ww + x * s
    cz: float = c + z2 * ww

    return Mat4(
            ax, ay, az, 0.0,
            bx, by, bz, 0.0,
            cx, cy, cz, 0.0,
            0.0, 0.0, 0.0, 1.0)


def m4_frustum(
        left: float,
        right: float,
        bottom: float,
        top: float,
        far: float,
        near: float) -> Mat4:
    '''Get a frustum matrix 4x4'''
    rlInv: float = 1.0 / (right - left)
    tbInv: float = 1.0 / (top - bottom)
    fnInv: float = 1.0 / (far - near)

    x: float = 2.0 * near * rlInv
    y: float = 2.0 * near * tbInv
    z: float = -1.0
    a: float = (right + left) * rlInv
    b: float = (top + bottom) * tbInv
    c: float = -(far + near) * fnInv
    d: float = -(2.0 * far * near) * fnInv

    return Mat4(
            x, 0.0, 0.0, 0.0,
            0.0, y, 0.0, 0.0,
            a, b, c, z,
            0.0, 0.0, d, 0.0)


def m4_projection(fov: float, aspect: float, near: float, far: float) -> Mat4:
    '''Get a projection matrix 4x4'''
    if fov <= 0.0 or fov >= PI:
        raise Mat4Err('m4 projection fov out of range')

    if aspect <= 0.0 or near <= 0.0 or far <= 0.0:
        raise Mat4Err('m4 projection aspect out of range')

    top: float = near * tan(0.5 * fov)
    bottom: float = top * -1.0
    left: float = bottom * aspect
    right: float = top * aspect

    return m4_frustum(left, right, bottom, top, far, near)


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

    return (b00 - b01 - b02 + b03 +
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
            b00 * b11 -
            b01 * b10 +
            b02 * b09 +
            b03 * b08 -
            b04 * b07 +
            b05 * b06)

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


def m4_to_array(m4: Mat4) -> list[float]:
    '''Mat4x4 to list[float]'''
    return [
            m4.ax, m4.ay, m4.az, m4.aw,
            m4.bx, m4.by, m4.bz, m4.bw,
            m4.cx, m4.cy, m4.cz, m4.cw,
            m4.dx, m4.dy, m4.dz, m4.dw]


def m4_to_multi_array(m4: Mat4) -> list[list[float]]:
    '''Mat4x4 to list[list[float]]'''
    return [
            [m4.ax, m4.ay, m4.az, m4.aw],
            [m4.bx, m4.by, m4.bz, m4.bw],
            [m4.cx, m4.cy, m4.cz, m4.cw],
            [m4.dx, m4.dy, m4.dz, m4.dw]]


# --- Quaternion

@dataclass(eq=False, slots=True)
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0


def qt_copy(qt: Quaternion) -> Quaternion:
    return Quaternion(qt.x, qt.y, qt.z, qt.z)


def qt_length_sq(qt: Quaternion) -> float:
    '''Return queternion length squared'''
    x2: float = sqr(qt.x)
    y2: float = sqr(qt.y)
    z2: float = sqr(qt.z)
    w2: float = sqr(qt.w)
    return x2 + y2 + z2 + w2


def qt_unit(qt: Quaternion) -> Quaternion:
    '''Return a copy of this qt with a unit length'''
    lsq: float = qt_length_sq(qt)
    if is_zero(lsq):
        return qt_copy(qt)
    inv = inv_sqrt(lsq)
    return Quaternion(qt.x*inv, qt.y*inv, qt.z*inv, qt.w*inv)


def qt_in_unit(qt: Quaternion) -> bool:
    return is_one(qt_length_sq(qt))


def qt_from_axis(angle_deg: float, axis: Vec3) -> Quaternion:
    lsq = v3_length_sq(axis)
    if is_zero(lsq):
        return Quaternion(w=1.0)

    x = axis.x
    y = axis.y
    z = axis.z

    if not v3_is_unit(axis):
        inv = 1.0 / v3_length(axis)
        x *= inv
        y *= inv
        z *= inv

    angle_rad = to_radians(angle_deg)
    c = cos(angle_rad * 0.5)
    s = sin(angle_rad * 0.5)

    return Quaternion(x=x*s, y=y*s, z=z*s, w=c)


def qt_lerp(start: Quaternion, end: Quaternion, by: float) -> Quaternion:
    x: float = lerp(start.x, end.x, by)
    y: float = lerp(start.y, end.y, by)
    z: float = lerp(start.z, end.z, by)
    w: float = lerp(start.w, end.w, by)
    return Quaternion(x, y, z, w)


def qt_to_mat4(qt: Quaternion) -> Mat4:
    unit = qt_unit(qt)

    x2 = sqr(unit.x)
    y2 = sqr(unit.y)
    z2 = sqr(unit.z)
    w2 = sqr(unit.w)

    xy = unit.x * unit.y
    zw = unit.z * unit.w

    ax = x2 - y2 - z2 + w2
    ay = 2.0 * (xy - zw)
    az = 2.0 * (xy + zw)

    bx = 2.0 * (xy + zw)
    by = -x2 + y2 - z2 + w2
    bz = 2.0 * (xy - zw)

    cx = 2.0 * (xy - zw)
    cy = 2.0 * (xy + zw)
    cz = -x2 - y2 + z2 + w2

    return Mat4(
            ax, ay, az, 0.0,
            bx, by, bz, 0.0,
            cx, cy, cz, 0.0,
            0.0, 0.0, 0.0, 1.0)
