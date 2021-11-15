'''
'''
import glfw
import ctypes

from math import tan, cos, sin
from OpenGL import GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
from dataclasses import dataclass, field
from typing import Any


# --- GLOBALS


SCREEN_WIDTH: int = 500
SCREEN_HEIGHT: int = 500


# --- HELPERS


def to_c_array(arr: list[float]):
    ''' '''
    # Example:
    # arr = (ctypes.c_float * 10)
    return (gl.GLfloat * len(arr))(*arr)


# --- C TYPES


NULL_PTR = ctypes.c_void_p(0)
FLOAT_SIZE = ctypes.sizeof(gl.GLfloat)


# --- MATH HELPERS


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
    '''Are float value x and y the same'''
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
    return val ** 0.5


def inv_sqrt(val: float) -> float:
    '''Inverse sqrt'''
    return 1.0 / sqrt(val)


def clamp(val: float, low: float, high: float) -> float:
    '''Clamp value between low and high'''
    return max(low, min(val, high))


# --- V3

@dataclass(eq=False, slots=True)
class V3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def v3_copy(v3: V3) -> V3:
    '''Return a copy of passed in V3'''
    return V3(v3.x, v3.y, v3.z)


def v3_add(a: V3, b: V3) -> V3:
    '''V3 add'''
    vx: float = a.x + b.x
    vy: float = a.y + b.y
    vz: float = a.z + b.z
    return V3(vx, vy, vz)


def v3_sub(a: V3, b: V3) -> V3:
    '''V3 sub'''
    vx: float = a.x - b.x
    vy: float = a.y - b.y
    vz: float = a.z - b.z
    return V3(vx, vy, vz)


def v3_scale(a: V3, by: float) -> V3:
    '''V3 scale'''
    vx: float = a.x * by
    vy: float = a.y * by
    vz: float = a.z * by
    return V3(vx, vy, vz)


def v3_cross(a: V3, b: V3) -> V3:
    '''V3 cross product'''
    vx: float = (a.y * b.z) - (a.z * b.y)
    vy: float = (a.z * b.x) - (a.x * b.z)
    vz: float = (a.x * b.y) - (a.y * b.x)
    return V3(vx, vy, vz)


def v3_unit(a: V3) -> V3:
    '''V3 unit length'''
    inv: float = inv_sqrt(v3_length_sq(a))
    return v3_scale(a, inv)


def v3_dot(a: V3, b: V3) -> float:
    '''V3 dot product'''
    return ((a.x * b.x) +
            (a.y * b.y) +
            (a.z * b.z))


def v3_length_sq(a: V3) -> float:
    '''V3 length sqr'''
    return sqr(a.x) + sqr(a.y) + sqr(a.z)


def v3_length_sqrt(a: V3) -> float:
    '''V3 length sqrt'''
    return sqrt(sqr(a.x) + sqr(a.y) + sqr(a.z))


def v3_equil(a: V3, b: V3) -> bool:
    '''V3 is equil'''
    return (is_equil(a.x, b.x) and
            is_equil(a.y, b.y) and
            is_equil(a.z, b.z))


# --- M4

class M4Err(Exception):
    '''Custom error for matrix 4x4'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, slots=True)
class M4:
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


def m4_init_scaler(v3: V3) -> M4:
    '''Init a matrix 4x4's values for a scaler matrix'''
    return M4(ax=v3.x, by=v3.y, cz=v3.z, dw=1.0)


def m4_init_translate(v3: V3) -> M4:
    '''Init a matrix 4x4's values for a translation matrix'''
    return M4(dx=v3.x, dy=v3.y, dz=v3.z, dw=1.0)


def m4_init_rotation_x(angle_radians: float):
    '''Init a matrix 4x4's values for a rotation on the x axis'''
    c = cos(angle_radians)
    s = sin(angle_radians)

    return M4(ax=1.0, by=c, bz=-s, cy=s, cz=c, dw=1.0)


def m4_init_rotation_y(angle_radians: float) -> M4:
    '''Init a matrix 4x4's values for a rotation on the y axis'''
    c = cos(angle_radians)
    s = sin(angle_radians)

    return M4(ax=c, az=s, by=1.0, cx=-s, cz=c, dw=1.0)


def m4_init_rotation_z(angle_radians: float) -> M4:
    '''Init a matrix 4x4's values for a rotation on the z axis'''
    c = cos(angle_radians)
    s = sin(angle_radians)

    return M4(ax=c, ay=-s, bx=s, by=c, cz=1.0, dw=1.0)


def m4_add(a: M4, b: M4) -> M4:
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

    return M4(ax, ay, az, aw,
              bx, by, bz, bw,
              cx, cy, cz, cw,
              dx, dy, dz, dw)


def m4_sub(a: M4, b: M4) -> M4:
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

    return M4(ax, ay, az, aw,
              bx, by, bz, bw,
              cx, cy, cz, cw,
              dx, dy, dz, dw)


def m4_scale(a: M4, by: float) -> M4:
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

    return M4(ax, ay, az, aw,
              bx, by, bz, bw,
              cx, cy, cz, cw,
              dx, dy, dz, dw)


def m4_multiply(a: M4, b: M4) -> M4:
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

    return M4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw)


def m4_multiply_m4s(*args) -> M4:
    if not len(args):
        raise M4Err('cant multipy no m4s!!')

    from functools import reduce
    return reduce(m4_multiply, args)


def m4_look_at(eye: V3, target: V3, up: V3) -> M4:
    '''Get matrix 4x4 look-at value'''
    z: V3 = v3_unit(v3_sub(eye, target))

    if is_zero(z.x) and is_zero(z.y) and is_zero(z.z):
        return M4(ax=1.0, by=1.0, cz=1.0, dw=1.0)

    x: V3 = v3_unit(v3_cross(up, z))
    y: V3 = v3_unit(v3_cross(z, x))

    dx: float = v3_dot(x, eye) * -1.0
    dy: float = v3_dot(y, eye) * -1.0
    dz: float = v3_dot(z, eye) * -1.0

    return M4(
            x.x, y.x, z.x, 0.0,
            x.y, y.y, z.y, 0.0,
            x.z, y.z, z.z, 0.0,
            dx, dy, dz, 1.0)


def m4_from_axis(ang_radians: float, axis: V3) -> M4:
    '''Get a matrix 4x4 angle axis value'''
    x: float = axis.x
    y: float = axis.y
    z: float = axis.z

    if not is_one(v3_length_sq(axis)):
        inv: float = 1.0 / v3_length_sqrt(axis)
        x *= inv
        y *= inv
        z *= inv

    c: float = cos(ang_radians)
    s: float = sin(ang_radians)

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

    return M4(
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
        near: float) -> M4:
    '''Get a frustum matrix 4x4 '''
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

    return M4(
            x, 0.0, 0.0, 0.0,
            0.0, y, 0.0, 0.0,
            a, b, c, z,
            0.0, 0.0, d, 0.0)


def m4_projection(fov: float, aspect: float, near: float, far: float) -> M4:
    '''Get a projection matrix 4x4 '''
    if fov <= 0.0 or fov >= PI:
        raise M4Err('m4 projection fov out of range')

    if aspect <= 0.0 or near <= 0.0 or far <= 0.0:
        raise M4Err('m4 projection aspect out of range')

    top: float = near * tan(0.5 * fov)
    bottom: float = top * -1.0
    left: float = bottom * aspect
    right: float = top * aspect

    return m4_frustum(left, right, bottom, top, far, near)


def m4_determinant(m4: M4) -> float:
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


def m4_inverse(m4: M4) -> M4:
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

    return M4(
        ax, ay, az, aw,
        bx, by, bz, bw,
        cx, cy, cz, cw,
        dx, dy, dz, dw)


def m4_unit(m4: M4) -> M4:
    '''M4 unit'''
    det: float = m4_determinant(m4)
    if is_zero(det):
        return m4

    return m4_scale(m4, 1.0 / det)


def m4_to_array(m4: M4) -> list[float]:
    return [
            m4.ax, m4.ay, m4.az, m4.aw,
            m4.bx, m4.by, m4.bz, m4.bw,
            m4.cx, m4.cy, m4.cz, m4.cw,
            m4.dx, m4.dy, m4.dz, m4.dw]


def m4_to_multi_array(m4: M4) -> list[list[float]]:
    return [
            [m4.ax, m4.ay, m4.az, m4.aw],
            [m4.bx, m4.by, m4.bz, m4.bw],
            [m4.cx, m4.cy, m4.cz, m4.cw],
            [m4.dx, m4.dy, m4.dz, m4.dw]]


# --- TRANSFORM


@dataclass(eq=False, repr=False, slots=True)
class Transform:
    position: V3 = V3()
    scale: V3 = V3(1.0, 1.0, 1.0)
    angle_radians: float = 0.0


def transform_get_transform_m4(trans: Transform) -> M4:
    '''Get transform matrix 4x4'''
    r: M4 = m4_from_axis(trans.angle_radians, V3(z=1.0))
    t: M4 = m4_init_translate(trans.position)
    s: M4 = m4_init_scaler(trans.scale)

    return m4_multiply_m4s(r, t, s)


def transform_get_inv_transform_m4(trans: Transform) -> M4:
    '''Get inverse transform matrix 4x4'''
    return m4_inverse(transform_get_transform_m4(trans))


# --- CLOCK


@dataclass(eq=False, repr=False, slots=True)
class Clock:
    ticks: int = 0
    delta: float = 1.0 / 60.0
    last_time_step: float = 0.0
    accumalate: float = 0.0


def clock_update(clock: Clock) -> None:
    '''Update clock '''
    current_time_step: float = glfw.get_time()
    elapsed: float = current_time_step - clock.last_time_step
    clock.last_time_step = current_time_step
    clock.accumalate += elapsed

    while clock.accumalate >= clock.delta:
        clock.accumalate -= clock.delta
        clock.ticks += 1.0


# --- CUBE:
# TODO(14/11/1021) ...
@dataclass(eq=False, repr=False, slots=True)
class Cube:
    width: float = 1.0
    height: float = 1.0
    depth: float = 1.0
    verts: list[float] = field(default_factory=list)

    points: int = 8
    data_size: int = 108
    components: int = 3

    def __post_init__(self):
        w: float = self.width / 2.0
        h: float = self.height / 2.0
        d: float = self.depth / 2.0

        # point data
        p0 = [-w, -h, -d]
        p1 = [w, -h, -d]
        p2 = [-w, h, -d]
        p3 = [w, h, -d]
        p4 = [-w, -h, d]
        p5 = [w, -h, d]
        p6 = [-w, h, d]
        p7 = [w, h, d]

        points = [
                p5, p1, p3,
                p5, p3, p7,
                p0, p4, p6,
                p0, p6, p2,
                p6, p7, p3,
                p6, p3, p2,
                p0, p1, p5,
                p0, p5, p4,
                p4, p5, p7,
                p4, p7, p6,
                p1, p0, p2,
                p1, p2, p3]

        # flattern array
        from functools import reduce
        from operator import iconcat
        self.verts = reduce(iconcat, points, [])

# --- SHADER


@dataclass(eq=False, repr=False, match_args=False, slots=True)
class Shader:
    program_id: int = 0

    def __post_init__(self):
        self.program_id = gl.glCreateProgram()


def shader_default(shader: Shader) -> None:
    '''Simple Shader'''

    vert: str = '''#version 330 core
    layout (location = 0) in vec3 a_position;

    out vec3 b_col;

    uniform vec3 color;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        b_col = color;
        mat4 mvp = projection * view * model;
        gl_Position = mvp * vec4(a_position, 1.0);
    }
    '''

    frag: str = '''#version 330 core

    in vec3 b_col;
    out vec4 c_col;

    void main () {
        c_col = vec4(b_col, 1.0);
    }
    '''

    shader.program_id = compileProgram(
        compileShader(vert, gl.GL_VERTEX_SHADER),
        compileShader(frag, gl.GL_FRAGMENT_SHADER)
    )


def shader_clean(shader: Shader) -> None:
    ''' '''
    gl.glDeleteProgram(shader.program_id)


def shader_use(shader: Shader) -> None:
    ''' '''
    gl.glUseProgram(shader.program_id)


def shader_set_vec3(shader: Shader, var_name: str, data: V3) -> None:
    ''' '''
    location_id = gl.glGetUniformLocation(shader.program_id, var_name)
    gl.glUniform3f(location_id, data.x, data.y, data.z)


def shader_set_m4(shader: Shader, var_name: str, data: M4) -> None:
    ''' '''
    location_id = gl.glGetUniformLocation(shader.program_id, var_name)
    gl.glUniformMatrix4fv(location_id, 1, gl.GL_FALSE, m4_to_multi_array(data))


# --- VBO

@dataclass(eq=False, repr=False, slots=True)
class Vbo:
    vao: int = 0
    data_size: int = 1   # len of data passed in
    components: int = 3     # x y z
    normalized: bool = False
    vbos: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.vao = gl.glGenVertexArrays(1)


def vbo_clean(vbo: Vbo) -> None:
    '''Clean vbo'''
    gl.glDeleteVertexArrays(1, vbo.vao)
    for v in vbo.vbos:
        gl.glDeleteBuffers(1, v)


def vbo_add_data(vbo: Vbo, arr: list[float]) -> None:
    '''Add data to vbo'''
    v_buffer = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, v_buffer)
    gl.glBindVertexArray(vbo.vao)

    vbo.vbos.append(v_buffer)

    normal = gl.GL_TRUE if vbo.normalized else gl.GL_FALSE

    gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            len(arr) * FLOAT_SIZE,
            to_c_array(arr),
            gl.GL_STATIC_DRAW
    )

    gl.glVertexAttribPointer(
            len(vbo.vbos) - 1,
            vbo.components,
            gl.GL_FLOAT,
            normal,
            0,
            NULL_PTR
    )

    gl.glEnableVertexAttribArray(len(vbo.vbos) - 1)


def vbo_use(vbo: Vbo) -> None:
    '''Bind vbo to vertex Array'''

    count = vbo.data_size // vbo.components
    if count <= 0:
        return
    gl.glBindVertexArray(vbo.vao)
    gl.glDrawArrays(gl.GL_TRIANGLES, 0, count)


# --- CAMERA


@dataclass(eq=False, repr=False, slots=True)
class Camera:
    position: V3 = V3()
    front: V3 = V3(z=-1.0)
    up: V3 = V3(y=1.0)
    right: V3 = V3(x=1.0)
    aspect: float = 1.0

    fovy: float = PIOVER2
    yaw: float = PIOVER2 * -1.0
    pitch: float = 0.0
    znear: float = 0.01
    zfar: float = 1000.0


def camera_update_pitch(cam: Camera, angR: float) -> None:
    low: float = to_radians(-89.0)
    high: float = to_radians(89.0)
    cam.pitch = clamp(angR, low, high)

    x: float = cos(cam.yaw) * cos(cam.pitch)
    y: float = sin(cam.pitch)
    z: float = sin(cam.yaw) * cos(cam.pitch)

    cam.front = v3_unit(V3(x, y, z))
    cam.right = v3_unit(v3_cross(cam.front, V3(y=1.0)))
    cam.up = v3_unit(v3_cross(cam.right, cam.front))


def camera_update_yaw(cam: Camera, angR: float) -> None:
    cam.yaw = angR

    x: float = cos(cam.yaw) * cos(cam.pitch)
    y: float = sin(cam.pitch)
    z: float = sin(cam.yaw) * cos(cam.pitch)

    cam.front = v3_unit(V3(x, y, z))
    cam.right = v3_unit(v3_cross(cam.front, V3(y=1.0)))
    cam.up = v3_unit(v3_cross(cam.right, cam.front))


def camera_update_fovy(cam: Camera, angR: float) -> None:
    low: float = to_radians(1.0)
    high: float = to_radians(45.0)
    cam.fovy = clamp(angR, low, high)

    x: float = cos(cam.yaw) * cos(cam.pitch)
    y: float = sin(cam.pitch)
    z: float = sin(cam.yaw) * cos(cam.pitch)

    cam.front = v3_unit(V3(x, y, z))
    cam.right = v3_unit(v3_cross(cam.front, V3(y=1.0)))
    cam.up = v3_unit(v3_cross(cam.right, cam.front))


def camera_view_matrix(cam: Camera) -> M4:
    '''Camerea get view matrix'''
    return m4_look_at(cam.position, v3_add(cam.position, cam.front), cam.up)


def camera_projection_matrix(cam: Camera) -> M4:
    '''Camera get projection matrix'''
    return m4_projection(cam.fovy, cam.aspect, cam.znear, cam.zfar)


# --- GL WINDOW


class GlWindowErr(Exception):
    '''Custom error for gl window'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class GlWindow:
    window: Any = None
    width: int = 0
    height: int = 0
    title: str = "glfw_window"

    def __post_init__(self):
        self.window = glfw.create_window(
                self.width,
                self.height,
                self.title,
                None,
                None
        )

        if not self.window:
            raise GlWindowErr('failed to init glfw window')


def gl_window_should_close(gl_win: GlWindow) -> bool:
    '''Close window'''
    return True if glfw.window_should_close(gl_win.window) else False


def gl_window_center_screen_position(gl_win: GlWindow) -> None:
    '''Center glwindow to center of screen'''
    video = glfw.get_video_mode(glfw.get_primary_monitor())

    x: float = (video.size.width // 2) - (gl_win.width // 2)
    y: float = (video.size.height // 2) - (gl_win.height // 2)

    glfw.set_window_pos(gl_win.window, x, y)


# --- MAIN


def main():
    ''' '''
    if not glfw.init():
        return

    try:
        gl_window = GlWindow(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
        glfw.make_context_current(gl_window.window)
        gl_window_center_screen_position(gl_window)

        clock = Clock()

        camera = Camera(
                position=V3(z=3.0),
                aspect=float(SCREEN_WIDTH) / float(SCREEN_HEIGHT))

        cube = Cube()

        shader: Shader = Shader()
        shader_default(shader)

        vbo: Vbo = Vbo(data_size=cube.data_size)

        vbo_add_data(vbo, cube.verts)

        while not gl_window_should_close(gl_window):
            clock_update(clock)

            gl.glClearColor(0.3, 0.2, 0.2, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            # ---

            shader_use(shader)
            vbo_use(vbo)

            model = m4_from_axis(
                    to_radians(clock.ticks),
                    V3(x=0.5, y=1.0))

            view = camera_view_matrix(camera)
            proj = camera_projection_matrix(camera)

            shader_set_vec3(shader, 'color', V3(x=1.0, y=0.5))
            shader_set_m4(shader, 'model', model)
            shader_set_m4(shader, 'view', view)
            shader_set_m4(shader, 'projection', proj)

            # ---
            glfw.swap_buffers(gl_window.window)
            glfw.poll_events()

    except GlWindowErr:
        glfw.terminate()

    finally:
        vbo_clean(vbo)
        shader_clean(shader)

        glfw.terminate()

        print('CLOSED')


if __name__ == '__main__':
    main()
