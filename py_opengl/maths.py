"""Maths
"""
import math
from typing import Final


# --- CONSTANTS


PI: Final[float]= 3.14159265358979323846
PHI: Final[float]= 1.57079632679489661923
TAU: Final[float]= 6.28318530717958647693
EPSILON: Final[float]= 1e-6
RAD: Final[float]= 0.01745329251994329577
DEG: Final[float]= 57.2957795130823208768
MAX_FLOAT: Final[float]= 1.7976931348623157e+308
MIN_FLOAT: Final[float]= 2.2250738585072014e-308
E: Final[float]= 2.71828182845904523536
INFINITY: Final[float]= math.inf
NEGATIVE_INFINITY: Final[float]= -math.inf


# --- FUNCTIONS


def tan(val: float) -> float:
    return math.tan(val)


def sin(val: float) -> float:
    return math.sin(val)


def cos(val: float) -> float:
    return math.cos(val)


def arccos(val: float) -> float:
    return math.acos(val)


def arcsin(val: float) -> float:
    return math.asin(val)


def arctan2(y: float, x: float) -> float:
    return math.atan2(y, x)


def absf(x: float) -> float:
    return float(abs(x))


def absi(x: int) -> int:
    return int(abs(x))


def gcd(x: int, y: int) -> int:
    return math.gcd(x, y)


def swapf(x: float, y: float) -> tuple[float, float]:
    return (float(y), float(x))


def swapi(x: int, y: int) -> tuple[int, int]:
    return (int(y), int(x))


def is_zero(val: float) -> bool:
    return absf(val) <= EPSILON


def is_one(val: float) -> bool:
    return is_zero(val - 1.0)


def is_equil(x: float, y: float) -> bool:
    return absf(x - y) <= EPSILON


def is_infinite(val: float) -> bool:
    return math.isinf(val)


def signum(val: float) -> int:
    if is_zero(val):
        return 0
    return 1 if val > 0.0 else -1


def to_rad(degrees: float) -> float:
    return degrees * RAD


def to_deg(radians: float) -> float:
    return radians * DEG


def sqr(val: float) -> float:
    return val * val

def cube(val: float) -> float:
    return val * val * val

def sqrt(val: float) -> float:
    return math.sqrt(val)


def floor(val: float) -> float:
    return float(math.floor(val))


def inv_sqrt(val: float) -> float:
    return 1.0 / sqrt(val)


def maxf(x: float, y: float) -> float:
    return float(x) if x > y else float(y)


def maxi(x: int, y: int) -> int:
    return int(x) if x > y else int(y)


def minf(x: float, y: float) -> float:
    return float(x) if x < y else float(y)


def mini(x: int, y: int) -> int:
    return int(x) if x < y else int(y)


def clampf(val: float, low: float, high: float) -> float:
    if val <= low:
        return float(low)

    if val >= high:
        return float(high)

    return float(val)


def clampi(val: int, low: int, high: int) -> int:
    if val <= low:
        return int(low)

    if val >= high:
        return int(high)

    return int(val)


def lerp(start: float, to: float, weight: float) -> float:
    return start + weight * (to - start)


def normalize(val: float, low: float, high: float) -> float:
    return (val - low) / (high - low)


def stepify(val: float, steps: float) -> float:
    """Snaps value to given steps
    """
    return (val // steps) * steps


def wrap(val: int, low: int, high: int) -> int:
    """wrap int value between low and high - 1

    Example
    ---
    wrap(6, 1, 5) => 2

    wrap(5, 1, 6) => 1

    wrap(7, 2, 5) => 4
    """
    return ((val - low) % (high - low) + low)


def wrap_theta(val: float) -> float:
    """Wrap theta between -PI and PI
    """
    if absf(val) > PI:
        val += PI
        val -= float(val // TAU) * TAU
        val -= PI
    return val


def ping_pong(val: int, length: int) -> int:
    """repeat value from 0 to high and back
    """
    val= wrap(val, 0, length * 2)
    val= length - absi(val - length)
    return val


def tri_area_signed(
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float
) -> float:
    """Return tri signed area
    """
    return (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)


# --- VECTOR_2 (X, Y)


class Vec2Error(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class Vec2:

    __slots__= ('x', 'y')

    def __init__(self, x: float= 0.0, y: float= 0.0) -> None:
        self.x: float= x
        self.y: float= y

    def __add__(self, other):
        if not isinstance(other, Vec2):
            raise Vec2Error('not of type Vec2')

        x: float= self.x + other.x
        y: float= self.y + other.y

        return Vec2(x, y)

    def __sub__(self, other):
        if not isinstance(other, Vec2):
            raise Vec2Error('not of type Vec2')

        x: float= self.x - other.x
        y: float= self.y - other.y

        return Vec2(x, y)

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise Vec2Error('other was not of type float or int')

        x: float= self.x * other
        y: float= self.y * other
        return Vec2(x, y)

    def __str__(self) -> str:
        return f'V2({self.x}, {self.y})'

    @staticmethod
    def one() -> 'Vec2':
        return Vec2(1.0, 1.0)

    @staticmethod
    def zero() -> 'Vec2':
        return Vec2(0.0, 0.0)

    @staticmethod
    def create_unit_x() -> 'Vec2':
        return Vec2(x=1.0)

    @staticmethod
    def create_unit_y() -> 'Vec2':
        return Vec2(y=1.0)

    @staticmethod
    def create_from_value(value: float) -> 'Vec2':
        return Vec2(value, value)

    def to_list(self) -> list[float]:
        """Return 'xy' components
        """
        return [self.x, self.y]

    def get_at(self, idx: int) -> float:
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        raise Vec2Error('out of range')

    def set_at(self, idx: int, value: float) -> None:
        match idx:
            case 0:
                self.x= value
            case 1:
                self.y= value
    
    def sum(self) -> float:
        """Return sum of components
        """
        return self.x + self.y

    def lerp(self, to: 'Vec2', weight: float) -> 'Vec2':
        """Return a lerped vec2 between self and to
        """
        x: float= lerp(self.x, to.x, weight)
        y: float= lerp(self.y, to.y, weight)

        return Vec2(x, y)

    def to_unit(self) -> None:
        """Convert to unit length
        """
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            return

        inv: float= inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv

    def unit(self) -> 'Vec2':
        """Return a copy of self with unit length
        """
        lsq: float= self.length_sqr()
        if is_zero(lsq):
            return self.copy()

        return self * inv_sqrt(lsq)

    def scale(self, by: float) -> None:
        """Scale self by
        """
        self.x *= by
        self.y *= by
        
    def copy(self) -> 'Vec2':
        """Return a copy of the self
        """
        return Vec2(self.x, self.y)

    def perpendicular(self) -> 'Vec2':
        """Return the perpendicular of self
        """
        return Vec2(-self.y, self.x)

    def rotate_by(self, angle_deg: float) -> 'Vec2':
        """Return vec2 rotate by angle
        """
        angle_rad: float= to_rad(angle_deg)
        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        x: float= self.x * c - self.y * s
        y: float= self.x * s + self.y * c

        return Vec2(x, y)

    def angle_to(self, other: 'Vec2') -> float:
        """Return angle in radians to self and other
        """
        ang: float= arctan2(other.y, self.x) - arctan2(self.y, other.x)

        if ang > PI:
            ang -= TAU
        elif ang <= -PI:
            ang += TAU

        return ang

    def angle_between(self, other: 'Vec2') -> float:
        """Return angle in radians between self and other
        """
        return arccos(
            self.dot(other) / (self.length_sqrt() * other.length_sqrt())
        )

    def sum_total(self) -> float:
        """Return sum total of components 'xyz'
        """
        return self.x + self.y

    def cross(self, other: 'Vec2') -> float:
        """Return the cross product between self and another
        """
        return (self.y * other.z) - (self.z * other.y)

    def project(self, other: 'Vec2') -> 'Vec2':
        """Return the projection between self and other vec3
        """
        return other * (self.dot(other) / other.length_sqr())

    def reject(self, other: 'Vec2') -> 'Vec2':
        """Return the reject between self and other vec3
        """
        return self - self.project(other)

    def length_sqr(self) -> float:
        """Return the squared length
        """
        return sqr(self.x) + sqr(self.y)

    def length_sqrt(self) -> float:
        """Return the square root length
        """
        return sqrt(self.length_sqr())

    def dot(self, other: 'Vec2') -> float:
        """Return the dot product between self and other vec2
        """
        return (self.x * other.x) + (self.y * other.y)

    def is_unit(self) -> bool:
        """Check if the current length of self is unit
        """
        return is_one(self.length_sqr())

    def is_zero(self) -> bool:
        """Check if the 'xy' components have a zero value
        """
        return is_zero(self.x) and is_zero(self.y)

    def is_equil(self, other: 'Vec2') -> bool:
        """Check if self and other have the same 'xy' component values
        """
        return is_equil(self.x, other.x) and is_equil(self.y, other.y)


# --- VECTOR_3 (X, Y, Z)


class Vec3Error(Exception):
    '''Custom error for Vec3'''

    def __init__(self, msg: str):
        super().__init__(msg)


class Vec3:

    __slots__= ('x', 'y', 'z')
    
    def __init__(self, x: float= 0.0, y: float= 0.0, z: float= 0.0) -> None:
        self.x: float= x
        self.y: float= y
        self.z: float= z

    def __add__(self, other):
        if not isinstance(other, Vec3):
            raise Vec3Error('not of type Vec3')

        x: float= self.x + other.x
        y: float= self.y + other.y
        z: float= self.z + other.z

        return Vec3(x, y, z)

    def __sub__(self, other):
        if not isinstance(other, Vec3):
            raise Vec3Error('not of type Vec3')

        x: float= self.x - other.x
        y: float= self.y - other.y
        z: float= self.z - other.z

        return Vec3(x, y, z)

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise Vec3Error('other was not of type float or int')

        x: float= self.x * other
        y: float= self.y * other
        z: float= self.z * other

        return Vec3(x, y, z)

    def __str__(self) -> str:
        return f'V3({self.x}, {self.y}, {self.z})'

    @staticmethod
    def one() -> 'Vec3':
        return Vec3(1.0, 1.0, 1.0)

    @staticmethod
    def zero() -> 'Vec3':
        return Vec3(0.0, 0.0, 0.0)

    @staticmethod
    def create_unit_x() -> 'Vec3':
        return Vec3(x=1.0)

    @staticmethod
    def create_unit_y() -> 'Vec3':
        return Vec3(y=1.0)

    @staticmethod
    def create_unit_z() -> 'Vec3':
        return Vec3(z=1.0)

    @staticmethod
    def create_from_v2(v2: Vec2, z: float) -> 'Vec3':
        return Vec3(v2.x, v2.y, z)

    @staticmethod
    def create_from_max(a: 'Vec3', b: 'Vec3') -> 'Vec3':
        """Vec3 with the max 'xyz' component values from ever a or b
        """
        return Vec3(
            maxf(a.x, b.x),
            maxf(a.y, b.y),
            maxf(a.z, b.z)
        )

    @staticmethod
    def create_from_min(a: 'Vec3', b: 'Vec3') -> 'Vec3':
        """Vec3 with the min 'xyz' component values from ever a or b
        """
        return Vec3(
            minf(a.x, b.x),
            minf(a.y, b.y),
            minf(a.z, b.z)
        )

    @staticmethod
    def create_from_value(value: float) -> 'Vec3':
        """Create from single value
        """
        return Vec3(value, value, value)

    @staticmethod
    def catmullrom(a: 'Vec3', b: 'Vec3', c: 'Vec3', d: 'Vec3', dt: float) -> 'Vec3':
        result: Vec3= Vec3(0.0, 0.0, 0.0)
        if dt <= 0.0: 
            result.set_from(b)
        elif dt >= 1.0:
            result.set_from(c)
        else:
            t2: float= sqr(dt)
            t3: float= cube(dt)

            v0: Vec3= a * (-0.5 * t3 + t2 - 0.5 * dt)
            v1: Vec3= b * (1.5 * t3 + -2.5 * t2 + 1.0)
            v2: Vec3= c * (-1.5 * t3 + 2.0 * t2 + 0.5 * dt)
            v3: Vec3= d * (0.5 * t3 - 0.5 * t2)

            result.setfrom(v0 + v1 + v2 + v3)

        return result

    @staticmethod
    def barycentric(a: 'Vec3', b: 'Vec3', c: 'Vec3', d: 'Vec3'):
        p0: Vec3= b - a
        p1: Vec3= c - a
        p2: Vec3= d - a

        d0: float= p0.dot(p0)
        d1: float= p0.dot(p1)
        d2: float= p1.dot(p1)
        d3: float= p2.dot(p0)
        d4: float= p2.dot(p1)

        inv: float= 1.0 / (d0 * d2 - d1 * d1)
        v: float= (d2 * d3 - d1 * d4) * inv
        w: float= (d0 * d4 - d1 * d3) * inv
        u: float= 1.0 - v - w

        return Vec3(u, v, w)

    def sum(self) -> float:
        """Return sum of components 'xyz'
        """
        return self.x + self.y + self.z

    def abs(self) -> 'Vec3':
        """Return a copy of self with positive values
        """
        return Vec3(absf(self.x), absf(self.y), absf(self.z))

    def to_list(self) -> list[float]:
        """Return 'xyz' components
        """
        return [self.x, self.y, self.z]

    def lerp(self, to: 'Vec3', weight: float) -> 'Vec3':
        """Return lerped vec3 between self and to
        """
        return Vec3(
            lerp(self.x, to.x, weight),
            lerp(self.y, to.y, weight),
            lerp(self.z, to.z, weight)
        )

    def xy(self) -> Vec2:
        """Return 'xy' components
        """
        return Vec2(self.x, self.y)

    def scale(self, by: float) -> None:
        """Scale self by
        """
        self.x *= by
        self.y *= by
        self.z *= by

    def add(self, other: 'Vec3') -> None:
        """Add other vec3's xyz values to self xyz values
        """
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def subtract(self, other: 'Vec3') -> None:
        """Subtract other vec3's xyz values to self xyz values
        """
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

    def to_unit(self) -> None:
        """Convert to unit length
        """
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            return

        inv: float= inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv
        self.z *= inv

    def unit(self) -> 'Vec3':
        """Return a copy of this vec3 with a normal length

        Raises
        ---
        Vec3Error
            if length is zero
        """
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            return Vec3.zero()

        return self * inv_sqrt(lsq)

    def get_at(self, idx: int) -> float:
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        if idx == 2:
            return self.z

        raise Vec3Error('out of range')

    def set_at(self, idx: int, value: float) -> None:
         match idx:
            case 0:
                self.x= value
            case 1:
                self.y= value
            case 2:
                self.z= value

    def set_from(self, other: 'Vec3') -> None:
        self.x = other.x
        self.y = other.y
        self.z = other.z

    def copy(self) -> 'Vec3':
        """Return a copy of self
        """
        return Vec3(self.x, self.y, self.z)

    def cross(self, other: 'Vec3') -> 'Vec3':
        """Return the cross product between self and another
        """
        return Vec3(
            (self.y * other.z) - (self.z * other.y),
            (self.z * other.x) - (self.x * other.z),
            (self.x * other.y) - (self.y * other.x)
        )

    def project(self, other: 'Vec3') -> 'Vec3':
        """Return the projection between self and other vec3
        """
        return other * (self.dot(other) / other.length_sqr())

    def reject(self, other: 'Vec3') -> 'Vec3':
        """Return the reject between self and other vec3
        """
        return self - self.project(other)

    def rotate_by(self, angle_deg: float, axis: 'Vec3') -> 'Vec3':
        """Rotate by angle along unit axis
        """
        rad: float = to_rad(angle_deg)

        u: Vec3 = axis.copy()

        if not u.is_unit():
            u.to_unit()

        c: float= arccos(rad)
        s: float= arcsin(rad)

        m1: Vec3= Vec3(
            x= c + u.x * u.x * (1 - c),
            y= u.x * u.y * (1 - c) - u.z * s,
            z= u.x * u.z * (1 - c) + u.y * s
        )

        m2: Vec3= Vec3(
            x= u.y * u.x * (1 - c) + u.z * s,
            y= c + u.y * u.y * (1 - c),
            z= u.y * u.z * (1 - c) - u.x * s
        )

        m3: Vec3= Vec3(
            x= u.z * u.x * (1 - c) - u.y * s,
            y= u.z * u.y * (1 - c) + u.x * s,
            z= c + u.z * u.z * (1 - c)
        )

        return Vec3(
            self.dot(m1),
            self.dot(m2),
            self.dot(m3)
        )

    def transform(self, m4: 'Mat4') -> 'Vec3':
        x= self.dot(m4.col0().xyz())
        y= self.dot(m4.col1().xyz())
        z= self.dot(m4.col2().xyz())
        return Vec3(x, y, z)

    def sum_total(self) -> float:
        """Return sum total of components
        """
        return self.x + self.y + self.z

    def length_sqr(self) -> float:
        """Return the squared length
        """
        return sqr(self.x) + sqr(self.y) + sqr(self.z)

    def length_sqrt(self) -> float:
        """Return the square root length
        """
        return sqrt(self.length_sqr())

    def dist_sqr(self, other: 'Vec3') -> float:
        """
        """
        x: float= other.x - self.x
        y: float= other.y - self.y
        z: float= other.z - self.z
        return sqr(x) + sqr(y) + sqr(z)
        
    def dot(self, other: 'Vec3') -> float:
        """Return the dot product between self and other vec3
        """
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def is_unit(self) -> bool:
        """Check if the current length of self is normalized
        """
        return is_one(self.length_sqr())

    def is_zero(self) -> bool:
        """Check if the 'xyz' components have a zero value
        """
        return is_zero(self.x) and is_zero(self.y) and is_zero(self.z)

    def is_equil(self, other: 'Vec3') -> bool:
        """Check if self and other have the same *x, y, z* component values
        """
        return (
            is_equil(self.x, other.x) and
            is_equil(self.y, other.y) and
            is_equil(self.z, other.z)
        )


# --- VECTOR_4 (X, Y, Z, W)


class Vec4Error(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class Vec4:

    __slots__= ('x', 'y', 'z', 'w')

    def __init__(
        self,
        x: float= 0.0,
        y: float= 0.0,
        z: float= 0.0,
        w: float= 0.0,
    ) -> None:
        self.x: float= x
        self.y: float= y
        self.z: float= z
        self.w: float= w

    def __add__(self, other):
        if not isinstance(other, Vec4):
            raise Vec4Error('not of type Vec4')

        return Vec4(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w
        )

    def __sub__(self, other):
        if not isinstance(other, Vec4):
            raise Vec4Error('not of type Vec4')

        return Vec4(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w
        )

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise Vec4Error('other was not of type float or int')

        return Vec4(
            self.x * other,
            self.y * other,
            self.z * other,
            self.w * other
        )

    def __str__(self) -> str:
        return f'V4({self.x}, {self.y}, {self.z}, {self.w})'

    @staticmethod
    def one() -> 'Vec4':
        return Vec4(1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def zero() -> 'Vec4':
        return Vec4(0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def create_unit_x() -> 'Vec4':
        return Vec4(x= 1.0)

    @staticmethod
    def create_unit_y() -> 'Vec4':
        return Vec4(y= 1.0)

    @staticmethod
    def create_unit_z() -> 'Vec4':
        return Vec4(z= 1.0)

    @staticmethod
    def create_unit_w() -> 'Vec4':
        return Vec4(w= 1.0)

    @staticmethod
    def create_from_v2(xy: Vec2, z: float, w: float) -> 'Vec4':
        x: float= xy.x
        y: float= xy.y
        return Vec4(x, y, z, w)

    @staticmethod
    def create_from_v3(xyz: Vec3, w: float= 0.0) -> 'Vec4':
        x: float= xyz.x
        y: float= xyz.y
        z: float= xyz.z
        return Vec4(x, y, z, w)

    @staticmethod
    def create_from_value(value: float) -> 'Vec4':
        return Vec4(value, value, value, value)

    def sum(self) -> float:
        """Return sum of components
        """
        return self.x + self.y + self.z + self.w

    def to_list(self) -> list[float]:
        """Return list[float] of *xyzw* components
        """
        return [self.x, self.y, self.z, self.w]

    def xy(self) -> Vec2:
        """Return 'xy' components
        """
        return Vec2(self.x, self.y)

    def xyz(self) -> Vec3:
        """Return 'xyz' components
        """
        return Vec3(self.x, self.y, self.z)

    def scale(self, by: float) -> None:
        """Scale self by
        """
        self.x *= by
        self.y *= by
        self.z *= by
        self.w *= by

    def lerp(self, to: 'Vec4', weight: float) -> 'Vec4':
        """Return lerped vec4 between self and to
        """
        return Vec4(
            lerp(self.x, to.x, weight),
            lerp(self.y, to.y, weight),
            lerp(self.z, to.z, weight),
            lerp(self.w, to.w, weight)
        )

    def get_at(self, idx: int) -> float:
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        if idx == 2:
            return self.z
        if idx == 3:
            return self.w

        raise Vec4Error('out of range')

    def set_at(self, idx: int, value: float) -> None:
        match idx:
            case 0:
                self.x= value
            case 1:
                self.y= value
            case 2:
                self.z= value
            case 3:
                self.w= value

    def set_from(self, other: 'Vec4') -> None:
        self.x = other.x
        self.y = other.y
        self.z = other.z
        self.w = other.w

    def copy(self) -> 'Vec4':
        """Return a copy of the vec4

        Returns
        ---
        Vec4
            a copy
        """
        return Vec4(self.x, self.y, self.z, self.w)

    def to_unit(self) -> None:
        """Convert to unit length
        """
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            return
    
        inv: float= inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv
        self.z *= inv
        self.w *= inv

    def unit(self) -> 'Vec4':
        """Return self with unit length
        """
        lsq: float = self.length_sqr()

        if is_zero(lsq):
            raise Vec4Error('length of this vec4 was zero')

        return self * inv_sqrt(lsq)

    def barycentric(self, b: 'Vec4', c: 'Vec4', u: float, v: float) -> 'Vec4':
        a: Vec4= self.copy()
        result: Vec4= a + (b - a) * u + (c - a) * v

        return result

    def sum_total(self) -> float:
        """Return sum of all values 'xyzw'
        """
        return self.x + self.y + self.z + self.w

    def length_sqr(self) -> float:
        """Return the squared length
        """
        return sqr(self.x) + sqr(self.y) + sqr(self.z) + sqr(self.w)

    def length_sqrt(self) -> float:
        """Return the square root length
        """
        return sqrt(self.length_sqr())

    def dot(self, other: 'Vec4') -> float:
        """Return the dot product between self and other vec4
        """
        x: float= self.x * other.x
        y: float= self.y * other.y
        z: float= self.z * other.z
        w: float= self.w * other.w
        return x + y + z + w

    def is_unit(self) -> bool:
        """Check if the current length of self is normalized
        """
        return is_one(self.length_sqr())

    def is_equil(self, other: 'Vec4') -> bool:
        """Check if self and other have the same *x, y, z, w* component values
        """
        return (
            is_equil(self.x, other.x) and
            is_equil(self.y, other.y) and
            is_equil(self.z, other.z) and
            is_equil(self.w, other.w)
        )

    def is_zero(self) -> bool:
        """Check if the 'xyzw' components have a zero value
        """
        return (
            is_zero(self.x) and
            is_zero(self.y) and
            is_zero(self.z) and
            is_zero(self.w)
        )
        


# --- MATRIX_3


class Mat3Error(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class Mat3:

    __slots__= ('row0', 'row1', 'row2')

    def __init__(
        self,
        row0: Vec3= Vec3(),
        row1: Vec3= Vec3(),
        row2: Vec3= Vec3(),
    ) -> None:
        self.row0: Vec3= row0
        self.row1: Vec3= row1
        self.row2: Vec3= row2

    def __add__(self, other):
        if not isinstance(other, Mat3):
            raise Mat3Error('not of type mat3')
        r0= self.row0 + other.row0
        r1= self.row1 + other.row1
        r2= self.row2 + other.row2

        return Mat3(r0, r1, r2)

    def __sub__(self, other):
        if not isinstance(other, Mat3):
            raise Mat3Error('not of type mat3')
        r0= self.row0 - other.row0
        r1= self.row1 - other.row1
        r2= self.row2 - other.row2

        return Mat3(r0, r1, r2)

    def __mul__(self, other):
        if not isinstance(other, Mat3):
            raise Mat3Error('not of type Mat3')

        r0: Vec3= Vec3(
            self.row0.dot(other.col0()),
            self.row0.dot(other.col1()),
            self.row0.dot(other.col2())
        )

        r1: Vec3= Vec3(
            self.row1.dot(other.col0()),
            self.row1.dot(other.col1()),
            self.row1.dot(other.col2())
        )

        r2: Vec3= Vec3(
            self.row2.dot(other.col0()),
            self.row2.dot(other.col1()),
            self.row2.dot(other.col2()),
        )

        return Mat3(r0, r1, r2)

    @staticmethod
    def create_from_values(
        ax: float, ay: float, az: float,
        bx: float, by: float, bz: float,
        cx: float, cy: float, cz: float
    ) -> 'Mat3':
        return Mat3(
            Vec3(ax, ay, az),
            Vec3(bx, by, bz),
            Vec3(cx, cy, cz),
        )

    @staticmethod
    def create_from_axis(angle_deg: float, unit_axis: Vec3) -> 'Mat3':
        """Create a rotated matrix
        """
        if not unit_axis.is_unit():
            unit_axis.to_unit()

        rad: float= to_rad(angle_deg)
        x: float= unit_axis.x
        y: float= unit_axis.y
        z: float= unit_axis.z
        c: float= cos(rad)
        s: float= sin(rad)
        t: float = 1.0 - c

        xx: float= t * sqr(x)
        xy: float= t * x * y
        xz: float= t * x * z
        yy: float= t * sqr(y)
        yz: float= t * y * z
        zz: float= t * sqr(z)

        sin_x: float= s * x
        sin_y: float= s * y
        sin_z: float= s * z

        ax: float= xx + c
        ay: float= xy - sin_z
        az: float= xz + sin_y
        bx: float= xy + sin_z
        by: float= yy + c
        bz: float= yz - sin_x
        cx: float= xz - sin_y
        cy: float= yz + sin_x
        cz: float= zz + c

        return Mat3(
            Vec3(ax, ay, az),
            Vec3(bx, by, bz),
            Vec3(cx, cy, cz)
        )

    @staticmethod
    def create_from_quaternion(q: 'Quaternion') -> 'Mat3':
        """Create matrix from quetrnion values
        """
        x2: float= sqr(q.x)
        y2: float= sqr(q.y)
        z2: float= sqr(q.z)
        w2: float= sqr(q.w)

        xy: float= q.x * q.y
        xz: float= q.x * q.z
        xw: float= q.x * q.w

        yz: float= q.y * q.z
        yw: float= q.y * q.w

        zw: float= q.z * q.w

        s2: float= 2.0 / (x2 + y2 + z2 + w2)

        ax: float= 1.0 - (s2 * (y2 + z2))
        ay: float= s2 * (xy + zw)
        az: float= s2 * (xz - yw)
        bx: float= s2 * (xy - zw)
        by: float= 1.0 - (s2 * (x2 + z2))
        bz: float= s2 * (yz + xw)
        cx: float= s2 * (xz + yw)
        cy: float= s2 * (yz - xw)
        cz: float= 1.0 - (s2 * (x2 + y2))

        return Mat3(
            Vec3(ax, ay, az),
            Vec3(bx, by, bz),
            Vec3(cx, cy, cz),
        )

    @staticmethod
    def identity() -> 'Mat3':
        """Create an identity matrix
        """
        return Mat3(
            Vec3(x= 1.0),
            Vec3(y= 1.0),
            Vec3(z= 1.0)
        )

    @staticmethod
    def create_scaler(v3: Vec3) -> 'Mat3':
        """Create a scaler matrix
        """
        return Mat3(
            Vec3(x= v3.x),
            Vec3(y= v3.y),
            Vec3(z= v3.z)
        )

    @staticmethod
    def create_rotation_x(angle_deg: float) -> 'Mat3':
        """Create a rotation on x axis matrix
        """
        angle_rad: float= to_rad(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat3(
            Vec3(x= 1.0),
            Vec3(y= c, z= -s),
            Vec3(y= s, z= c)
        )

    @staticmethod
    def create_rotation_y(angle_deg: float) -> 'Mat3':
        """Create a rotation on y axis matrix
        """
        angle_rad: float= to_rad(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat3(
            Vec3(x= c, z= s),
            Vec3(y= 1.0),
            Vec3(x= -s, z= c)
        )

    @staticmethod
    def create_rotation_z(angle_deg: float) -> 'Mat3':
        """Create a rotation on z axis matrix
        """
        angle_rad: float= to_rad(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat3(
            Vec3(x= c, y= -s),
            Vec3(x= s, y= c),
            Vec3(z= 1.0)
        )

    def copy(self) -> 'Mat3':
        """Return a copy of self
        """
        return Mat3(self.row0.copy(), self.row1.copy(), self.row2.copy)

    def scale(self, by: float) -> 'Mat3':
        """Return a scaled copy of self
        """
        r0: Vec3= self.row0 * by
        r1: Vec3= self.row1 * by
        r2: Vec3= self.row2 * by
        return Mat3(r0, r1, r2)

    def sum(self) -> float:
        """Return sum of all rows
        """
        return (
            self.row0.sum() +
            self.row1.sum() +
            self.row2.sum()
        )

    def transpose(self) -> 'Mat3':
        """Return a transposed copy of self
        """
        r0: Vec3= self.col0()
        r1: Vec3= self.col1()
        r2: Vec3= self.col2()
        return Mat3(r0, r1, r2)

    def cofactor(self) -> 'Mat3':
        """Return a cofactor copy of self
        """
        ax: float= self.row0.x
        ay: float= -self.row0.y
        az: float= self.row0.z
        bx: float= -self.row1.bx
        by: float= self.row1.by
        bz: float= -self.row1.bz
        cx: float= self.row2.cx
        cy: float= -self.row2.cy
        cz: float= self.row2.cz

        return Mat3(
            Vec3(ax, ay, az),
            Vec3(bx, by, bz),
            Vec3(cx, cy, cz)
        )

    def to_unit(self) -> None:
        """Convert to unit length
        """
        det: float= self.determinant()

        if is_zero(det):
            return

        inv: float= 1.0 / det
        self.row0.set_from(self.row0 * inv)
        self.row1.set_from(self.row1 * inv)
        self.row2.set_from(self.row2 * inv)

    def unit(self) -> 'Mat3':
        """Return a copy of self with normalized length

        Raises
        ---
        Mat3Error
            if the determinant of self is zero
        """
        det: float= self.determinant()

        if is_zero(det):
            raise Mat3Error('length of this Mat3 was zero')

        inv: float= 1.0 / det

        r0: Vec3= self.row0 * inv
        r1: Vec3= self.row1 * inv
        r2: Vec3= self.row2 * inv

        return Mat3(r0, r1, r2)

    def inverse(self) -> 'Mat3':
        """Return the inverse of self
        """
        a: float= self.row0.x
        b: float= self.row0.y
        c: float= self.row0.z
        d: float= self.row1.x
        e: float= self.row1.y
        f: float= self.row1.z
        g: float= self.row2.x
        h: float= self.row2.y
        i: float= self.row2.z

        det: float= self.determinant()
        inv: float= 1.0 / det

        ax: float= (e * i - h * f) * inv
        ay: float= (g * f - d * i) * inv
        az: float= (d * h - g * e) * inv
        bx: float= (h * c - b * i) * inv
        by: float= (a * i - g * c) * inv
        bz: float= (g * b - a * h) * inv
        cx: float= (b * f - e * c) * inv
        cy: float= (d * c - a * f) * inv
        cz: float= (a * e - d * b) * inv

        return Mat3(
            Vec3(ax, ay, az),
            Vec3(bx, by, bz),
            Vec3(cx, cy, cz)
        )

    def col0(self) -> Vec3:
        """Return column of float values
        """
        return Vec3(self.row0.x, self.row1.x, self.row2.x)

    def col1(self) -> Vec3:
        """Return column of float values
        """
        return Vec3(self.row0.y, self.row1.y, self.row2.y)

    def col2(self) -> Vec3:
        """Return column of float values
        """
        return Vec3(self.row0.z, self.row1.z, self.row2.z)

    def multiply_v3(self, v3: Vec3) -> Vec3:
        """Return mat3 multiplyed by vec3
        """
        x: float= v3.dot(self.col0()),
        y: float= v3.dot(self.col1()),
        z: float= v3.dot(self.col2()),
        return Vec3(x, y, z)

    def get_rotation(self) -> 'Quaternion':
        """Get rotation from matrix
        """
        result: Vec4= Vec4()

        ax: float= self.row0.x
        ay: float= self.row0.y
        az: float= self.row0.z
        bx: float= self.row1.x
        by: float= self.row1.y
        bz: float= self.row1.z
        cx: float= self.row2.x
        cy: float= self.row2.y
        cz: float= self.row2.z

        if ax + by + cz > 0:
            sq: float= sqrt(ax + by + cz + 1.0) * 2.0
            result.w= 0.25 * sq
            result.x= (bz - cy) / sq
            result.y= (cx - az) / sq
            result.z= (ay - bx) / sq
        elif ax > by and ax > cz:
            sq: float= sqrt(1.0 + ax - by - cz) * 2.0
            result.w= (bz - cy) / sq
            result.x= 0.25 * sq
            result.y= (ay + bx) / sq
            result.z= (cx + az) / sq
        elif by > cz:
            sq: float= sqrt(1.0 + by - ax - cz) * 2.0
            result.w= (cx - az) / sq
            result.x= (ay + bx) / sq
            result.y= 0.25 * sq
            result.z= (bz + cy) / sq
        else:
            sq: float= sqrt(1.0 + cz - ax - by) * 2.0
            result.w= (ay - bx) / sq
            result.x= (cx + az) / sq
            result.y= (bz + cy) / sq
            result.z= 0.25 * sq

        if not result.is_unit():
            result.to_unit()

        return Quaternion.create_from_vec4(result)

    def get_at(self, row: int, col: int) -> float:
        if row == 0:
            return self.row0[col]
        if row == 1:
            return self.row1[col]
        if row == 2:
            return self.row2[col]
        raise Mat3Error('out of range')

    def set_at(self, row: int, col: int, value: float) -> None:
        if row == 0:
            self.row0[col]= value
            return

        if row == 1:
            self.row1[col]= value
            return

        if row == 2:
            self.row2[col]= value
            return

        raise Mat3Error('out of range')

    def determinant(self) -> float:
        """Return the determinant of self
        """
        a0: float= self.row0.x * self.row1.y * self.row2.z
        a1: float= self.row0.y * self.row1.z * self.row2.x
        a2: float= self.row0.z * self.row1.x * self.row2.y
        b1: float= self.row0.z * self.row1.y * self.row2.x
        b2: float= self.row0.x * self.row1.z * self.row2.y
        b3: float= self.row0.y * self.row1.x * self.row2.z

        return a0 + a1 + a2 - b1 - b2 - b3

    def trace(self) -> float:
        """Return the sum of the trace values
        """
        return self.row0.x + self.row1.y + self.row2.z

    def array(self) -> list[float]:
        """Return a single array of matrix values
        """
        return [
            self.row0.x, self.row0.y, self.row0.z,
            self.row1.x, self.row1.y, self.row1.z,
            self.row2.x, self.row2.y, self.row2.z
        ]

    def multi_array(self) -> list[list[float]]:
        """Return a multi dimentional array of matrix values
        """
        return [
            [self.row0.x, self.row0.y, self.row0.z],
            [self.row1.x, self.row1.y, self.row1.z],
            [self.row2.x, self.row2.y, self.row2.z]
        ]


# --- MATRIX_4


class Mat4Error(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class Mat4:

    __slots__= (
        'row0',
        'row1',
        'row2',
        'row3',
    )

    def __init__(
        self,
        row0: Vec4= Vec4(),
        row1: Vec4= Vec4(),
        row2: Vec4= Vec4(),
        row3: Vec4= Vec4(),
    ) -> None:
        self.row0: Vec4= row0
        self.row1: Vec4= row1
        self.row2: Vec4= row2
        self.row3: Vec4= row3

    def __add__(self, other):
        if not isinstance(other, Mat4):
            raise Mat4Error('not of type mat4')
        r0: Vec4= self.row0 + other.row0
        r1: Vec4= self.row1 + other.row1
        r2: Vec4= self.row2 + other.row2
        r3: Vec4= self.row3 + other.row3

        return Mat4(r0, r1, r2, r3)

    def __sub__(self, other):
        if not isinstance(other, Mat4):
            raise Mat4Error('not of type mat4')

        r0: Vec4= self.row0 - other.row0
        r1: Vec4= self.row1 - other.row1
        r2: Vec4= self.row2 - other.row2
        r3: Vec4= self.row3 - other.row3

        return Mat4(r0, r1, r2, r3)

    def __mul__(self, other):
        if not isinstance(other, Mat4):
            raise Mat4Error('not of type Mat4')
        r0: Vec4 = Vec4(
            self.row0.dot(other.col0()),
            self.row0.dot(other.col1()),
            self.row0.dot(other.col2()),
            self.row0.dot(other.col3())
        )

        r1: Vec4 = Vec4(
            self.row1.dot(other.col0()),
            self.row1.dot(other.col1()),
            self.row1.dot(other.col2()),
            self.row1.dot(other.col3())
        )

        r2: Vec4 = Vec4(
            self.row2.dot(other.col0()),
            self.row2.dot(other.col1()),
            self.row2.dot(other.col2()),
            self.row2.dot(other.col3())
        )

        r3: Vec4 = Vec4(
            self.row3.dot(other.col0()),
            self.row3.dot(other.col1()),
            self.row3.dot(other.col2()),
            self.row3.dot(other.col3())
        )

        return Mat4(r0, r1, r2, r3)

    @staticmethod
    def create_from_values(
        ax: float, ay: float, az: float, aw: float,
        bx: float, by: float, bz: float, bw: float,
        cx: float, cy: float, cz: float, cw: float,
        dx: float, dy: float, dz: float, dw: float
    ) -> 'Mat4':
        return Mat4(
            Vec4(ax, ay, az, aw),
            Vec4(bx, by, bz, bw),
            Vec4(cx, cy, cz, cw),
            Vec4(dx, dy, dz, dw),
        )

    @staticmethod
    def identity() -> 'Mat4':
        """Create an identity mat4 matrix
        """
        return Mat4(
            Vec4(x= 1.0),
            Vec4(y= 1.0),
            Vec4(z= 1.0),
            Vec4(w= 1.0)
        )

    @staticmethod
    def create_scaler(v3: Vec3) -> 'Mat4':
        """Create a scaler mat4 matrix
        """
        return Mat4(
            Vec4(x= v3.x),
            Vec4(y= v3.y),
            Vec4(z= v3.z),
            Vec4(w= 1.0)
        )

    @staticmethod
    def create_translation(v3: Vec3) -> 'Mat4':
        """Create a translation mat4 matrix
        """
        return Mat4(
            Vec4(x= 1.0),
            Vec4(y= 1.0),
            Vec4(z= 1.0),
            Vec4.create_from_v3(v3, 1.0)
        )

    @staticmethod
    def create_shear_x(shear_y: float, shear_z: float) -> 'Mat4':
        """Create a shear mat4 matrix on x axis
        """
        return Mat4(
            Vec4(x= 1.0, y= shear_y, z= shear_z),
            Vec4(y= 1.0),
            Vec4(z= 1.0),
            Vec4(w= 1.0)
        )

    @staticmethod
    def create_shear_y(shear_x: float, shear_z: float) -> 'Mat4':
        """Create a shear mat4 matrix on y axis
        """
        return Mat4(
            Vec4(x= 1.0),
            Vec4(x= shear_x, y= 1.0, z= shear_z),
            Vec4(z= 1.0),
            Vec4(w= 1.0)
        )

    @staticmethod
    def create_shear_z(shear_x: float, shear_y: float) -> 'Mat4':
        """Create a shear mat4 matrix on z axis
        """
        return Mat4(
            Vec4(x= 1.0),
            Vec4(y= 1.0),
            Vec4(x= shear_x, y= shear_y, z= 1.0),
            Vec4(w= 1.0)
        )

    @staticmethod
    def create_rotation_x(angle_deg: float) -> 'Mat4':
        """Create a rotation *X* mat4 matrix
        """
        angle_rad: float= to_rad(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat4(
            Vec4(x= 1.0),
            Vec4(y= c, z= -s),
            Vec4(y= s, z= c),
            Vec4(w= 1.0)
        )

    @staticmethod
    def create_rotation_y(angle_deg: float) -> 'Mat4':
        """Create a rotation *y* mat4 matrix
        """
        angle_rad: float= to_rad(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat4(
            Vec4(x= c, z= s),
            Vec4(y= 1.0),
            Vec4(x= -s, z= c),
            Vec4(w= 1.0)
        )

    @staticmethod
    def create_rotation_z(angle_deg: float) -> 'Mat4':
        """Create a rotation *z* mat4 matrix
        """
        angle_rad: float= to_rad(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat4(
            Vec4(x= c, y= -s),
            Vec4(x= s, y= c),
            Vec4(z= 1.0),
            Vec4(w= 1.0)
        )

    @staticmethod
    def create_from_axis(angle_deg: float, unit_axis: Vec3) -> 'Mat4':
        """Create a rotated matrix
        """
        if not unit_axis.is_unit():
            unit_axis.to_unit()

        rad: float= to_rad(angle_deg)
        x: float= unit_axis.x
        y: float= unit_axis.y
        z: float= unit_axis.z
        c: float= cos(rad)
        s: float= sin(rad)
        t: float = 1.0 - c

        xx: float= t * sqr(x)
        xy: float= t * x * y
        xz: float= t * x * z
        yy: float= t * sqr(y)
        yz: float= t * y * z
        zz: float= t * sqr(z)

        sin_x: float= s * x
        sin_y: float= s * y
        sin_z: float= s * z

        ax: float= xx + c
        ay: float= xy - sin_z
        az: float= xz + sin_y
        bx: float= xy + sin_z
        by: float= yy + c
        bz: float= yz - sin_x
        cx: float= xz - sin_y
        cy: float= yz + sin_x
        cz: float= zz + c

        return Mat4(
            Vec4(x= ax, y= ay, z= az),
            Vec4(x= bx, y= by, z= bz),
            Vec4(x= cx, y= cy, z= cz),
            Vec4(w= 1.0)
        )

    @staticmethod
    def create_from_quaternion(q: 'Quaternion') -> 'Mat4':
        x2: float= sqr(q.x)
        y2: float= sqr(q.y)
        z2: float= sqr(q.z)
        w2: float= sqr(q.w)

        xy: float= q.x * q.y
        xz: float= q.x * q.z
        xw: float= q.x * q.w

        yz: float= q.y * q.z
        yw: float= q.y * q.w

        zw: float= q.z * q.w

        s2: float= 2.0 / (x2 + y2 + z2 + w2)

        ax: float= 1.0 - (s2 * (y2 + z2))
        ay: float= s2 * (xy + zw)
        az: float= s2 * (xz - yw)
        bx: float= s2 * (xy - zw)
        by: float= 1.0 - (s2 * (x2 + z2))
        bz: float= s2 * (yz + xw)
        cx: float= s2 * (xz + yw)
        cy: float= s2 * (yz - xw)
        cz: float= 1.0 - (s2 * (x2 + y2))

        return Mat4(
            Vec4(x= ax, y= ay, z= az),
            Vec4(x= bx, y= by, z= bz),
            Vec4(x= cx, y= cy, z= cz),
            Vec4(w= 1.0)
        )

    @staticmethod
    def create_orthographic(
        left: float,
        right: float,
        top: float,
        bottom: float,
        near: float,
        far: float
    ) -> 'Mat4':
        """Create a orthographic projection matrix
        """
        inv_rl: float= 1.0 / (right - left)
        inv_tb: float= 1.0 / (top - bottom)
        inv_fn: float= 1.0 / (far - near)

        rl: float= right + left
        tb: float= top + bottom
        fn: float= far + near

        return Mat4(
            Vec4(x= 2.0 * inv_rl),
            Vec4(y= 2.0 * inv_tb),
            Vec4(z= -2.0 * inv_fn),
            Vec4(x= -rl * inv_rl, y= -tb * inv_tb, z= -fn * inv_fn, w= 1.0)
        )

    @staticmethod
    def create_perspective(
            fov: float,
            aspect: float,
            near: float,
            far: float
    ) -> 'Mat4':
        """Create a matrix 'perspective' field of view

        Raises
        ---
        Mat4Error
            if values like aspect equil zero
        """
        inv_r: float= 1.0 / to_rad(fov * 0.5)
        dz: float= far - near
        s: float= sin(inv_r)

        if is_zero(dz) or is_zero(s) or is_zero(aspect):
            raise Mat4Error("m4 projection values were zero")

        c: float= cos(inv_r) / s

        return Mat4(
            Vec4(x= c / aspect),
            Vec4(y= c),
            Vec4(z= -(far + near) / dz, w= -1.0),
            Vec4(z= -2.0 * near * far / dz)
        )

    @staticmethod
    def create_look_at(eye: Vec3, target: Vec3, up: Vec3) -> 'Mat4':
        """Create a  matrix 'look at' field of view
        """
        z: Vec3 = (eye - target)
        if not z.is_unit():
            z.to_unit()

        if z.is_zero():
            return Mat4.identity()

        x: Vec3= up.cross(z)
        if not x.is_unit():
            x.to_unit()

        y: Vec3= z.cross(x)
        if not y.is_unit():
            y.to_unit()

        d: Vec3= Vec3(
            -x.dot(eye),
            -y.dot(eye),
            -z.dot(eye)
        )

        return Mat4(
            Vec4(x= x.x, y= y.x, z= z.x),
            Vec4(x= x.y, y= y.y, z= z.y),
            Vec4(x= x.z, y= y.z, z= z.z),
            Vec4.create_from_v3(d, 1.0)
        )

    def add(self, other: 'Mat4') -> None:
        """Set self to the addition between self and other matrix
        """
        r0: Vec4= self.row0 + other.row0
        r1: Vec4= self.row1 + other.row1
        r2: Vec4= self.row2 + other.row2
        r3: Vec4= self.row3 + other.row3

        self.row0.x = r0.x
        self.row0.y = r0.y
        self.row0.z = r0.z
        self.row0.w = r0.w

        self.row1.x = r1.x
        self.row1.y = r1.y
        self.row1.z = r1.z
        self.row1.w = r1.w

        self.row2.x = r2.x
        self.row2.y = r2.y
        self.row2.z = r2.z
        self.row2.w = r2.w

        self.row3.x = r3.x
        self.row3.y = r3.y
        self.row3.z = r3.z
        self.row3.w = r3.w

    def multiply(self, other: 'Mat4') -> None:
        """Set matrix to multipication between self with other matrix
        """
        self.row0.x= self.row0.dot(other.col0())
        self.row0.y= self.row0.dot(other.col1())
        self.row0.z= self.row0.dot(other.col2())
        self.row0.w= self.row0.dot(other.col3())

        self.row1.x= self.row1.dot(other.col0())
        self.row1.y= self.row1.dot(other.col1())
        self.row1.z= self.row1.dot(other.col2())
        self.row1.w= self.row1.dot(other.col3())

        self.row2.x= self.row2.dot(other.col0())
        self.row2.y= self.row2.dot(other.col1())
        self.row2.z= self.row2.dot(other.col2())
        self.row2.w= self.row2.dot(other.col3())

        self.row3.x= self.row3.dot(other.col0())
        self.row3.y= self.row3.dot(other.col1())
        self.row3.z= self.row3.dot(other.col2())
        self.row3.w= self.row3.dot(other.col3())

    def copy(self) -> 'Mat4':
        """Return a copy
        """
        return Mat4(
            self.row0.copy(),
            self.row1.copy(),
            self.row2.copy(),
            self.row3.copy(),
        )

    def scale(self, by: float) -> 'Mat4':
        """Return matrix scaled by value
        """
        r0: Vec4= self.row0 * by
        r1: Vec4= self.row1 * by
        r2: Vec4= self.row2 * by
        r3: Vec4= self.row3 * by

        return Mat4(r0, r1, r2, r3)

    def transpose(self) -> 'Mat4':
        """Return the transpose of self matrix
        """
        r0: Vec4= self.col0()
        r1: Vec4= self.col1()
        r2: Vec4= self.col2()
        r3: Vec4= self.col3()

        return Mat4(r0, r1, r2, r3)

    def cofactor(self) -> 'Mat4':
        """Return the cofactor of self matrix
        """
        ax: float= self.row0.x
        ay: float= -self.row0.y
        az: float= self.row0.z
        aw: float= -self.row0.w
        bx: float= -self.row1.x
        by: float= self.row1.y
        bz: float= -self.row1.z
        bw: float= self.row1.w
        cx: float= self.row2.x
        cy: float= -self.row2.y
        cz: float= self.row2.z
        cw: float= -self.row2.w
        dx: float= -self.row3.x
        dy: float= self.row3.y
        dz: float= -self.row3.z
        dw: float= self.row3.w

        return Mat4(
            Vec4(ax, ay, az, aw),
            Vec4(bx, by, bz, bw),
            Vec4(cx, cy, cz, cw),
            Vec4(dx, dy, dz, dw)
        )

    def inverse(self) -> 'Mat4':
        """Return the inverse of self matrix
        """
        a00: float= self.row0.x
        a01: float= self.row0.y
        a02: float= self.row0.z
        a03: float= self.row0.w
        a10: float= self.row1.x
        a11: float= self.row1.y
        a12: float= self.row1.z
        a13: float= self.row1.w
        a20: float= self.row2.x
        a21: float= self.row2.y
        a22: float= self.row2.z
        a23: float= self.row2.w
        a30: float= self.row3.x
        a31: float= self.row3.y
        a32: float= self.row3.z
        a33: float= self.row3.w

        b00: float= a00 * a11 - a01 * a10
        b01: float= a00 * a12 - a02 * a10
        b02: float= a00 * a13 - a03 * a10
        b03: float= a01 * a12 - a02 * a11
        b04: float= a01 * a13 - a03 * a11
        b05: float= a02 * a13 - a03 * a12
        b06: float= a20 * a31 - a21 * a30
        b07: float= a20 * a32 - a22 * a30
        b08: float= a20 * a33 - a23 * a30
        b09: float= a21 * a32 - a22 * a31
        b10: float= a21 * a33 - a23 * a31
        b11: float= a22 * a33 - a23 * a32

        det: float= (
            b00 * b11 - b01 *
            b10 + b02 * b09 +
            b03 * b08 - b04 *
            b07 + b05 * b06
        )

        inv: float= 1.0 / det

        ax: float= (a11 * b11 - a12 * b10 + a13 * b09) * inv
        ay: float= (-a01 * b11 + a02 * b10 - a03 * b09) * inv
        az: float= (a31 * b05 - a32 * b04 + a33 * b03) * inv
        aw: float= (-a21 * b05 + a22 * b04 - a23 * b03) * inv
        bx: float= (-a10 * b11 + a12 * b08 - a13 * b07) * inv
        by: float= (a00 * b11 - a02 * b08 + a03 * b07) * inv
        bz: float= (-a30 * b05 + a32 * b02 - a33 * b01) * inv
        bw: float= (a20 * b05 - a22 * b02 + a23 * b01) * inv
        cx: float= (a10 * b10 - a11 * b08 + a13 * b06) * inv
        cy: float= (-a00 * b10 + a01 * b08 - a03 * b06) * inv
        cz: float= (a30 * b04 - a31 * b02 + a33 * b00) * inv
        cw: float= (-a20 * b04 + a21 * b02 - a23 * b00) * inv
        dx: float= (-a10 * b09 + a11 * b07 - a12 * b06) * inv
        dy: float= (a00 * b09 - a01 * b07 + a02 * b06) * inv
        dz: float= (-a30 * b03 + a31 * b01 - a32 * b00) * inv
        dw: float= (a20 * b03 - a21 * b01 + a22 * b00) * inv

        return Mat4(
            Vec4(ax, ay, az, aw),
            Vec4(bx, by, bz, bw),
            Vec4(cx, cy, cz, cw),
            Vec4(dx, dy, dz, dw)
        )

    def to_unit(self) -> None:
        """Convert to unit length
        """
        det: float= self.determinant()

        if is_zero(det):
            return

        inv: float= 1.0 / det
        self.row0.set_from(self.row0 * inv)
        self.row1.set_from(self.row1 * inv)
        self.row2.set_from(self.row2 * inv)
        self.row3.set_from(self.row3 * inv)

    def unit(self) -> 'Mat4':
        """Return a copy of matrix with unit length
        """
        det: float= self.determinant()

        if is_zero(det):
            raise Mat4Error('length of this Mat4 was zero')

        inv: float= 1.0 / det

        return Mat4(
            self.row0 * inv,
            self.row1 * inv,
            self.row2 * inv,
            self.row3 * inv,
        )

    def get_translation(self) -> Vec3:
        """Return the transformed values from matrix
        """
        return self.row3.xyz()

    def get_rotation(self) -> 'Quaternion':
        """Return the rotation values from matrix
        """

        ax: float = self.row0.x
        ay: float = self.row0.y
        az: float = self.row0.z

        bx: float = self.row1.x
        by: float = self.row1.y
        bz: float = self.row1.z

        cx: float = self.row2.x
        cy: float = self.row2.y
        cz: float = self.row2.z

        result: Vec4= Vec4()

        if ax + by + cz > 0:
            sq: float= sqrt(ax + by + cz + 1.0) * 2.0
            result.w= 0.25 * sq
            result.x= (bz - cy) / sq
            result.y= (cx - az) / sq
            result.z= (ay - bx) / sq
        elif ax > by and ax > cz:
            sq: float= sqrt(1.0 + ax - by - cz) * 2.0
            result.w= (bz - cy) / sq
            result.x= 0.25 * sq
            result.y= (ay + bx) / sq
            result.z= (cx + az) / sq
        elif by > cz:
            sq: float= sqrt(1.0 + by - ax - cz) * 2.0
            result.w= (cx - az) / sq
            result.x= (ay + bx) / sq
            result.y= 0.25 * sq
            result.z= (bz + cy) / sq
        else:
            sq: float= sqrt(1.0 + cz - ax - by) * 2.0
            result.w= (ay - bx) / sq
            result.x= (cx + az) / sq
            result.y= (bz + cy) / sq
            result.z= 0.25 * sq

        return Quaternion.create_from_vec4(result)

    def get_scale(self) -> Vec3:
        """Return the scaler values from matrix
        """
        x: float= self.row0.xyz().length_sqrt()
        y: float= self.row1.xyz().length_sqrt()
        z: float= self.row2.xyz().length_sqrt()

        return Vec3(x, y, z)

    def multiply_v4(self, v4: Vec4) -> Vec4:
        """Return matrix multiplyed by vec4
        """
        x: float= v4.dot(self.col0())
        y: float= v4.dot(self.col1())
        z: float= v4.dot(self.col2())
        w: float= v4.dot(self.col3())

        return Vec4(x, y, z, w)

    def col0(self) -> Vec4:
        """Return matrix column
        """
        x: float = self.row0.x
        y: float = self.row1.x
        z: float = self.row2.x
        w: float = self.row3.x

        return Vec4(x, y, z, w)

    def col1(self) -> Vec4:
        """Return matrix column
        """
        x: float = self.row0.y
        y: float = self.row1.y
        z: float = self.row2.y
        w: float = self.row3.y

        return Vec4(x, y, z, w)

    def col2(self) -> Vec4:
        """Return matrix column
        """
        x: float = self.row0.z
        y: float = self.row1.z
        z: float = self.row2.z
        w: float = self.row3.z

        return Vec4(x, y, z, w)

    def col3(self) -> Vec4:
        """Return matrix column
        """
        x: float = self.row0.w
        y: float = self.row1.w
        z: float = self.row2.w
        w: float = self.row3.w

        return Vec4(x, y, z, w)

    def sum(self) -> float:
        """Return sum of all rows
        """
        return (
            self.row0.sum() +
            self.row1.sum() +
            self.row2.sum() +
            self.row3.sum()
        )

    def get_at(self, row: int, col: int) -> float:
        if row == 0:
            return self.row0[col]
        if row == 1:
            return self.row1[col]
        if row == 2:
            return self.row2[col]
        if row == 3:
            return self.row3[col]

        raise Mat4Error('out of range')

    def set_at(self, row: int, col: int, value: float) -> float:
        if row == 0:
            self.row0.set_at(col, value)
            return

        if row == 1:
            self.row1.set_at(col, value)
            return

        if row == 2:
            self.row2.set_at(col, value)
            return

        if row == 3:
            self.row3.set_at(col, value)
            return

        raise Mat4Error('out of range')

    def determinant(self) -> float:
        """Return matrix determinant value
        """
        a00: float= self.row0.x
        a01: float= self.row0.y
        a02: float= self.row0.z
        a03: float= self.row0.w
        a10: float= self.row1.x
        a11: float= self.row1.y
        a12: float= self.row1.z
        a13: float= self.row1.w
        a20: float= self.row2.x
        a21: float= self.row2.y
        a22: float= self.row2.z
        a23: float= self.row2.w
        a30: float= self.row3.x
        a31: float= self.row3.y
        a32: float= self.row3.z
        a33: float= self.row3.w

        b00: float= a30 * a21 * a12 * a03
        b01: float= a20 * a31 * a12 * a03
        b02: float= a30 * a11 * a22 * a03
        b03: float= a10 * a31 * a22 * a03
        b10: float= a20 * a11 * a32 * a03
        b11: float= a10 * a21 * a32 * a03
        b12: float= a30 * a21 * a02 * a13
        b13: float= a20 * a31 * a02 * a13
        b20: float= a30 * a01 * a22 * a13
        b21: float= a00 * a31 * a22 * a13
        b22: float= a20 * a01 * a32 * a13
        b23: float= a00 * a21 * a32 * a13
        b30: float= a30 * a11 * a02 * a23
        b31: float= a10 * a31 * a02 * a23
        b32: float= a30 * a01 * a12 * a23
        b33: float= a00 * a31 * a12 * a23
        b40: float= a10 * a01 * a32 * a23
        b41: float= a00 * a11 * a32 * a23
        b42: float= a20 * a11 * a02 * a33
        b43: float= a10 * a21 * a02 * a33
        b50: float= a20 * a01 * a12 * a33
        b51: float= a00 * a21 * a12 * a33
        b52: float= a10 * a01 * a22 * a33
        b53: float= a00 * a11 * a22 * a33

        return(
            b00 - b01 - b02 + b03 +
            b10 - b11 - b12 + b13 +
            b20 - b21 - b22 + b23 +
            b30 - b31 - b32 + b33 +
            b40 - b41 - b42 + b43 +
            b50 - b51 - b52 + b53
        )

    def trace(self) -> float:
        """Return matrix trace value
        """
        return self.row0.x + self.row1.y + self.row2.z + self.row3.w

    def array(self) -> list[float]:
        """
        """
        return [
            self.row0.x, self.row0.y, self.row0.z, self.row0.w,
            self.row1.x, self.row1.y, self.row1.z, self.row1.w,
            self.row2.x, self.row2.y, self.row2.z, self.row2.w,
            self.row3.x, self.row3.y, self.row3.z, self.row3.w
        ]

    def multi_array(self) -> list[list[float]]:
        """
        """
        return [
            [self.row0.x, self.row0.y, self.row0.z, self.row0.w],
            [self.row1.x, self.row1.y, self.row1.z, self.row1.w],
            [self.row2.x, self.row2.y, self.row2.z, self.row2.w],
            [self.row3.x, self.row3.y, self.row3.z, self.row3.w]
        ]


# --- COORDS


class PolarCoord:

    __slots__= ('rho', 'theta')

    def __init__(self, rho: float= 0.0, theta: float= 0.0) -> None:
        self.rho: float= rho
        self.theta: float= theta

    @staticmethod
    def from_coords(x: float, y: float) -> 'PolarCoord':
        r: float= sqrt(sqr(x) + sqr(y))
        t: float= arctan2(y, x)

        return PolarCoord(
            rho= r,
            theta= t
        )

    def to_coords(self) -> Vec2:
        x: float= self.rho * cos(self.theta)
        y: float= self.rho * sin(self.theta)
        return Vec2(x, y)


class SphericalCoord:

    __slots__= ('rho', 'theta', 'phi')

    def __init__(self, rho: float= 0.0, theta: float= 0.0, phi: float= 0.0) -> None:
        self.rho: float= rho
        self.theta: float= theta
        self.phi: float= phi

    @staticmethod
    def from_coords(x: float, y: float, z: float) -> 'SphericalCoord':
        r: float= sqrt(sqr(x) + sqr(y) + sqr(z))
        p: float= arctan2(sqrt(sqr(x) + sqr(y)), z)
        t: float= arctan2(y, x)
        return SphericalCoord(r, t, p)

    def to_coords(self) -> Vec3:
        # x: float= self.rho * sin(self.phi) * cos(self.theta),
        # y: float= self.rho * sin(self.phi) * sin(self.theta),
        # z: float= self.rho * cos(self.phi)
        # return Vec3(x, y, z)

        x: float= self.rho * cos(self.phi) * cos(self.theta)
        y: float= self.rho * cos(self.phi) * sin(self.theta)
        z: float= self.rho * sin(self.phi)
        return Vec3(x, y, z)


# --- QUATERNION


class QuatError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class Quaternion:

    __slots__= ('x', 'y', 'z', 'w')

    def __init__(
        self,
        x: float= 0.0,
        y: float= 0.0,
        z: float= 0.0,
        w: float= 0.0,
    ) -> None:
        self.x: float= x
        self.y: float= y
        self.z: float= z
        self.w: float= w

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            raise QuatError('not of type Quaternion')

        return Quaternion(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w
        )

    def __sub__(self, other):
        if not isinstance(other, Quaternion):
            raise QuatError('not of type Quaternion')

        return Quaternion(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w
        )

    def __mul__(self, other):
        if not isinstance(other, Quaternion):
            raise QuatError('not of type Quaternion')

        x1: float= self.x
        y1: float= self.y
        z1: float= self.z
        w1: float= self.w

        x2: float= other.x
        y2: float= other.y
        z2: float= other.z
        w2: float= other.w

        cx: float= y1 * z2 - z1 * y2
        cy: float= z1 * x2 - x1 * z2
        cz: float= x1 * y2 - y1 * x2

        dt: float= x1 * x2 + y1 * y2 + z1 * z2

        return Quaternion(
            x1 * w2 + x2 * w1 + cx,
            y1 * w2 + y2 * w1 + cy,
            z1 * w2 + z2 * w1 + cz,
            w1 * w2 - dt
        )

    def __str__(self) -> str:
        return f'Q({self.x}, {self.y}, {self.z}, {self.w})'

    @staticmethod
    def create_from_vec4(v4: Vec4) -> 'Quaternion':
        return Quaternion(v4.x, v4.y, v4.z, v4.w)

    @staticmethod
    def create_from_euler(v3: Vec3) -> 'Quaternion':
        """Create from euler angles
        """
        v0: Vec3= v3 * 0.5

        sx: float= sin(to_rad(v0.x))
        cx: float= cos(to_rad(v0.x))
        sy: float= sin(to_rad(v0.y))
        cy: float= cos(to_rad(v0.y))
        sz: float= sin(to_rad(v0.z))
        cz: float= cos(to_rad(v0.z))

        return Quaternion(
            (sx * cy * cz) + (cx * sy * sz),
            (cx * sy * cz) + (sx * cy * sz),
            (cx * cy * sz) - (sx * sy * cz),
            (cx * cy * cz) - (sx * sy * sz)
        )

    @staticmethod
    def create_from_axis(angle_deg: float, unit_axis: Vec3) -> 'Quaternion':
        """Create a quaternion from an angle and axis of rotation
        """
        if not unit_axis.is_unit():
            unit_axis.to_unit()

        angle_rad: float= to_rad(angle_deg * 0.5)
        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Quaternion(
            unit_axis.x * s,
            unit_axis.y * s,
            unit_axis.z * s,
            c
        )

    @staticmethod
    def create_rotate_to(start: Vec3, to: Vec3) -> 'Quaternion':
        """Return rotation between two vec3's
        """
        dot: float= start.dot(to)

        if dot < -1.0 + EPSILON:
            v3: Vec3= Vec3.create_unit_x().cross(start)

            if v3.length_sqrt() < EPSILON:
                v3= Vec3.create_unit_y().cross(start)

            v3.to_unit()
            return Quaternion.create_from_axis(to_deg(PI), v3)

        if dot > absf(-1.0 + EPSILON):
            return Quaternion(w=1.0)

        v3: Vec3= start.cross(to)
        return Quaternion(v3.x, v3.y, v3.z, 1.0 + dot)

    def lerp(self, to: 'Quaternion', weight: float) -> 'Quaternion':
        """Return the interpolation between two quaternions
        """
        return Quaternion(
            lerp(self.x, to.x, weight),
            lerp(self.y, to.y, weight),
            lerp(self.z, to.z, weight),
            lerp(self.w, to.w, weight)
        )

    def nlerp(self, to: 'Quaternion', weight: float) -> 'Quaternion':
        """Return normalized lerp
        """
        result= self.lerp(to, weight)

        if not result.is_unit():
            result.to_unit()

        return result

    def slerp(self, to: 'Quaternion', weight: float) -> 'Quaternion':
        """Return the spherical linear interpolation between two quaternions
        """
        weight0: float= 0.0
        weight1: float= 0.0
        cosine: float= self.dot(to)

        if cosine < 0.0:
            cosine *= -1.0
            to.scale(-1.0)

        if cosine < 0.99:
            omega: float= arccos(cosine)
            sinom: float= sin(omega)
            inv_sinom: float= 1.0 / sinom
            weight0= sin(omega * (1.0 - weight)) * inv_sinom
            weight1= sin(omega * weight) * inv_sinom
        else:
            weight0= 1.0 - weight
            weight1= weight

        result= Quaternion(
            weight0 * self.x + weight1 * to.x,
            weight0 * self.y + weight1 * to.y,
            weight0 * self.z + weight1 * to.z,
            weight0 * self.w + weight1 * to.w
        )

        if not result.is_unit():
            result.to_unit()

        return result

    def copy(self) -> 'Quaternion':
        """Return a copy of self
        """
        return Quaternion(self.x, self.y, self.z, self.z)

    def get_at(self, idx: int) -> float:
        match idx:
            case 0:
                return self.x
            case 1:
                return self.y
            case 2:
                return self.z
            case 3:
                return self.w
        raise QuatError('out of range')

    def set_at(self, idx: int, value: float) -> None:
        match idx:
            case 0:
                self.x= value
            case 1:
                self.y= value
            case 2:
                self.z= value
            case 3:
                self.w= value

    def set_from(self, other: 'Quaternion') -> None:
        """Set values based on other quaternions values
        """
        self.w= other.w
        self.x= other.x
        self.y= other.y
        self.z= other.z

    def conjugate(self) -> 'Quaternion':
        """Return the conjugate of self
        """
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def scale(self, by: float) -> 'Quaternion':
        """Return a scaled copy of self based of *by* float value
        """
        return Quaternion(
            self.x * by,
            self.y * by,
            self.z * by,
            self.w * by
        )

    def inverse(self) -> 'Quaternion':
        """Return the invserse of self
        """
        inv: float= 1.0 / self.length_sqr()
        return Quaternion(
            -self.x * inv,
            -self.y * inv,
            -self.z * inv,
            self.q * inv
        )

    def unit(self) -> 'Quaternion':
        """Return a copy of self with normalized length

        Raises
        ---
        QuatError
            length of self was zero
        """
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            raise QuatError('length was zero')

        return self.scale(inv_sqrt(lsq))

    def to_unit(self) -> None:
        """Convert to unit length
        """
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            return

        inv: float= inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv
        self.z *= inv
        self.w *= inv

    def to_euler(self) -> Vec3:
        """Return self as euler values
        """
        threshold: float= 0.4999995

        w2: float= sqr(self.w)
        x2: float= sqr(self.x)
        y2: float= sqr(self.y)
        z2: float= sqr(self.z)

        len_sq: float= x2 + y2 + z2 + w2
        test: float= (self.x * self.z) + (self.w * self.y)

        if test > threshold * len_sq:
            return Vec3(0.0, PHI, 2.0 * arctan2(self.x, self.y))

        if test < -threshold * len_sq:
            return Vec3(0.0, -PHI, -2.0 * arctan2(self.x, self.w))

        xy: float= 2.0 * ((self.w * self.x) - (self.y * self.z))
        xx: float= w2 - x2 - y2 + z2
        zy: float= 2.0 * ((self.w * self.z) - (self.z * self.y))
        zx: float= w2 + x2 - y2 - z2

        x: float= arctan2(xy, xx)
        y: float= arcsin(2.0 * test / len_sq)
        z: float= arctan2(zy, zx)

        return Vec3(x, y, z)

    def to_axis(self) -> Vec3:
        """Return self as axis angles
        """
        qt_cpy: Quaternion= self.copy()

        if qt_cpy.w > 1.0:
            qt_cpy.to_unit()

        scl: float= sqrt(1.0 - sqr(qt_cpy.w))

        if is_zero(scl):
            raise QuatError('length of quaternion was zero')

        inv: float = 1.0 / scl
        return Vec3(
            self.x * inv,
            self.y * inv,
            self.z * inv
        )

    def xyz(self) -> Vec3:
        """Return 'xyz' components as a vec3
        """
        return Vec3(self.x, self.y, self.z)

    def to_mat4(self) -> Mat4:
        """Create a mat4 matrix based on quaternion's current values
        """
        x2: float= sqr(self.x)
        y2: float= sqr(self.y)
        z2: float= sqr(self.z)
        w2: float= sqr(self.w)

        xy: float= self.x * self.y
        xz: float= self.x * self.z
        xw: float= self.x * self.w

        yz: float= self.y * self.z
        yw: float= self.y * self.w

        zw: float= self.z * self.w

        s2: float= 2.0 / (x2 + y2 + z2 + w2)

        ax: float= 1.0 - (s2 * (y2 + z2))
        ay: float= s2 * (xy + zw)
        az: float= s2 * (xz - yw)
        aw: float= 0.0
        bx: float= s2 * (xy - zw)
        by: float= 1.0 - (s2 * (x2 + z2))
        bz: float= s2 * (yz + xw)
        bw: float= 0.0
        cx: float= s2 * (xz + yw)
        cy: float= s2 * (yz - xw)
        cz: float= 1.0 - (s2 * (x2 + y2))
        cw: float= 0.0
        dx: float= 0.0
        dy: float= 0.0
        dz: float= 0.0
        dw: float= 1.0

        return Mat4.create_from_values(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw
        )

    def length_sqr(self) -> float:
        """Return the squared length
        """
        return sqr(self.x) + sqr(self.y) + sqr(self.z) + sqr(self.w)

    def length_sqrt(self) -> float:
        """Return the square root length
        """
        return sqrt(self.length_sqr())

    def dot(self, other: 'Quaternion') -> float:
        """Return the dot product between self and other
        """
        x2: float= self.x * other.x
        y2: float= self.y * other.y
        z2: float= self.z * other.z
        w2: float= self.w * other.w

        return x2 + y2 + z2 + w2

    def get_axis_angle_rotation(self) -> float:
        """Return the current angle of self in radians
        """
        return arccos(self.w) * 2.0

    def get_rotation_axis(self) -> Vec3:
        """Return the current axis
        """
        s: float= sin(self.get_axis_angle() * 0.5)
        if not is_zero(s):
            inv: float= 1.0 / s
            return Vec3(
                self.x * inv,
                self.y * inv,
                self.z * inv
            )

        return Vec3.create_unit_x()

    def is_equil(self, other: 'Quaternion') -> bool:
        """Check if the components of self are the same as other's components
        """
        return (
            is_equil(self.x, other.x) and
            is_equil(self.y, other.y) and
            is_equil(self.z, other.z) and
            is_equil(self.w, other.w)
        )

    def is_unit(self) -> bool:
        """Check if the length of self is one
        """
        return is_one(self.length_sqr())
