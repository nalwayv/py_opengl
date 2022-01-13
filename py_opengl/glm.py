"""
GL MATH
---


Classes
---
Vec2 :
    vec2(x, y)

Vec3 :
    vec3(x, y, z)

Mat3 :
    matrix 3x3

Mat4 :
    matrix 4x4

Plain :
    direction and normal

Quaternion :
    quaternion(x, y, z, w)
"""
import math
from dataclasses import dataclass

PI: float = 3.14159265358979323846
PIOVER2: float = 1.57079632679489661923
TAU: float = 6.28318530717958647693
EPSILON: float = 0.00000000000000022204
INFINITY: float = float('inf')
NEGATIVE_INFINITY: float = float('-inf')


def is_zero(val: float) -> bool:
    """Check if values is zero

    Parameters
    ---
    val : float
        check this float value

    Returns
    ---
    bool
        is zero or not
    """
    return abs(val) <= EPSILON


def is_one(val: float) -> bool:
    """Check if value is one

    Parameters
    ---
    val : float
        check this float value

    Returns
    ---
    bool
        if one or not
    """
    return is_zero(val - 1.0)


def is_equil(x: float, y: float) -> bool:
    """Check if two float values equil one another

    Parameters
    ---
    x : float
        first float value
    y : float
        seconf float value

    Returns
    ---
    bool
        are equil or not
    """
    return abs(x - y) <= EPSILON


def to_radians(val: float) -> float:
    """Return value in radians

    Parameters
    ---
    val : float
        value in degreese

    Returns
    ---
    float
        value in radians
    """
    return val * 0.01745329251994329577


def to_degreese(val: float) -> float:
    """Return value in degreese

    Parameters
    ---
    val : float
        value in radians

    Returns
    ---
    float
        value in degreese
    """
    return val * 57.2957795130823208768


def sqr(val: float) -> float:
    """Return value passed in squared with itself

    Parameters
    ---
    val : float
        value to be squared

    Returns
    ---
    float
        squared result
    """     
    return val * val


def sqrt(val: float) -> float:
    """Return the sqrt of value passed in

    Parameters
    ---
    val : float
        value to find the sqrt of

    Returns
    ---
    float
        its sqrt value
    """
    return math.sqrt(val)


def inv_sqrt(val: float) -> float:
    """Return the inverse sqrt of the value passed in

    Parameters
    ---
    val : float
        value to find the inv sqrt of

    Example
    ---
    ```python
    inv = inv_sqrt(25)
    print(25 * inv)
    ```

    Returns
    ---
    float
        its inv sqrt value
    """
    return 1.0 / sqrt(val)


def clamp(
        val: float | int,
        low: float | int,
        high: float | int) -> float | int:
    """Clamp value between low and high

    Returns
    ---
    float | int
        clamped value
    """
    return max(low, min(val, high))


def lerp(start: float, end: float, weight: float) -> float:
    """Return amount of *linear interpolation* between start and end based on weight

    Parameters
    ---
    start : float
        from
    end : float
        to
    weight : float
        value between 0 and 1

    Returns
    ---
    float
        lerp amount
    """
    return start + clamp(weight, 0, 1) * (end - start)


def normalize(val: float, low: float, high: float) -> float:
    """Normalize value between low and high

    Parameters
    ---
    val : float
        value to be normalized
    low : float
        low value
    high : float
        high value

    Returns
    ---
    float
        normalized value
    """
    return (val - low) / (high - low)


def tan(val: float) -> float:
    """Return *trigonometry tan* value

    Parameters
    ---
    val : float
        value to find tan of

    Returns
    ---
    float
        the tangent of value in radians
    """
    return math.tan(val)


def sin(val: float) -> float:
    """Return *trigonometry sin* value

    Parameters
    ---
    val : float

    Returns
    ---
    float
        the sine of value in radians
    """
    return math.sin(val)


def cos(val: float) -> float:
    """Return *trigonometry cos* value

    Parameters
    ---
    val : float

    Returns
    ---
    float
        the cosine of value in radians
    """
    return math.cos(val)


def arccos(val: float) -> float:
    """Return the arc cosine value

    Parameters
    ---
    val : float
        value has to be between -1 and 1

    Returns
    ---
    float
        the value in radians between 0 and PI\n
        -1 = pi\n
         1 = 0
    """
    return math.acos(val)


def arcsin(val: float) -> float:
    """Return the arc aign value

    Parameters
    ---
    val : float
        value has to be between -1 and 1

    Returns
    ---
    float
        the value in radians between 0 and PI\n
        -1 = -pi/2
         1 = pi/2
    """
    return math.asin(val)


def arctan2(y: float, x: float) -> float:
    """Return the arc tangent of value y / x

    Parameters
    ---
    y : float
        
    x : float

    Returns
    ---
    float
        the value in radians
    """
    return math.atan2(y, x)


# --- VECTOR2(X, Y)


class Vec2Error(Exception):
    '''Custom error for Vec2'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __getitem__(self, idx):
        match clamp(idx, 0, 1):
            case 0: return self.x
            case 1: return self.y
            case _: raise Vec2Error('out of range')

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
        if not isinstance(other, (float, int)):
            raise Vec2Error('other was not of type float or int')
        x: float = self.x * other
        y: float = self.y * other
        return Vec2(x, y)

    @staticmethod
    def one() -> 'Vec2':
        return Vec2(1.0, 1.0)

    def to_unit(self) -> None:
        """Normalize length

        Raises
        ---
        Vec2Error
            if current length is zero
        """
        lsq: float = self.length_sqr()

        if is_zero(lsq):
            raise Vec2Error('length of this vec2 was zero')

        inv = inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv

    def unit(self) -> 'Vec2':
        """Return a copy of self with a normal length

        Returns
        ---
        Vec2
            a copy byt with a normalized length
        """
        lsq: float = self.length_sqr()
        if is_zero(lsq):
            return self.copy()
        return self * inv_sqrt(lsq)

    def copy(self) -> 'Vec2':
        """Return a copy of the self

        Returns
        ---
        Vec2
            a copy
        """ 
        return Vec2(self.x, self.y)

    def cross(self, other: 'Vec2') -> float:
        """Return the cross product between self and another vec2

        Returns
        ---
        float
            cross product between self and other
        """
        return (self.y * other.z) - (self.z * other.y)

    def project(self, other: 'Vec2') -> 'Vec2':
        """Return the projection between self and other vec3

        Returns
        ---
        Vec2
        """
        return other * (self.dot(other) / other.length_sqr())

    def reject(self, other: 'Vec2') -> 'Vec2':
        """Return the reject between self and other vec3

        Returns
        ---
        Vec2
        """
        return self - self.project(other)

    def length_sqr(self) -> float:
        """Return the squared length

        Returns
        ---
        float
        """
        return sqr(self.x) + sqr(self.y) + sqr(self.z)

    def length_sqrt(self) -> float:
        """Return the square root length

        Returns
        ---
        float
        """
        return sqrt(self.length_sqr())

    def distance(self, other: 'Vec2') -> float:
        """Return the distance between self and other vec2

        Parameters
        ---
        other : Vec2

        Returns
        ---
        float
        """
        dir: Vec2 = other - self
        return dir.length_sqrt()

    def dot(self, other: 'Vec2') -> float:
        """Return the dot product between self and other vec3

        Parameters
        ---
        other : Vec2

        Returns
        ---
        float
        """
        return (self.x * other.x) + (self.y * other.y)

    def is_unit(self) -> bool:
        """Check if the current length of self is normalized

        Returns
        ---
        bool
        """
        return is_one(self.length_sqr())

    def is_zero(self) -> bool:
        """Check if the current *x, y* components of self are zero in value

        Returns
        ---
        bool
        """
        return is_zero(self.x) and is_zero(self.y)

    def is_equil(self, other: 'Vec2') -> bool:
        """Check if self and other have the same *x, y* component values

        Parameters
        ---
        other : Vec2

        Returns
        ---
        bool
        """
        return is_equil(self.x, other.x) and is_equil(self.y, other.y)


# --- VECTOR3(X, Y, Z)


class Vec3Error(Exception):
    '''Custom error for Vec3'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __getitem__(self, idx):
        match clamp(idx, 0, 2):
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
        if not isinstance(other, (float, int)):
            raise Vec3Error('other was not of type float or int')
        x: float = self.x * other
        y: float = self.y * other
        z: float = self.z * other
        return Vec3(x, y, z)

    @staticmethod
    def one() -> 'Vec3':
        return Vec3(1.0, 1.0, 1.0)

    @staticmethod
    def from_v2(v2: Vec2) -> 'Vec3':
        return Vec3(x=v2.x, y=v2.y)

    def to_unit(self) -> None:
        """Normalize length

        Raises
        ---
        Vec3Error
            if current length is zero
        """
        lsq: float = self.length_sqr()

        if is_zero(lsq):
            raise Vec3Error('length of this vec3 was zero')

        inv = inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv
        self.z *= inv

    def unit(self) -> 'Vec3':
        """Return a copy of this vec3 with a normal length

        Returns
        ---
        Vec3
            a copy byt with a normalized length
        """
        lsq: float = self.length_sqr()
        if is_zero(lsq):
            return self.copy()
        return self * inv_sqrt(lsq)

    def copy(self) -> 'Vec3':
        """Return a copy of the vec3

        Returns
        ---
        Vec3
            a copy
        """ 
        return Vec3(self.x, self.y, self.z)

    def cross(self, other: 'Vec3') -> 'Vec3':
        """Return the cross product between self and another vec3

        Returns
        ---
        Vec3
            cross product between self and other
        """
        return Vec3(
            (self.y * other.z) - (self.z * other.y),
            (self.z * other.x) - (self.x * other.z),
            (self.x * other.y) - (self.y * other.x))

    def project(self, other: 'Vec3') -> 'Vec3':
        """Return the projection between self and other vec3

        Returns
        ---
        Vec3
        """
        return other * (self.dot(other) / other.length_sqr())

    def reject(self, other: 'Vec3') -> 'Vec3':
        """Return the reject between self and other vec3

        Returns
        ---
        Vec3
        """
        return self - self.project(other)

    def length_sqr(self) -> float:
        """Return the squared length

        Returns
        ---
        float
        """
        return sqr(self.x) + sqr(self.y) + sqr(self.z)

    def length_sqrt(self) -> float:
        """Return the square root length

        Returns
        ---
        float
        """
        return sqrt(self.length_sqr())

    def distance(self, other: 'Vec3') -> float:
        """Return the distance between self and other vec3

        Parameters
        ---
        other : Vec3

        Returns
        ---
        float
        """
        dir: Vec3 = other - self
        return dir.length_sqrt()

    def dot(self, other: 'Vec3') -> float:
        """Return the dot product between self and other vec3

        Parameters
        ---
        other : Vec3

        Returns
        ---
        float
        """
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def is_unit(self) -> bool:
        """Check if the current length of self is normalized

        Returns
        ---
        bool
        """
        return is_one(self.length_sqr())

    def is_zero(self) -> bool:
        """Check if the current *x, y, z* components of self are zero in value

        Returns
        ---
        bool
        """
        return is_zero(self.x) and is_zero(self.y) and is_zero(self.z)

    def is_equil(self, other: 'Vec3') -> bool:
        """Check if self and other have the same *x, y, z* component values

        Parameters
        ---
        other : Vec3

        Returns
        ---
        bool
        """
        return is_equil(self.x, other.x) and is_equil(self.y, other.y) and is_equil(self.z, other.z)


# --- VECTOR4(X, Y, Z, W)


class Vec4Error(Exception):
    '''Custom error for Vec4'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Vec4:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0

    def __getitem__(self, idx):
        match clamp(idx, 0, 3):
            case 0: return self.x
            case 1: return self.y
            case 2: return self.z
            case 3: return self.w
            case _: raise Vec4Error('out of range')

    def __add__(self, other):
        if not isinstance(other, Vec4):
            raise Vec4Error('not of type Vec4')
        x: float = self.x + other.x
        y: float = self.y + other.y
        z: float = self.z + other.z
        w: float = self.w + other.w
        return Vec4(x, y, z, w)

    def __sub__(self, other):
        if not isinstance(other, Vec4):
            raise Vec4Error('not of type Vec4')
        x: float = self.x - other.x
        y: float = self.y - other.y
        z: float = self.z - other.z
        w: float = self.w - other.w
        return Vec4(x, y, z, w)

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise Vec4Error('other was not of type float or int')
        x: float = self.x * other
        y: float = self.y * other
        z: float = self.z * other
        w: float = self.w * other
        return Vec4(x, y, z, w)

    @staticmethod
    def one() -> 'Vec4':
        return Vec4(1.0, 1.0, 1.0, 1.0)

    def copy(self) -> 'Vec4':
        """Return a copy of the vec4

        Returns
        ---
        Vec4
            a copy
        """ 
        return Vec4(self.x, self.y, self.z, self.w)

    def to_unit(self) -> None:
        """Normalize length

        Raises
        ---
        Vec4Error
            if current length is zero
        """
        lsq: float = self.length_sqr()

        if is_zero(lsq):
            raise Vec4Error('length of this vec4 was zero')

        inv = inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv
        self.z *= inv
        self.w *= inv

    def unit(self) -> 'Vec4':
        """Return a copy of this vec4 with a normal length

        Returns
        ---
        Vec4
            a copy with a normalized length
        """
        lsq: float = self.length_sqr()
        if is_zero(lsq):
            return self.copy()
        return self * inv_sqrt(lsq)

    def length_sqr(self) -> float:
        """Return the squared length

        Returns
        ---
        float
        """
        return sqr(self.x) + sqr(self.y) + sqr(self.z) + sqr(self.w)

    def length_sqrt(self) -> float:
        """Return the square root length

        Returns
        ---
        float
        """
        return sqrt(self.length_sqr())

    def distance(self, other: 'Vec4') -> float:
        """Return the distance between self and other vec4

        Parameters
        ---
        other : Vec4

        Returns
        ---
        float
        """
        dir: Vec4 = other - self
        return dir.length_sqrt()

    def dot(self, other: 'Vec4') -> float:
        """Return the dot product between self and other vec4

        Parameters
        ---
        other : Vec4

        Returns
        ---
        float
        """
        return (
            (self.x * other.x) +
            (self.y * other.y) +
            (self.z * other.z) +
            (self.w * other.w)
        )

    def is_unit(self) -> bool:
        """Check if the current length of self is normalized

        Returns
        ---
        bool
        """
        return is_one(self.length_sqr())

    def is_zero(self) -> bool:
        """Check if the current *x, y, z* components of self are zero in value

        Returns
        ---
        bool
        """
        return is_zero(self.x) and is_zero(self.y) and is_zero(self.z)

    def is_equil(self, other: 'Vec4') -> bool:
        """Check if self and other have the same *x, y, z, w* component values

        Parameters
        ---
        other : Vec4

        Returns
        ---
        bool
        """
        return (
            is_equil(self.x, other.x) and
            is_equil(self.y, other.y) and
            is_equil(self.z, other.z) and
            is_equil(self.w, other.w)
        )


# --- MATRIX3
class Mat3Error(Exception):
    '''Custom error for matrix 3x3'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Mat3:
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    bx: float = 0.0
    by: float = 0.0
    bz: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    cz: float = 0.0

    def __getitem__(self, idx):
        match clamp(idx, 0, 8):
            case 0: return self.ax
            case 1: return self.ay
            case 2: return self.az
            case 3: return self.bx
            case 4: return self.by
            case 5: return self.bz
            case 6: return self.cx
            case 7: return self.cy
            case 8: return self.cz
            case _: raise Mat3Error('out of range')

    def __add__(self, other):
        if not isinstance(other, Mat3):
            raise Mat3Error('not of type mat3')

        ax: float = self.ax + other.ax
        ay: float = self.ay + other.ay
        az: float = self.az + other.az
        bx: float = self.bx + other.bx
        by: float = self.by + other.by
        bz: float = self.bz + other.bz
        cx: float = self.cx + other.cx
        cy: float = self.cy + other.cy
        cz: float = self.cz + other.cz

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz,
        )

    def __sub__(self, other):
        if not isinstance(other, Mat3):
            raise Mat3Error('not of type mat3')

        ax: float = self.ax - other.ax
        ay: float = self.ay - other.ay
        az: float = self.az - other.az
        bx: float = self.bx - other.bx
        by: float = self.by - other.by
        bz: float = self.bz - other.bz
        cx: float = self.cx - other.cx
        cy: float = self.cy - other.cy
        cz: float = self.cz - other.cz

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz,
        )


    def __mul__(self, other):
        if not isinstance(other, Mat3):
            raise Mat3Error('not of type Mat3')

        ax: float = (
            self.ax * other.ax +
            self.ay * other.bx +
            self.az * other.cx
        )
        ay: float = (
            self.ax * other.ay +
            self.ay * other.by +
            self.az * other.cy
        )
        az: float = (
            self.ax * other.az +
            self.ay * other.bz +
            self.az * other.cz
        )
        bx: float = (
            self.bx * other.ax +
            self.by * other.bx +
            self.bz * other.cx
        )
        by: float = (
            self.bx * other.ay +
            self.by * other.by +
            self.bz * other.cy
        )
        bz: float = (
            self.bx * other.az +
            self.by * other.bz +
            self.bz * other.cz
        )
        cx: float = (
            self.cx * other.ax +
            self.cy * other.bx +
            self.cz * other.cx
        )
        cy: float = (
            self.cx * other.ay +
            self.cy * other.by +
            self.cz * other.cy
        )
        cz: float = (
            self.cx * other.az +
            self.cy * other.bz +
            self.cz * other.cz
        )

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz,
        )

    @staticmethod
    def identity() -> 'Mat3':
        """Create an identity mat3 matrix

        Returns
        ---
        Mat3
        """
        return Mat3(ax=1.0, by=1.0, cz=1.0)

    @staticmethod
    def create_scaler(v3: Vec3) -> 'Mat3':
        """Create a scaler mat3 matrix

        Returns
        ---
        Mat3
        """
        return Mat3(ax=v3.x, by=v3.y, cz=v3.z)
    
    @staticmethod
    def create_x_rotation(angle_deg: float) -> 'Mat3':
        """Create a rotation *X* mat3 matrix

        Returns
        ---
        Mat3
        """
        angle_rad: float = to_radians(angle_deg)

        c: float = cos(angle_rad)
        s: float = sin(angle_rad)

        return Mat3(ax=1.0, by=c, bz=-s, cy=s, cz=c)

    @staticmethod
    def create_y_rotation(angle_deg: float) -> 'Mat3':
        """Create a rotation *y* mat3 matrix

        Returns
        ---
        Mat3
        """
        angle_rad: float = to_radians(angle_deg)

        c: float = cos(angle_rad)
        s: float = sin(angle_rad)

        return Mat3(ax=c, az=s, by=1.0, cx=-s, cz=c)

    @staticmethod
    def create_z_rotation(angle_deg: float) -> 'Mat3':
        """Create a rotation *z* mat3 matrix

        Returns
        ---
        Mat3
        """
        angle_rad: float = to_radians(angle_deg)

        c: float = cos(angle_rad)
        s: float = sin(angle_rad)

        return Mat3(ax=c, ay=-s, bx=s, by=c, cz=1.0)

    def copy(self) -> 'Mat3':
        """Return a copy

        Returns
        ---
        Mat3
        """
        return Mat3(
            self.ax, self.ay, self.az,
            self.bx, self.by, self.bz,
            self.cx, self.cy, self.cz
        )

    def scale(self, by: float) -> 'Mat3':
        """Return a scaled copy of self

        Returns
        ---
        Mat3
        """
        ax: float = self.ax * by
        ay: float = self.ay * by
        az: float = self.az * by

        bx: float = self.bx * by
        by: float = self.by * by
        bz: float = self.bz * by

        cx: float = self.cx * by
        cy: float = self.cy * by
        cz: float = self.cz * by

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz,
        )

    def transpose(self) -> 'Mat3':
        """Return a transposed copy of self

        Returns
        ---
        Mat3
        """
        ax: float = self.ax
        ay: float = self.bx
        az: float = self.cx
        bx: float = self.ay
        by: float = self.by
        bz: float = self.cy
        cx: float = self.az
        cy: float = self.bz
        cz: float = self.cz

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz,
        )

    def cofactor(self) -> 'Mat3':
        """Return a cofactor copy of self

        Returns
        ---
        Mat3
        """
        ax: float = self.ax
        ay: float = -self.ay
        az: float = self.az
        bx: float = -self.bx
        by: float = self.by
        bz: float = -self.bz
        cx: float = self.cx
        cy: float = -self.cy
        cz: float = self.cz

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz
        )

    def to_unit(self) -> None:
        """Normalize the length of self

        Raises
        ---
        Mat3Error
            if the determinant of self is zero
        """
        det: float = self.determinant()

        if is_zero(det):
            raise Mat3Error('length of this Mat4 was zero')

        inv = 1.0 / det
        self.ax *= inv  
        self.ay *= inv 
        self.az *= inv  
        self.aw *= inv  
        self.bx *= inv  
        self.by *= inv  
        self.bz *= inv  
        self.bw *= inv  
        self.cx *= inv  
        self.cy *= inv  
        self.cz *= inv  
        self.cw *= inv  
        self.dx *= inv  
        self.dy *= inv  
        self.dz *= inv  
        self.dw *= inv  

    def unit(self) -> 'Mat3':
        """Return a copy of self with normalized length

        Returns
        ---
        Mat3
            
        Raises
        ---
        Mat3Error
            if the determinant of self is zero
        """
        det: float = self.determinant()
        if is_zero(det):
            raise Mat3Error('length of this Mat4 was zero')
        return self.scale(1.0 / det)


    def inverse(self) -> 'Mat3':
        """Return the inverse of self

        Returns
        ---
        Mat3
        """
        a: float = self.ax
        b: float = self.ay
        c: float = self.az
        d: float = self.bx
        e: float = self.by
        f: float = self.bz
        g: float = self.cx
        h: float = self.cy
        i: float = self.cz

        det: float = self.determinant()
        inv: float = 1.0 / det

        ax: float = (e * i - h * f) * inv
        ay: float = (g * f - d * i) * inv
        az: float = (d * h - g * e) * inv
        bx: float = (h * c - b * i) * inv
        by: float = (a * i - g * c) * inv
        bz: float = (g * b - a * h) * inv
        cx: float = (b * f - e * c) * inv
        cy: float = (d * c - a * f) * inv
        cz: float = (a * e - d * b) * inv

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz
        )

    def row0(self) -> tuple[float, float, float]:
        """Return the first row of float values

        Returns
        ---
        tuple[float, float, float]
        """
        return (self.ax, self.ay, self.az)

    def row1(self) -> tuple[float, float, float]:
        """Return the second row of float values
        
        Returns
        ---
        tuple[float, float, float]
        """
        return (self.bx, self.by, self.bz)

    def row2(self) -> tuple[float, float, float]:
        """Return the third row of float values

        Returns
        ---
        tuple[float, float, float]
        """
        return (self.cx, self.cy, self.cz)

    def col0(self) -> tuple[float, float, float]:
        """Return the first column of float values

        Returns
        ---
        tuple[float, float, float]
        """
        return (self.ax, self.bx, self.cx)

    def col1(self) -> tuple[float, float, float]:
        """Return the second column of float values

        Returns
        ---
        tuple[float, float, float]
        """
        return (self.ay, self.by, self.cy)

    def col2(self) -> tuple[float, float, float]:
        """Return the third column of float values

        Returns
        ---
        tuple[float, float, float]
        """
        return (self.az, self.bz, self.cz)

    def at(self, row: int, col: int) -> float:
        return self[col * 3 + row]

    def determinant(self) -> float:
        """Return the determinant of self

        Returns
        ---
        float
        """
        a0: float = self.ax * self.by * self.cz
        a1: float = self.ay * self.bz * self.cx
        a2: float = self.az * self.bx * self.cy
        b1: float = self.az * self.by * self.cx
        b2: float = self.ax * self.bz * self.cy
        b3: float = self.ay * self.bx * self.cz

        return a0 + a1 + a2 - b1 - b2 - b3

    def trace(self) -> float:
        """Return the sum of the trace values

        Returns
        ---
        float
        """
        return self.ax + self.by + self.cz

    def array(self) -> list[float]:
        """Return self as a list of floats

        Returns
        ---
        list[float]
        """
        return [
            self.ax, self.ay, self.az,
            self.bx, self.by, self.bz,
            self.cx, self.cy, self.cz,
        ]

    def multi_array(self) -> list[list[float]]:
        """Return self as a list of list floats

        Returns
        ---
        list[list[float]]
        """
        return [
            [self.ax, self.ay, self.az],
            [self.bx, self.by, self.bz],
            [self.cx, self.cy, self.cz]
        ]


# --- MATRIX4


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
            dx, dy, dz, dw
        )

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
            dx, dy, dz, dw
        )

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
            dx, dy, dz, dw
        )

    @staticmethod
    def identity() -> 'Mat4':
        """Create an identity mat4 matrix

        Returns
        ---
        Mat4
        """
        return Mat4(ax=1.0, by=1.0, cz=1.0, dw=1.0)

    @staticmethod
    def create_scaler(v3: Vec3) -> 'Mat4':
        """Create a scaler mat4 matrix

        Returns
        ---
        Mat4
        """
        return Mat4(ax=v3.x, by=v3.y, cz=v3.z, dw=1.0)

    @staticmethod
    def create_translation(v3: Vec3) -> 'Mat4':
        """Create a translation mat4 matrix

        Returns
        ---
        Mat4
        """
        return Mat4(ax=1.0, by=1.0, cz=1.0, dx=v3.x, dy=v3.y, dz=v3.z, dw=1.0)

    @staticmethod
    def create_x_rotation(angle_deg: float) -> 'Mat4':
        """Create a rotation *X* mat4 matrix

        Returns
        ---
        Mat4
        """
        angle_rad: float = to_radians(angle_deg)

        c: float = cos(angle_rad)
        s: float = sin(angle_rad)

        return Mat4(ax=1.0, by=c, bz=-s, cy=s, cz=c, dw=1.0)

    @staticmethod
    def create_y_rotation(angle_deg: float) -> 'Mat4':
        """Create a rotation *y* mat4 matrix

        Returns
        ---
        Mat4
        """
        angle_rad: float = to_radians(angle_deg)

        c: float = cos(angle_rad)
        s: float = sin(angle_rad)

        return Mat4(ax=c, az=s, by=1.0, cx=-s, cz=c, dw=1.0)

    @staticmethod
    def create_z_rotation(angle_deg: float) -> 'Mat4':
        """Create a rotation *z* mat4 matrix

        Returns
        ---
        Mat4
        """
        angle_rad: float = to_radians(angle_deg)

        c: float = cos(angle_rad)
        s: float = sin(angle_rad)

        return Mat4(ax=c, ay=-s, bx=s, by=c, cz=1.0, dw=1.0)

    @staticmethod
    def from_axis(angle_deg: float, axis: Vec3) -> 'Mat4':
        """Create a mat4 matrix from an angle and axis of rotation

        Returns
        ---
        Mat4
        """
        axis_cpy: Vec3 = axis.copy()

        if not axis_cpy.is_unit():
            axis_cpy.to_unit()

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
            dw=1.0
        )

    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3) -> 'Mat4':
        """Create a mat4 matrix *look at* field of view

        Returns
        ---
        Mat4
        """
        z: Vec3 = (eye - target).unit()

        if z.is_zero():
            return Mat4.identity()

        x: Vec3 = up.cross(z).unit()
        y: Vec3 = z.cross(x).unit()

        dx: float = -x.dot(eye)
        dy: float = -y.dot(eye)
        dz: float = -z.dot(eye)

        return Mat4(
            x.x, y.x, z.x, 0.0,
            x.y, y.y, z.y, 0.0,
            x.z, y.z, z.z, 0.0,
            dx, dy, dz, 1.0
        )

    @staticmethod
    def frustum(left: float, right: float, bottom: float, top: float, far: float, near: float) -> 'Mat4':
        """Create a mat4 matrix *frustum* field of view

        Returns
        ---
        Mat4
        """
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
            dz=d, dw=0.0
        )

    @staticmethod
    def ortho(left: float, right: float, bottom: float, top: float, near: float, far: float) -> 'Mat4':
        """Create a mat4 matrix *ortho* field of view

        Returns
        ---
        Mat4
        """
        lr: float = 1.0 / (left - right)
        bt: float = 1.0 / (bottom - top)
        nf: float = 1.0 / (near - far)

        x: float = -2.0 * lr
        y: float = -2.0 * bt
        z: float = 2.0 * nf

        a: float = (right + left) * lr
        b: float = (top + bottom) * bt
        c: float = (far + near) * nf

        return Mat4(ax=x, by=y, cz=z, dx=a, dy=b, dz=c, dw=1.0)

    @staticmethod
    def perspective(fov: float, aspect: float, znear: float, zfar: float) -> 'Mat4':
        """Create a mat4 matrix *perspective projection* field of view

        Returns
        ---
        Mat4

        Raises
        ---
        Mat4Error
            if fov is below zero or fov more then PI
        Mat4Error
            if znear or zfar are less or equil to zero or znear is more or equil to zfar
        """

        if fov <= 0.0 or fov > PI:
            raise Mat4Error('m4 projection fov out of range')

        if znear <= 0.0 or zfar <= 0.0 or znear >= zfar:
            raise Mat4Error('m4 projection aspect out of range')

        fovy: float = 1.0 / tan(fov * 0.5)
        fovx: float = fovy / aspect
        zrange: float = zfar / (zfar - znear)

        return Mat4(ax=fovx, by=fovy, cz=zrange, cw=-1.0, dz=znear*zrange)

    def copy(self) -> 'Mat4':
        """Return a copy

        Returns
        ---
        Mat4
        """
        return Mat4(
            self.ax, self.ay, self.az, self.aw,
            self.bx, self.by, self.bz, self.bw,
            self.cx, self.cy, self.cz, self.cw,
            self.dx, self.dy, self.dz, self.dw
        )

    def scale(self, by: float) -> 'Mat4':
        """Return a scaled copy of self

        Returns
        ---
        Mat4
        """
        ax: float = self.ax * by
        ay: float = self.ay * by
        az: float = self.az * by
        aw: float = self.aw * by
        bx: float = self.bx * by
        by: float = self.by * by
        bz: float = self.bz * by
        bw: float = self.bw * by
        cx: float = self.cx * by
        cy: float = self.cy * by
        cz: float = self.cz * by
        cw: float = self.cw * by
        dx: float = self.dx * by
        dy: float = self.dy * by
        dz: float = self.dz * by
        dw: float = self.dw * by

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw
        )

    def transpose(self) -> 'Mat4':
        """Return a transposed copy of self

        Returns
        ---
        Mat4
        """
        ax: float = self.ax
        ay: float = self.bx
        az: float = self.cx
        aw: float = self.dx
        bx: float = self.ay
        by: float = self.by
        bz: float = self.cy
        bw: float = self.dy
        cx: float = self.az
        cy: float = self.bz
        cz: float = self.cz
        cw: float = self.dz
        dx: float = self.aw
        dy: float = self.bw
        dz: float = self.cw
        dw: float = self.dw

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw
        )

    def cofactor(self) -> 'Mat4':
        """Return a cofactor copy of self

        Returns
        ---
        Mat4
        """
        ax: float = self.ax
        ay: float = -self.ay
        az: float = self.az
        aw: float = -self.aw
        bx: float = -self.bx
        by: float = self.by
        bz: float = -self.bz
        bw: float = self.bw
        cx: float = self.cx
        cy: float = -self.cy
        cz: float = self.cz
        cw: float = -self.cw
        dx: float = -self.dx
        dy: float = self.dy
        dz: float = -self.dz
        dw: float = self.dw

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw,
        )

    def inverse(self) -> 'Mat4':
        """Return the inverse of self

        Returns
        ---
        Mat4
        """
        a00: float = self.ax
        a01: float = self.ay
        a02: float = self.az
        a03: float = self.aw
        a10: float = self.bx
        a11: float = self.by
        a12: float = self.bz
        a13: float = self.bw
        a20: float = self.cx
        a21: float = self.cy
        a22: float = self.cz
        a23: float = self.cw
        a30: float = self.dx
        a31: float = self.dy
        a32: float = self.dz
        a33: float = self.dw

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
            b07 + b05 * b06
        )

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
            dx, dy, dz, dw
        )

    def to_unit(self) -> None:
        """Normalize the length of self

        Raises
        ---
        Mat4Error
            if the determinant of self is zero
        """
        det: float = self.determinant()

        if is_zero(det):
            raise Mat4Error('length of this Mat4 was zero')

        inv = 1.0 / det
        self.ax *= inv  
        self.ay *= inv 
        self.az *= inv  
        self.aw *= inv  
        self.bx *= inv  
        self.by *= inv  
        self.bz *= inv  
        self.bw *= inv  
        self.cx *= inv  
        self.cy *= inv  
        self.cz *= inv  
        self.cw *= inv  
        self.dx *= inv  
        self.dy *= inv  
        self.dz *= inv  
        self.dw *= inv  

    def unit(self) -> 'Mat4':
        """Return a copy of self with normalized length

        Returns
        ---
        Mat4
            
        Raises
        ---
        Mat4Error
            if the determinant of self is zero
        """
        det: float = self.determinant()
        if is_zero(det):
            raise Mat4Error('length of this Mat4 was zero')
        return self.scale(1.0 / det)

    def row0(self) -> tuple[float, float, float, float]:
        """Return the first row of float values

        Returns
        ---
        tuple[float, float, float, float]
        """
        return (self.ax, self.ay, self.az, self.aw)

    def row1(self) -> tuple[float, float, float, float]:
        """Return the second row of float values
        
        Returns
        ---
        tuple[float, float, float, float]
        """
        return (self.bx, self.by, self.bz, self.bw)

    def row2(self) -> tuple[float, float, float, float]:
        """Return the third row of float values

        Returns
        ---
        tuple[float, float, float, float]
        """
        return (self.cx, self.cy, self.cz, self.cw)

    def row3(self) -> tuple[float, float, float, float]:
        """Return the fourth row of float values

        Returns
        ---
        tuple[float, float, float, float]
        """
        return (self.dx, self.dy, self.dz, self.dw)

    def col0(self) -> tuple[float, float, float, float]:
        """Return the first column of float values

        Returns
        ---
        tuple[float, float, float, float]
        """
        return (self.ax, self.bx, self.cx, self.dx)

    def col1(self) -> tuple[float, float, float, float]:
        """Return the second column of float values

        Returns
        ---
        tuple[float, float, float, float]
        """
        return (self.ay, self.by, self.cy, self.dy)

    def col2(self) -> tuple[float, float, float, float]:
        """Return the third column of float values

        Returns
        ---
        tuple[float, float, float, float]
        """
        return (self.az, self.bz, self.cz, self.dz)

    def col3(self) -> tuple[float, float, float, float]:
        """Return the fourth column of float values

        Returns
        ---
        tuple[float, float, float, float]
        """
        return (self.aw, self.bw, self.cw, self.dw)

    def at(self, row: int, col: int) -> float:
        return self[col * 4 + row]

    def determinant(self) -> float:
        """Return the determinant of self

        Returns
        ---
        float
        """
        a00: float = self.ax
        a01: float = self.ay
        a02: float = self.az
        a03: float = self.aw
        a10: float = self.bx
        a11: float = self.by
        a12: float = self.bz
        a13: float = self.bw
        a20: float = self.cx
        a21: float = self.cy
        a22: float = self.cz
        a23: float = self.cw
        a30: float = self.dx
        a31: float = self.dy
        a32: float = self.dz
        a33: float = self.dw

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
            b50 - b51 - b52 + b53
        )

    def array(self) -> list[float]:
        """Return self as a list of floats

        Returns
        ---
        list[float]
        """
        return [
            self.ax, self.ay, self.az, self.aw,
            self.bx, self.by, self.bz, self.bw,
            self.cx, self.cy, self.cz, self.cw,
            self.dx, self.dy, self.dz, self.dw
        ]

    def multi_array(self) -> list[list[float]]:
        """Return self as a list of list floats

        Returns
        ---
        list[list[float]]
        """
        return [
            [self.ax, self.ay, self.az, self.aw],
            [self.bx, self.by, self.bz, self.bw],
            [self.cx, self.cy, self.cz, self.cw],
            [self.dx, self.dy, self.dz, self.dw]
        ]


# --- PLAIN


class PlainError(Exception):
    '''Custom error for plain'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Plain:
    normal: Vec3 = Vec3()
    direction: float = 0.0

    def __post_init__(self):
        if not self.normal.is_unit():
            self.normal.to_unit()

    @staticmethod
    def from_verts(p1: Vec3, p2: Vec3, p3: Vec3) -> 'Plain':
        """Create from three points

        Returns
        ---
        Plain
        """
        a: Vec3 = p2 - p1
        b: Vec3 = p3 - p1
        n: Vec3 = a.cross(b)
        n.to_unit()
        d: float = n.x * p1.x + n.y * p1.y + n.z * p1.z
        return Plain(n, -d)

    def copy(self) -> 'Plain':
        """Return a copy of self

        Returns
        ---
        Plain
        """
        return Plain(
            Vec3(
                self.normal.x,
                self.normal.y,
                self.normal.z,
            ),
            self.dir
        )

    def dot(self, v4: Vec4) -> float:
        x: float = self.normal.x * v4.x
        y: float = self.normal.y * v4.y
        z: float = self.normal.z * v4.z
        w: float = self.direction * v4.w
        return x + y + z + w

    def dot_coord(self, v3: Vec3) -> float:
        x: float = self.normal.x * v3.x
        y: float = self.normal.y * v3.y
        z: float = self.normal.z * v3.z
        w: float = self.direction
        return x + y + z + w

    def dot_normal(self, v3: Vec3) -> float:
        x: float = self.normal.x * v3.x
        y: float = self.normal.y * v3.y
        z: float = self.normal.z * v3.z
        return x + y + z

    def unit(self) -> 'Plain':
        """Return a copy of self that has been normalized

        Returns
        ---
        Plain
        """
        ls: float = self.normal.length_sqr()

        if is_one(ls):
            return self.copy()

        inv: float = inv_sqrt(ls)
        return Plain(
            self.normal * inv,
            self.direction * inv
        )

    def to_unit(self) -> None:
        """Normalize the length of self
        """
        ls = sqr(self.normal.x) + sqr(self.normal.y) + sqr(self.normal.z)
        if is_one(ls):
            raise PlainError('length of self was zero')
        inv = inv_sqrt(ls)

        self.normal.x *= inv
        self.normal.y *= inv
        self.normal.z *= inv
        self.direction *= inv

    def is_equil(self, other: 'Plain') -> bool:
        """Check if self and other have the same component values

        Parameters
        ---
        other : Plain

        Returns
        ---
        bool
        """
        check_n: bool = self.normal.is_equil(other.normal)
        check_d: bool = is_equil(self.direction, other.direction)
        return check_n and check_d


# --- QUATERNION


class QuatError(Exception):
    '''Custom error for quaternion'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0

    def __getitem__(self, idx):
        match clamp(idx, 0, 3):
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

    @staticmethod
    def from_euler(angles: Vec3) -> 'Quaternion':
        """Create from euler angles

        Returns
        ---
        Quaternion
        """
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

    @staticmethod
    def from_axis(angle_deg: float, axis: Vec3) -> 'Quaternion':
        """Create a quaternion from an angle and axis of rotation

        Returns
        ---
        Quaternion

        """
        cpy: Vec3 = axis.copy()

        if not cpy.is_unit():
            cpy.to_unit()

        angle_rad: float = to_radians(angle_deg * 0.5)
        c: float = cos(angle_rad)
        s: float = sin(angle_rad)

        x: float = cpy.x * s
        y: float = cpy.y * s
        z: float = cpy.z * s
        w: float = c

        return Quaternion(x, y, z, w)

    @staticmethod
    def lerp(start: 'Quaternion', end: 'Quaternion', weight: float) -> 'Quaternion':
        """Create a quaternion based on the *linear interpolation* between start and end based on weight

        Returns
        ---
        Quaternion
        """
        t: float = weight
        t1: float = 1.0 - t

        dot = start.dot(end)

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

        return Quaternion(x, y, z, w).unit()

    @staticmethod
    def slerp(start: 'Quaternion', end: 'Quaternion', weight: float) -> 'Quaternion':
        '''Return a slerp quaternion'''

        dot: float = start.dot(end)
        flip: bool = False

        if dot < 0.0:
            flip = True
            dot *= -1.0

        s1, s2 = 0.0, 0.0
        if dot > (1.0 - EPSILON):
            s1 = 1.0 - weight
            s2 = -weight if flip else weight
        else:
            o: float = arccos(dot)
            inv: float = 1.0 / sin(o)
            s1 = sin((1.0 - weight) * o) * inv
            s2 = -sin(weight * o) * inv if flip else sin(weight * o) * inv

        x: float = s1 * start.x + s2 * end.x
        y: float = s1 * start.y + s2 * end.y
        z: float = s1 * start.z + s2 * end.z
        w: float = s1 * start.w + s2 * end.w

        return Quaternion(x, y, z, w)

    def copy(self) -> 'Quaternion':
        """Return a copy of self

        Returns
        ---
        Quaternion
        """
        return Quaternion(self.x, self.y, self.z, self.z)

    def conjugate(self) -> 'Quaternion':
        """Return the conjugate of self

        Returns
        ---
        Quaternion
        """
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def scale(self, by: float) -> 'Quaternion':
        """Return a scaled copy of self based of *by* float value

        Returns
        ---
        Quaternion
        """
        x = self.x * by
        y = self.y * by
        z = self.z * by
        w = self.w * by
        return Quaternion(x, y, z, w)

    def inverse(self) -> 'Quaternion':
        """Return the invserse of self

        Returns
        ---
        Quaternion
        """
        inv: float = 1.0 / self.length_sqr()
        return Quaternion(-self.x * inv, -self.y * inv, -self.z * inv, self.q * inv)

    def to_unit(self) -> None:
        """Normalize the length of self

        Raises
        ---
        QuatError
            length of self was zero
        """
        lsq: float = self.length_sqr()

        if is_zero(lsq):
            raise QuatError('length was zero')

        inv = inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv
        self.z *= inv
        self.w *= inv

    def unit(self) -> 'Quaternion':
        """Return a copy of self with normalized length

        Returns
        ---
        Quaternion  
            
        Raises
        ---
        QuatError
            length of self was zero
        """
        lsq: float = self.length_sqr()

        if is_zero(lsq):
            raise QuatError('length was zero')

        return self.scale(inv_sqrt(lsq))

    def to_euler(self) -> Vec3:
        """Return self as euler values

        Returns
        ---
        Vec3
        """
        threshold: float = 0.4999995

        w2: float = sqr(self.w)
        x2: float = sqr(self.x)
        y2: float = sqr(self.y)
        z2: float = sqr(self.z)

        len_sq: float = x2 + y2 + z2 + w2
        test: float = (self.x * self.z) + (self.w * self.y)

        if test > threshold * len_sq:
            return Vec3(0.0, PIOVER2, 2.0 * arctan2(self.x, self.y))

        if test < -threshold * len_sq:
            return Vec3(0.0, -PIOVER2, -2.0 * arctan2(self.x, self.w))

        xy: float = 2.0 * ((self.w * self.x) - (self.y * self.z))
        xx: float = w2 - x2 - y2 + z2
        zy: float = 2.0 * ((self.w * self.z) - (self.z * self.y))
        zx: float = w2 + x2 - y2 - z2

        x: float = arctan2(xy, xx)
        y: float = arcsin(2.0 * test / len_sq)
        z: float = arctan2(zy, zx)

        return Vec3(x, y, z)

    def to_axis(self) -> Vec3:
        """Return self as axis angles

        Returns
        ---
        Vec3
        """
        qt_cpy = self.copy()

        if qt_cpy.w > 1.0:
            qt_cpy.to_unit()

        scl = sqrt(1.0 - sqr(qt_cpy.w))

        if is_zero(scl):
            return Vec3(qt_cpy.x, qt_cpy.y, qt_cpy.z)

        d = 1.0 / scl
        x = self.x * d
        y = self.y * d
        z = self.z * d

        return Vec3(x, y, z)

    def to_mat4(self) -> Mat4:
        """Create a mat4 matrix based on quaternion's current values

        Returns
        ---
        Mat4
        """
        x2: float = sqr(self.x)
        y2: float = sqr(self.y)
        z2: float = sqr(self.z)
        w2: float = sqr(self.w)

        xy: float = self.x * self.y
        xz: float = self.x * self.z
        xw: float = self.x * self.w

        yz: float = self.y * self.z
        yw: float = self.y * self.w

        zw: float = self.z * self.w

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

    def length_sqr(self) -> float:
        """Return the squared length

        Returns
        ---
        float
        """
        return sqr(self.x) + sqr(self.y) + sqr(self.z) + sqr(self.w)

    def length_sqrt(self) -> float:
        """Return the square root length

        Returns
        ---
        float
        """
        return sqrt(self.length_sqr())

    def dot(self, other: 'Quaternion') -> float:
        """Return the dot product between self and other

        Parameters
        ---
        other : Quaternion

        Returns
        ---
        float
        """
        return (
            (self.x * other.x) +
            (self.y * other.y) +
            (self.z * other.z) +
            (self.w * other.w)
        )

    def angle(self) -> float:
        """Return the current angle of self in radians

        Returns
        ---
        float
        """
        return 2.0 * arccos(self.w)

    def is_equil(self, other: 'Quaternion') -> bool:
        """Check if the components of self are the same as other's components

        Parameters
        ---
        other : Quaternion

        Returns
        ---
        bool
        """
        return (
            is_equil(self.x, other.x) and
            is_equil(self.y, other.y) and
            is_equil(self.z, other.z) and
            is_equil(self.w, other.w)
        )

    def is_unit(self) -> bool:
        """Check if the length of self is one

        Returns
        ---
        bool
        """
        return is_one(self.length_sqr())