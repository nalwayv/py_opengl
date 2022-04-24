"""Maths
"""
import math
from dataclasses import dataclass
from typing import Final, NamedTuple

# --- CONSTANTS


PI: Final[float]= 3.14159265358979323846
PHI: Final[float]= 1.57079632679489661923
TAU: Final[float]= 6.28318530717958647693
EPSILON: Final[float]= 1e-6
MAX_FLOAT: Final[float]= 1.7976931348623157e+308
MIN_FLOAT: Final[float]= 2.2250738585072014e-308
E: Final[float]= 2.71828182845904523536
INFINITY: Final[float]= math.inf
NEGATIVE_INFINITY: Final[float]= -math.inf


# ---


class Point2D(NamedTuple):
    x: float
    y: float


class Point3D(NamedTuple):
    x: float
    y: float
    z: float


class Point4D(NamedTuple):
    x: float
    y: float
    z: float
    w: float


# --- FUNCTIONS


def absf(x: float) -> float:
    """Return the absolute of value x

    Parameters
    ---
    x : float

    Returns
    ---
    float
    """
    return float(abs(x))


def absi(x: int) -> int:
    """Return the absolute of value x

    Parameters
    ---
    x : int

    Returns
    ---
    int
    """
    return int(abs(x))


def is_zero(val: float) -> bool:
    """Check if values is zero

    Parameters
    ---
    val : float

    Returns
    ---
    bool
        is zero or not
    """
    return absf(val) <= EPSILON


def is_one(val: float) -> bool:
    """Check if value is one

    Parameters
    ---
    val : float

    Returns
    ---
    bool
    """
    return is_zero(val - 1.0)


def is_equil(x: float, y: float) -> bool:
    """Check if two float values equil

    Parameters
    ---
    x : float

    y : float

    Returns
    ---
    bool
    """
    return absf(x - y) <= EPSILON


def is_infinite(val) -> bool:
    """Check if value if infinite

    Returns
    ---
    bool
    """
    return math.isinf(val)


def to_radians(degrees: float) -> float:
    """Convert degreese to radians

    Parameters
    ---
    degrees : float

    Returns
    ---
    float
    """
    return degrees * 0.01745329251994329577


def to_degreese(radians: float) -> float:
    """Convert radians to degrees

    Parameters
    ---
    radians : float

    Returns
    ---
    float
    """
    return radians * 57.2957795130823208768


def sqr(val: float) -> float:
    """Return the value squared

    Parameters
    ---
    val : float

    Returns
    ---
    float
    """     
    return val * val


def sqrt(val: float) -> float:
    """Return the sqrt

    Parameters
    ---
    val : float

    Returns
    ---
    float
    """
    return math.sqrt(val)


def inv_sqrt(val: float) -> float:
    """Return the inverse sqrt

    Parameters
    ---
    val : float

    Returns
    ---
    float
    """
    return 1.0 / sqrt(val)


def maxf(x: float, y: float) -> float:
    """Return max float between x and y

    Parameters
    ---
    x : float
        
    y : float

    Returns
    ---
    float
    """
    return x if x > y else y


def maxi(x: int, y: int) -> int:
    """Return max int between x and y

    Parameters
    ---
    x : int
        
    y : int
        
    Returns
    ---
    int
    """
    return x if x > y else y


def minf(x: float, y: float) -> float:
    """Return min float between x and y

    Parameters
    ---
    x : float
        
    y : float
        
    Returns
    ---
    float
    """
    return x if x < y else y


def mini(x: int, y: int) -> int:
    """Return min int between x and y

    Parameters
    ---
    x : int
        
    y : int
        

    Returns
    ---
    int
    """
    return x if x < y else y


def clampf(val: float, low: float, high: float) -> float:
    """Clamp float value between low and high

    Returns
    ---
    float
    """
    if val <= low:
        return low

    if val >= high:
        return high

    return val


def clampi(val: int, low: int, high: int) -> int:
    """Clamp int value between low and high

    Returns
    ---
    int
    """
    if val <= low:
        return low

    if val >= high:
        return high

    return val


def lerp(start: float, to: float, weight: float) -> float:
    """Return amount of *linear interpolation* between start and yo based on weight

    Parameters
    ---
    start : float

    to : float
        
    weight : float

    Returns
    ---
    float
    """
    return start + weight * (to - start)


def cubic_lerp(start: float, start_tan: float, to: float, to_tan: float, weight) -> float:
    """Return cubic lerp

    Parameters
    ---
    start : float

    start_tan : float

    to : float

    to_tan : float

    Returns
    ---
    float
    """
    a: float= start
    b: float= start_tan
    c: float= to
    d: float= to_tan
    t: float= weight
    p: float= (d - c) - (a - b)

    squared: float= t * t
    cubed: float= squared * t

    return cubed * p + squared * ((a - b) - p) + t * (c - a) + b


def normalize(val: float, low: float, high: float) -> float:
    """Normalize value between low and high

    Parameters
    ---
    val : float

    low : float

    high : float

    Returns
    ---
    float
    """
    return (val - low) / (high - low)


def tan(val: float) -> float:
    """Return *trigonometry tan* value

    Parameters
    ---
    val : float

    Returns
    ---
    float
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
    """
    return math.cos(val)


def arccos(val: float) -> float:
    """Return the arc cosine value

    value has to be between -1 and 1 else will raise an error
    
    the value in radians between 0 and PI\n
    -1 = PI\n
    1 = 0

    Parameters
    ---
    val : float

    Returns
    ---
    float
    """
    return math.acos(val)


def arcsin(val: float) -> float:
    """Return the arc aign value

    value has to be between -1 and 1 else will raise an error

    the value in radians between 0 and PI\n
    -1 = -PI/2\n
    1 = PI/2

    Parameters
    ---
    val : float

    Returns
    ---
    float
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
    """
    return math.atan2(y, x)


def stepify(val: float, steps: float) -> float:
    """Snaps value to given steps

    Parameters
    ---
    value : float

    steps : float

    Returns
    ---
    float
    """
    return (val // steps) * steps


def wrap(val: int, low: int, high: int) -> int:
    """wrap int value between low and high - 1

    Parameters
    ---
    value : int
    
    low : int
    
    high : int

    Example
    ---
    wrap(6, 1, 5) => 2

    wrap(5, 1, 6) => 1
    
    wrap(7, 2, 5) => 4

    Returns
    ---
    int
    """
    return ((val - low) % (high - low) + low)


def repeat(val: int, length: int) -> int:
    """repeat value from 0 to high
    
    Parameters    
    ---
    value : int

    length : int
    
    Return
    ---
    int
    """
    return val - (val // length) * val


def ping_pong(val: int, length:int) -> int:
    """repeat value from 0 to high and back
    
    Parameters    
    ---
    value : int

    length : int
    
    Return
    ---
    int
    """
    val= repeat(val, length * 2)
    val= length - absi(val - length)
    return val


def catmullrom(pos_a: float, pos_b: float, pos_c: float, pos_d: float, amount: float) -> float:
    """catmull_rom interpolation using the specified positions

    Parameters
    ---
    pos_a : float

    pos_b : float

    pos_c : float

    pos_d : float

    amount : float

    Returns
    ---
    float
    """
    squared: float= amount * amount
    cubed: float= squared * amount
    return (
        0.5 * (2.0 * pos_b + (pos_c - pos_a) * amount +
        (2.0 * pos_a - 5.0 * pos_b + 4.0 * pos_c - pos_d) * squared +
        (3.0 * pos_b - pos_a - 3.0 * pos_c + pos_d) * cubed)
    )  


def hetmit(value_a: float, tangent_a: float, value_b: float, tangent_b: float, amount: float) -> float:
    """hermite spline interpolation

    Parameters
    ---
    value_a : float

    tangent_a : float

    value_b : float

    tangent_b : float

    amount : float

    Returns
    ---
    float
    """
    if amount == 0.0:
        return value_a

    if amount == 1.0:
        return value_b

    squared: float= amount * amount
    cubed: float= squared * amount
    return (
        (2.0 * value_a - 2.0 * value_b + tangent_b + tangent_a) * cubed +
        (3.0 * value_b - 3.0 * value_a - 2.0 * tangent_a - tangent_b) * squared +
        tangent_a * amount + value_a
    )



# --- VECTOR_2 (X, Y)



class Vec2Error(Exception):
    """Custom error for Vec2"""

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Vec2:
    x: float= 0.0
    y: float= 0.0

    def __getitem__(self, idx):
        match clampi(idx, 0, 1):
            case 0: return self.x
            case 1: return self.y
        raise Vec2Error('out of range')

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

    @staticmethod
    def one() -> 'Vec2':
        return Vec2(1.0, 1.0)

    @staticmethod
    def zero() -> 'Vec2':
        return Vec2(0.0, 0.0)

    @staticmethod
    def unit_x() -> 'Vec2':
        return Vec2(x=1.0)

    @staticmethod
    def unit_y() -> 'Vec2':
        return Vec2(y=1.0)

    def to_str(self) -> str:
        """Return string format

        Returns
        ---
        str
        """
        return f"X: {self.x}, Y: {self.y}"

    def lerp(self, to: 'Vec2', weight: float) -> 'Vec2':
        """Return lerp between begin end end vec3

        Parameters
        ---
        begin : Vec2

        end : Vec2
            
        weight : float

        Returns
        ---
        Vec2
        """
        x: float= lerp(self.x, to.x, weight)
        y: float= lerp(self.y, to.y, weight)

        return Vec2(x, y)
    
    def to_unit(self) -> None:
        """Normalize length

        Raises
        ---
        Vec2Error
            if current length is zero
        """
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            raise Vec2Error('length of this vec2 was zero')

        inv: float= inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv

    def unit(self) -> 'Vec2':
        """Return a copy of self with a normal length

        Returns
        ---
        Vec2
            a copy of self with a normalized length
        """
        lsq: float= self.length_sqr()
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

    def perpendicular(self) -> 'Vec2':
        """Return the prpendicular of self

        Returns
        ---
        Vec2
        """
        return Vec2(-self.y, self.x)

    def rotate_by(self, angle_deg: float) -> 'Vec2':
        """Rotate by angle

        Returns
        ---
        Vec2
            vec2 rotated by
        """
        angle_rad: float= to_radians(angle_deg)
        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        x: float= self.x * c - self.y * s
        y: float= self.x * s + self.y * c

        return Vec2(x, y)

    def angle_to(self, other: 'Vec2') -> float:
        """Return the nagle in radians to other

        Parameters
        ---
        other : Vec2

        Returns
        ---
        float
            angle to in radians
        """
        ang: float= arctan2(other.y, self.x) - arctan2(self.y, other.x)
        
        if ang > PI:
            ang -= TAU
        elif ang <= -PI:
            ang += TAU

        return ang

    def angle_between(self, other: 'Vec2') -> float:
        """Return the angle in radians between self and other

        Parameters
        ---
        other : Vec2

        Returns
        ---
        float
            angle between self and other
        """
        return arccos(
            self.dot(other) / (self.length_sqrt() * other.length_sqrt())
        )
    
    def sum_total(self) -> float:
        """Return sum total of all values

        Returns
        ---
        float
        """
        return self.x + self.y

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
        return sqr(self.x) + sqr(self.y)

    def length_sqrt(self) -> float:
        """Return the square root length

        Returns
        ---
        float
        """
        return sqrt(self.length_sqr())

    def distance_sqrt(self, other: 'Vec2') -> float:
        """Return the distance between self and other vec2

        Parameters
        ---
        other : Vec2

        Returns
        ---
        float
        """
        dir: Vec2= other - self
        return dir.length_sqrt()

    def distance_sqr(self, other: 'Vec2') -> float:
        """Return the distance between self and other vec2

        Parameters
        ---
        other : Vec2

        Returns
        ---
        float
        """
        dir: Vec2= other - self
        return dir.length_sqr()


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



# --- VECTOR_3 (X, Y, Z)



class Vec3Error(Exception):
    '''Custom error for Vec3'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq= False, repr= False, slots= True)
class Vec3:
    x: float= 0.0
    y: float= 0.0
    z: float= 0.0

    def __getitem__(self, idx):
        match clampi(idx, 0, 2):
            case 0: return self.x
            case 1: return self.y
            case 2: return self.z
        raise Vec3Error('out of range')

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

    @staticmethod
    def one() -> 'Vec3':
        return Vec3(1.0, 1.0, 1.0)

    @staticmethod
    def zero() -> 'Vec3':
        return Vec3(0.0, 0.0, 0.0)

    @staticmethod
    def unit_x() -> 'Vec3':
        return Vec3(x=1.0)

    @staticmethod
    def unit_y() -> 'Vec3':
        return Vec3(y=1.0)

    @staticmethod
    def unit_z() -> 'Vec3':
        return Vec3(z=1.0)

    @staticmethod
    def from_v2(v2: Vec2, z: float) -> 'Vec3':
        return Vec3(x= v2.x, y= v2.y, z= z)

    @staticmethod
    def from_max(a: 'Vec3', b: 'Vec3') -> 'Vec3':
        x: float= maxf(a.x, b.x)
        y: float= maxf(a.y, b.y)
        z: float= maxf(a.z, b.z)

        return Vec3(x, y, z)

    @staticmethod
    def from_min(a: 'Vec3', b: 'Vec3') -> 'Vec3':
        x: float= minf(a.x, b.x)
        y: float= minf(a.y, b.y)
        z: float= minf(a.z, b.z)

        return Vec3(x, y, z)

    def to_str(self) -> str:
        """Return string format

        Returns
        ---
        str
        """
        return f"X: {self.x}, Y: {self.y}, Z: {self.z}"

    def lerp(self, to: 'Vec3', weight: float) -> 'Vec3':
        """Return lerp between begin and end vec3

        Parameters
        ---
        begin : Vec3
        
        end : Vec3
            
        weight : float

        Returns
        ---
        Vec3
            
        """
        x: float= lerp(self.x, to.x, weight)
        y: float= lerp(self.y, to.y, weight)
        z: float= lerp(self.z, to.z, weight)
    
        return Vec3(x, y, z)

    def xy(self) -> Vec2:
        """Return *xy* components

        Returns
        ---
        Vec3
        """
        return Vec2(self.x, self.y)

    def to_unit(self) -> None:
        """Normalize length

        Raises
        ---
        Vec3Error
            if length is zero
        """
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            raise Vec3Error('length of this vec3 was zero')

        inv = inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv
        self.z *= inv

    def unit(self) -> 'Vec3':
        """Return a copy of this vec3 with a normal length

        Raises
        ---
        Vec3Error
            if length is zero

        Returns
        ---
        Vec3
        """
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            raise Vec3Error('length of this vec3 was zero')

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
            (self.x * other.y) - (self.y * other.x)
        )

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

    def rotate_by(self, angle_deg: float, axis: 'Vec3') -> 'Vec3':
        """Rotate by angle along unit axis

        Parameters
        ----------
        angle_deg : float

        axis : Vec3

        Returns
        -------
        Vec3
            rotated vec3
        """
        rad: float = to_radians(angle_deg)

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

        x: float= self.dot(m1)
        y: float= self.dot(m2)
        z: float= self.dot(m3)

        return Vec3(x, y, z)

    def sum_total(self) -> float:
        """Return sum total of all values

        Returns
        ---
        float
        """
        return self.x + self.y + self.z

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

    def distance_sqrt(self, other: 'Vec3') -> float:
        """Return the distance between self and other vec3

        Parameters
        ---
        other : Vec3

        Returns
        ---
        float
        """
        dir: Vec3= other - self
        return dir.length_sqrt()

    def distance_sqr(self, other: 'Vec3') -> float:
        """Return the distance between self and other vec3

        Parameters
        ---
        other : Vec3

        Returns
        ---
        float
        """
        dir: Vec3= other - self
        return dir.length_sqr()

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
        return (
            is_equil(self.x, other.x) and
            is_equil(self.y, other.y) and
            is_equil(self.z, other.z)
        )



# --- VECTOR_4 (X, Y, Z, W)



class Vec4Error(Exception):
    '''Custom error for Vec4'''

    def __init__(self, msg: str):
        super().__init__(msg)



@dataclass(eq= False, repr= False, slots= True)
class Vec4:
    x: float= 0.0
    y: float= 0.0
    z: float= 0.0
    w: float= 0.0

    def __getitem__(self, idx):
        match clampi(idx, 0, 3):
            case 0: return self.x
            case 1: return self.y
            case 2: return self.z
            case 3: return self.w

        raise Vec4Error('out of range')

    def __add__(self, other):
        if not isinstance(other, Vec4):
            raise Vec4Error('not of type Vec4')
    
        return Vec4(
            x= self.x + other.x,
            y= self.y + other.y,
            z= self.z + other.z,
            w= self.w + other.w
        )

    def __sub__(self, other):
        if not isinstance(other, Vec4):
            raise Vec4Error('not of type Vec4')

        return Vec4(
            x= self.x - other.x,
            y= self.y - other.y,
            z= self.z - other.z,
            w= self.w - other.w
        )

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise Vec4Error('other was not of type float or int')

        return Vec4(
            x= self.x * other,
            y= self.y * other,
            z= self.z * other,
            w= self.w * other
        )
        
    @staticmethod
    def one() -> 'Vec4':
        return Vec4(1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def zero() -> 'Vec4':
        return Vec4(0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def unit_x() -> 'Vec4':
        return Vec4(x= 1.0)

    @staticmethod
    def unit_y() -> 'Vec4':
        return Vec4(y= 1.0)

    @staticmethod
    def unit_z() -> 'Vec4':
        return Vec4(z= 1.0)

    @staticmethod
    def unit_w() -> 'Vec4':
        return Vec4(w= 1.0)

    @staticmethod
    def from_v2(xy: Vec2, z: float, w: float) -> 'Vec4':
        return Vec4(
            x= xy.x,
            y= xy.y,
            z= z,
            w= w
        )

    @staticmethod
    def from_v3(xyz: Vec3, w: float) -> 'Vec4':
        return Vec4(
            x= xyz.x,
            y= xyz.y,
            z= xyz.z,
            w= w
        )

    def to_str(self) -> str:
        """Return string format

        Returns
        ---
        str
        """
        return f"X: {self.x}, Y: {self.y} Z: {self.z}, W: {self.w}"

    def xy(self) -> Vec2:
        """Return *xy* components

        Returns
        ---
        Vec3
        """
        return Vec2(self.x, self.y)

    def xyz(self) -> Vec3:
        """Return *xyz* components

        Returns
        ---
        Vec3
        """
        return Vec3(self.x, self.y, self.z)

    def lerp(self, to: 'Vec4', weight: float) -> 'Vec4':
        """Return lerp between begin ben and end vec4

        Parameters
        ---
        begin : Vec4
            
        end : Vec4
            
        weight : float
            
        Returns
        ---
        Vec4
        """
        x: float= lerp(self.x, to.x, weight)
        y: float= lerp(self.y, to.y, weight)
        z: float= lerp(self.z, to.z, weight)
        w: float= lerp(self.w, to.w, weight)

        return Vec4(x, y, z, w)

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

        inv: float = inv_sqrt(lsq)
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
            raise Vec4Error('length of this vec4 was zero')

        return self * inv_sqrt(lsq)

    def sum_total(self) -> float:
        """Return sum total of all values

        Returns
        ---
        float
        """
        return self.x + self.y + self.z + self.w

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

    def distance_sqrt(self, other: 'Vec4') -> float:
        """Return the distance between self and other vec4

        Parameters
        ---
        other : Vec4

        Returns
        ---
        float
        """
        dir: Vec4= other - self
        return dir.length_sqrt()

    def distance_sqr(self, other: 'Vec4') -> float:
        """Return the distance between self and other vec4

        Parameters
        ---
        other : Vec4

        Returns
        ---
        float
        """
        dir: Vec4= other - self
        return dir.length_sqr()

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

    def is_zero(self) -> bool:
        """Check if the current *x, y, z* components of self are zero in value

        Returns
        ---
        bool
        """
        return (
            is_zero(self.x) and
            is_zero(self.y) and
            is_zero(self.z) and
            is_zero(self.w)
        )


# --- MATRIX_3


class Mat3Error(Exception):
    '''Custom error for matrix 3x3'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq= False, repr= False, slots= True)
class Mat3:
    ax: float= 0.0
    ay: float= 0.0
    az: float= 0.0
    bx: float= 0.0
    by: float= 0.0
    bz: float= 0.0
    cx: float= 0.0
    cy: float= 0.0
    cz: float= 0.0

    def __getitem__(self, idx):
        match clampi(idx, 0, 8):
            case 0: 
                return self.ax
            case 1: 
                return self.ay
            case 2: 
                return self.az
            case 3: 
                return self.bx
            case 4: 
                return self.by
            case 5: 
                return self.bz
            case 6: 
                return self.cx
            case 7: 
                return self.cy
            case 8: 
                return self.cz

        raise Mat3Error('out of range')

    def __add__(self, other):
        if not isinstance(other, Mat3):
            raise Mat3Error('not of type mat3')

        ax: float= self.ax + other.ax
        ay: float= self.ay + other.ay
        az: float= self.az + other.az
        bx: float= self.bx + other.bx
        by: float= self.by + other.by
        bz: float= self.bz + other.bz
        cx: float= self.cx + other.cx
        cy: float= self.cy + other.cy
        cz: float= self.cz + other.cz

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz
        )

    def __sub__(self, other):
        if not isinstance(other, Mat3):
            raise Mat3Error('not of type mat3')

        ax: float= self.ax - other.ax
        ay: float= self.ay - other.ay
        az: float= self.az - other.az
        bx: float= self.bx - other.bx
        by: float= self.by - other.by
        bz: float= self.bz - other.bz
        cx: float= self.cx - other.cx
        cy: float= self.cy - other.cy
        cz: float= self.cz - other.cz

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz
        )

    def __mul__(self, other):
        if not isinstance(other, Mat3):
            raise Mat3Error('not of type Mat3')

        ax: float= self.ax * other.ax + self.ay * other.bx + self.az * other.cx
        ay: float= self.ax * other.ay + self.ay * other.by + self.az * other.cy
        az: float= self.ax * other.az + self.ay * other.bz + self.az * other.cz
        bx: float= self.bx * other.ax + self.by * other.bx + self.bz * other.cx
        by: float= self.bx * other.ay + self.by * other.by + self.bz * other.cy
        bz: float= self.bx * other.az + self.by * other.bz + self.bz * other.cz
        cx: float= self.cx * other.ax + self.cy * other.bx + self.cz * other.cx
        cy: float= self.cx * other.ay + self.cy * other.by + self.cz * other.cy
        cz: float= self.cx * other.az + self.cy * other.bz + self.cz * other.cz
        
        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz
        )

    @staticmethod
    def identity() -> 'Mat3':
        """Create an identity mat3 matrix

        Returns
        ---
        Mat3
        """
        return Mat3(ax= 1.0, by= 1.0, cz= 1.0)

    @staticmethod
    def create_scaler(v3: Vec3) -> 'Mat3':
        """Create a scaler mat3 matrix

        Returns
        ---
        Mat3
        """
        return Mat3(ax= v3.x, by= v3.y, cz= v3.z)
    
    @staticmethod
    def create_x_rotation(angle_deg: float) -> 'Mat3':
        """Create a rotation *X* mat3 matrix

        Returns
        ---
        Mat3
        """
        angle_rad: float= to_radians(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat3(ax= 1.0, by= c, bz= -s, cy= s, cz= c)

    @staticmethod
    def create_y_rotation(angle_deg: float) -> 'Mat3':
        """Create a rotation *y* mat3 matrix

        Returns
        ---
        Mat3
        """
        angle_rad: float= to_radians(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat3(ax= c, az= s, by= 1.0, cx= -s, cz= c)

    @staticmethod
    def create_z_rotation(angle_deg: float) -> 'Mat3':
        """Create a rotation *z* mat3 matrix

        Returns
        ---
        Mat3
        """
        angle_rad: float= to_radians(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat3(ax= c, ay= -s, bx= s, by= c, cz= 1.0)

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
        ax: float= self.ax * by
        ay: float= self.ay * by
        az: float= self.az * by
        bx: float= self.bx * by
        by: float= self.by * by
        bz: float= self.bz * by
        cx: float= self.cx * by
        cy: float= self.cy * by
        cz: float= self.cz * by

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz
        )

    def transpose(self) -> 'Mat3':
        """Return a transposed copy of self

        Returns
        ---
        Mat3
        """
        ax: float= self.ax
        ay: float= self.bx
        az: float= self.cx
        bx: float= self.ay
        by: float= self.by
        bz: float= self.cy
        cx: float= self.az
        cy: float= self.bz
        cz: float= self.cz

        return Mat3(
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz
        )

    def cofactor(self) -> 'Mat3':
        """Return a cofactor copy of self

        Returns
        ---
        Mat3
        """
        ax: float= self.ax
        ay: float= -self.ay
        az: float= self.az
        bx: float= -self.bx
        by: float= self.by
        bz: float= -self.bz
        cx: float= self.cx
        cy: float= -self.cy
        cz: float= self.cz

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
        det: float= self.determinant()

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
        det: float= self.determinant()

        if is_zero(det):
            raise Mat3Error('length of this Mat4 was zero')

        return self.scale(1.0 / det)


    def inverse(self) -> 'Mat3':
        """Return the inverse of self

        Returns
        ---
        Mat3
        """
        a: float= self.ax
        b: float= self.ay
        c: float= self.az
        d: float= self.bx
        e: float= self.by
        f: float= self.bz
        g: float= self.cx
        h: float= self.cy
        i: float= self.cz

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
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz
        )

    def row0(self) -> Vec3:
        """Return the first row of float values

        Returns
        ---
        Vec3
        """
        return Vec3(self.ax, self.ay, self.az)

    def row1(self) -> Vec3:
        """Return the second row of float values
        
        Returns
        ---
        Vec3
        """
        return Vec3(self.bx, self.by, self.bz)

    def row2(self) -> Vec3:
        """Return the third row of float values

        Returns
        ---
        Vec3
        """
        return Vec3(self.cx, self.cy, self.cz)

    def col0(self) -> Vec3:
        """Return the first column of float values

        Returns
        ---
        Vec3
        """
        return Vec3(self.ax, self.bx, self.cx)

    def col1(self) -> Vec3:
        """Return the second column of float values

        Returns
        ---
        Vec3
        """
        return Vec3(self.ay, self.by, self.cy)

    def col2(self) -> Vec3:
        """Return the third column of float values

        Returns
        ---
        Vec3
        """
        return Vec3(self.az, self.bz, self.cz)

    def at(self, row: int, col: int) -> float:
        return self[col * 3 + row]

    def determinant(self) -> float:
        """Return the determinant of self

        Returns
        ---
        float
        """
        a0: float= self.ax * self.by * self.cz
        a1: float= self.ay * self.bz * self.cx
        a2: float= self.az * self.bx * self.cy
        b1: float= self.az * self.by * self.cx
        b2: float= self.ax * self.bz * self.cy
        b3: float= self.ay * self.bx * self.cz

        return a0 + a1 + a2 - b1 - b2 - b3

    def trace(self) -> float:
        """Return the sum of the trace values

        Returns
        ---
        float
        """
        return self.ax + self.by + self.cz

    def array(self) -> list[float]:
        """

        Returns
        ---
        list[float]
        """
        return [
            self.ax, self.ay, self.az,
            self.bx, self.by, self.bz,
            self.cx, self.cy, self.cz
        ]

    def multi_array(self) -> list[list[float]]:
        """

        Returns
        ---
        list[list[float]]
        """
        return [
            [self.ax, self.ay, self.az],
            [self.bx, self.by, self.bz],
            [self.cx, self.cy, self.cz]
        ]
  


# --- MATRIX_4



class Mat4Error(Exception):
    '''Custom error for matrix 4x4'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq= False, repr= True, slots= True)
class Mat4:
    ax: float= 0.0
    ay: float= 0.0
    az: float= 0.0
    aw: float= 0.0
    bx: float= 0.0
    by: float= 0.0
    bz: float= 0.0
    bw: float= 0.0
    cx: float= 0.0
    cy: float= 0.0
    cz: float= 0.0
    cw: float= 0.0
    dx: float= 0.0
    dy: float= 0.0
    dz: float= 0.0
    dw: float= 0.0

    def __getitem__(self, idx):
        match clampi(idx, 0, 15):
            case 0: 
                return self.ax
            case 1: 
                return self.ay
            case 2: 
                return self.az
            case 3: 
                return self.aw
            case 4: 
                return self.bx
            case 5: 
                return self.by
            case 6: 
                return self.bz
            case 7: 
                return self.bw
            case 8: 
                return self.cx
            case 9: 
                return self.cy
            case 10: 
                return self.cz
            case 11: 
                return self.cw
            case 12: 
                return self.dx
            case 13: 
                return self.dy
            case 14: 
                return self.dz
            case 15: 
                return self.dw
        raise Mat4Error('out of range')

    def __add__(self, other):
        if not isinstance(other, Mat4):
            raise Mat4Error('not of type mat4')

        ax: float= self.ax + other.ax
        ay: float= self.ay + other.ay
        az: float= self.az + other.az
        aw: float= self.aw + other.aw
        bx: float= self.bx + other.bx
        by: float= self.by + other.by
        bz: float= self.bz + other.bz
        bw: float= self.bw + other.bw
        cx: float= self.cx + other.cx
        cy: float= self.cy + other.cy
        cz: float= self.cz + other.cz
        cw: float= self.cw + other.cw
        dx: float= self.dx + other.dx
        dy: float= self.dy + other.dy
        dz: float= self.dz + other.dz
        dw: float= self.dw + other.dw

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw
        )

    def __sub__(self, other):
        if not isinstance(other, Mat4):
            raise Mat4Error('not of type mat4')

        ax: float= self.ax - other.ax
        ay: float= self.ay - other.ay
        az: float= self.az - other.az
        aw: float= self.aw - other.aw
        bx: float= self.bx - other.bx
        by: float= self.by - other.by
        bz: float= self.bz - other.bz
        bw: float= self.bw - other.bw
        cx: float= self.cx - other.cx
        cy: float= self.cy - other.cy
        cz: float= self.cz - other.cz
        cw: float= self.cw - other.cw
        dx: float= self.dx - other.dx
        dy: float= self.dy - other.dy
        dz: float= self.dz - other.dz
        dw: float= self.dw - other.dw

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw
        )

    def __mul__(self, other):
        if not isinstance(other, Mat4):
            raise Mat4Error('not of type Mat4')
        ax: float= (
            (self.ax * other.ax) +
            (self.ay * other.bx) +
            (self.az * other.cx) +
            (self.aw * other.dx)
        )

        ay: float= (
            (self.ax * other.ay) +
            (self.ay * other.by) +
            (self.az * other.cy) +
            (self.aw * other.dy)
        )

        az: float= (
            (self.ax * other.az) +
            (self.ay * other.bz) +
            (self.az * other.cz) +
            (self.aw * other.dz)
        )

        aw: float= (
            (self.ax * other.aw) +
            (self.ay * other.bw) +
            (self.az * other.cw) +
            (self.aw * other.dw)
        )

        bx: float= (
            (self.bx * other.ax) +
            (self.by * other.bx) +
            (self.bz * other.cx) +
            (self.bw * other.dx)
        )

        by: float= (
            (self.bx * other.ay) +
            (self.by * other.by) +
            (self.bz * other.cy) +
            (self.bw * other.dy)
        )

        bz: float= (
            (self.bx * other.az) +
            (self.by * other.bz) +
            (self.bz * other.cz) +
            (self.bw * other.dz)
        )

        bw: float= (
            (self.bx * other.aw) +
            (self.by * other.bw) +
            (self.bz * other.cw) +
            (self.bw * other.dw)
        )

        cx: float= (
            (self.cx * other.ax) +
            (self.cy * other.bx) +
            (self.cz * other.cx) +
            (self.cw * other.dx)
        )

        cy: float= (
            (self.cx * other.ay) +
            (self.cy * other.by) +
            (self.cz * other.cy) +
            (self.cw * other.dy)
        )

        cz: float= (
            (self.cx * other.az) +
            (self.cy * other.bz) +
            (self.cz * other.cz) +
            (self.cw * other.dz)
        )

        cw: float= (
            (self.cx * other.aw) +
            (self.cy * other.bw) +
            (self.cz * other.cw) +
            (self.cw * other.dw)
        )

        dx: float= (
            (self.dx * other.ax) +
            (self.dy * other.bx) +
            (self.dz * other.cx) +
            (self.dw * other.dx)
        )

        dy: float= (
            (self.dx * other.ay) +
            (self.dy * other.by) +
            (self.dz * other.cy) +
            (self.dw * other.dy)
        )

        dz: float= (
            (self.dx * other.az) +
            (self.dy * other.bz) +
            (self.dz * other.cz) +
            (self.dw * other.dz)
        )

        dw: float= (
            (self.dx * other.aw) +
            (self.dy * other.bw) +
            (self.dz * other.cw) +
            (self.dw * other.dw)
        )

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
        return Mat4(ax= 1.0, by= 1.0, cz= 1.0, dw= 1.0)

    @staticmethod
    def create_scaler(v3: Vec3) -> 'Mat4':
        """Create a scaler mat4 matrix

        Returns
        ---
        Mat4
        """
        return Mat4(ax= v3.x, by= v3.y, cz= v3.z, dw= 1.0)

    @staticmethod
    def create_translation(v3: Vec3) -> 'Mat4':
        """Create a translation mat4 matrix

        Returns
        ---
        Mat4
        """
        return Mat4(
            ax= 1.0,
            by= 1.0,
            cz= 1.0,
            dx= v3.x,
            dy= v3.y,
            dz= v3.z,
            dw= 1.0
        )

    @staticmethod
    def create_x_shear(shear_y: float, shear_z: float) -> 'Mat4':
        """Create a shear mat4 matrix on x axis

        Returns
        ---
        Mat4
        """
        return Mat4(
            ax= 1.0,
            ay= shear_y,
            az= shear_z,
            by= 1.0,
            cz= 1.0,
            dw= 1.0
        )

    @staticmethod
    def create_y_shear(shear_x: float, shear_z: float) -> 'Mat4':
        """Create a shear mat4 matrix on y axis

        Returns
        ---
        Mat4
        """
        return Mat4(
            ax= 1.0,
            bx= shear_x, 
            by= 1.0, 
            bz= shear_z,
            cz= 1.0,
            dw= 1.0
        )

    @staticmethod
    def create_z_shear(shear_x: float, shear_y: float) -> 'Mat4':
        """Create a shear mat4 matrix on z axis

        Returns
        ---
        Mat4
        """
        return Mat4(
            ax= 1.0,
            by= 1.0,
            cx= shear_x,
            cy= shear_y, 
            cz= 1.0,
            dw= 1.0
        )

    @staticmethod
    def create_x_rotation(angle_deg: float) -> 'Mat4':
        """Create a rotation *X* mat4 matrix

        Returns
        ---
        Mat4
        """
        angle_rad: float= to_radians(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat4(ax= 1.0, by= c, bz= -s, cy= s, cz= c, dw= 1.0)

    @staticmethod
    def create_y_rotation(angle_deg: float) -> 'Mat4':
        """Create a rotation *y* mat4 matrix

        Returns
        ---
        Mat4
        """
        angle_rad: float= to_radians(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat4(ax= c, az= s, by= 1.0, cx= -s, cz= c, dw= 1.0)
      
    @staticmethod
    def create_z_rotation(angle_deg: float) -> 'Mat4':
        """Create a rotation *z* mat4 matrix

        Returns
        ---
        Mat4
        """
        angle_rad: float= to_radians(angle_deg)

        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Mat4(ax= c, ay= -s, bx= s, by= c, cz= 1.0, dw= 1.0)   
  
    @staticmethod
    def from_axis(angle_deg: float, unit_axis: Vec3) -> 'Mat4':
        """Create a rotated matrix

        Parameters
        ----------
        angle_deg : float

        axis : Vec3

        Returns
        -------
        Mat4
            rotated mat4
        """
        if not unit_axis.is_unit():
            unit_axis.to_unit()

        rad: float= to_radians(angle_deg)
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
        aw: float= 0.0
        bx: float= xy + sin_z
        by: float= yy + c
        bz: float= yz - sin_x
        bw: float= 0.0
        cx: float= xz - sin_y
        cy: float= yz + sin_x
        cz: float= zz + c
        cw: float= 0.0
        dx: float= 0.0
        dy: float= 0.0
        dz: float= 0.0
        dw: float= 1.0

        return Mat4 (
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw
        )


    @staticmethod
    def from_quat(qx: float, qy: float, qz: float, qw: float) -> 'Mat4':
        x2: float= sqr(qx)
        y2: float= sqr(qy)
        z2: float= sqr(qz)
        w2: float= sqr(qw)

        xy: float= qx * qy
        xz: float= qx * qz
        xw: float= qx * qw

        yz: float= qy * qz
        yw: float= qy * qw

        zw: float= qz * qw

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

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw
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

        x: Vec3= up.cross(z).unit()
        y: Vec3= z.cross(x).unit()

        dx: float= -x.dot(eye)
        dy: float= -y.dot(eye)
        dz: float= -z.dot(eye)

        return Mat4(
            x.x, y.x, z.x, 0.0,
            x.y, y.y, z.y, 0.0,
            x.z, y.z, z.z, 0.0,
            dx, dy, dz, 1.0
        )

    @staticmethod
    def frustum_projection(fov: float, aspect: float, near: float, far: float) -> 'Mat4':
        """Create a mat4 matrix *frustum projection* field of view

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

        if near <= 0.0 or far <= 0.0 or near >= far:
            raise Mat4Error('m4 projection aspect out of range')

        y: float= 1.0 / tan(fov * 0.5)
        x: float= y / aspect
        r: float= far / (far - near)
        n: float= near * r

        return Mat4(ax= x, by= y, cz= r, cw= -1.0, dz= n)

    @staticmethod
    def ortho_projection(left: float, right: float, top: float, bottom: float, near: float, far: float) -> 'Mat4':
        """Create a mat4 matrix *ortho_projection* field of view

        Returns
        ---
        Mat4
        """
        rl_inv: float= 1.0 / (right - left)
        tb_inv: float= 1.0 / (top - bottom)
        fn_inv: float= 1.0 / (far - near)

        rl: float= right + left
        tb: float= top + bottom
        fn: float= far + near

        return Mat4(
            ax= 2.0 * rl_inv,
            by= 2.0 * tb_inv,
            cz= -2.0 * fn_inv,
            dx= -rl * rl_inv,
            dy= -tb * tb_inv,
            dz= -fn * fn_inv,
            dw= 1.0
        )

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
        ax: float= self.ax * by
        ay: float= self.ay * by
        az: float= self.az * by
        aw: float= self.aw * by
        bx: float= self.bx * by
        by: float= self.by * by
        bz: float= self.bz * by
        bw: float= self.bw * by
        cx: float= self.cx * by
        cy: float= self.cy * by
        cz: float= self.cz * by
        cw: float= self.cw * by
        dx: float= self.dx * by
        dy: float= self.dy * by
        dz: float= self.dz * by
        dw: float= self.dw * by

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
        ax: float= self.ax
        ay: float= self.bx
        az: float= self.cx
        aw: float= self.dx
        bx: float= self.ay
        by: float= self.by
        bz: float= self.cy
        bw: float= self.dy
        cx: float= self.az
        cy: float= self.bz
        cz: float= self.cz
        cw: float= self.dz
        dx: float= self.aw
        dy: float= self.bw
        dz: float= self.cw
        dw: float= self.dw

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
        ax: float= self.ax
        ay: float= -self.ay
        az: float= self.az
        aw: float= -self.aw
        bx: float= -self.bx
        by: float= self.by
        bz: float= -self.bz
        bw: float= self.bw
        cx: float= self.cx
        cy: float= -self.cy
        cz: float= self.cz
        cw: float= -self.cw
        dx: float= -self.dx
        dy: float= self.dy
        dz: float= -self.dz
        dw: float= self.dw

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw
        )

    def inverse(self) -> 'Mat4':
        """Return the inverse of self

        Returns
        ---
        Mat4
        """
        a00: float= self.ax
        a01: float= self.ay
        a02: float= self.az
        a03: float= self.aw
        a10: float= self.bx
        a11: float= self.by
        a12: float= self.bz
        a13: float= self.bw
        a20: float= self.cx
        a21: float= self.cy
        a22: float= self.cz
        a23: float= self.cw
        a30: float= self.dx
        a31: float= self.dy
        a32: float= self.dz
        a33: float= self.dw

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

        det: float = (
            b00 * b11 - b01 *
            b10 + b02 * b09 +
            b03 * b08 - b04 *
            b07 + b05 * b06
        )

        inv: float = 1.0 / det

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
        det: float= self.determinant()

        if is_zero(det):
            raise Mat4Error('length of this Mat4 was zero')

        inv: float= 1.0 / det
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
        det: float= self.determinant()

        if is_zero(det):
            raise Mat4Error('length of this Mat4 was zero')

        return self.scale(1.0 / det)

    def get_translation(self) -> Vec3:
        """Return translation

        Retruns
        ---
        Vec3
        """
        return Vec3(self.dx, self.dy, self.dz)

    def get_rotation(self) -> Vec4:
        """Get rotation from matrix
        
        Returns
        ---
        Vec4
        """
        x: float= 0.0
        y: float= 0.0
        z: float= 0.0
        w: float= 0.0

        if self.ax + self.by + self.cz > 0:
            s: float= sqrt(self.ax + self.by + self.cz + 1.0) * 2.0
            w= 0.25 * s
            x= (self.bz - self.cy) / s
            y= (self.cx - self.az) / s
            z= (self.ay - self.bx) / s
        elif self.ax > self.by and self.ax > self.cz:
            s: float= sqrt(1.0 + self.ax - self.by - self.cz) * 2.0
            w= (self.bz - self.cy) / s
            x= 0.25 * s
            y= (self.ay + self.bx) / s
            z= (self.cx + self.az) / s
        elif self.by > self.cz: 
            s: float= sqrt(1.0 + self.by - self.ax - self.cz) * 2.0
            w= (self.cx - self.az) / s
            x= (self.ay + self.bx) / s
            y= 0.25 * s
            z= (self.bz + self.cy) / s
        else:
            s: float= sqrt(1.0 + self.cz - self.ax - self.by) * 2.0
            w= (self.ay - self.bx) / s
            x= (self.cx + self.az) / s
            y= (self.bz + self.cy) / s
            z= 0.25 * s

        return Vec4(x, y, z, w)

    def get_scale(self) -> Vec3:
        """Return scaler from matrix
        """
        x: float= sqrt(sqr(self.ax) + sqr(self.ay) + sqr(self.az))
        y: float= sqrt(sqr(self.bx) + sqr(self.by) + sqr(self.bz))
        z: float= sqrt(sqr(self.cx) + sqr(self.cy) + sqr(self.cz))

        if self.determinant() < 0:
            x = x * -1.0

        return Vec3(x, y, z)

    def multiply_v4(self, v4: Vec4) -> Vec4:
        """Return mat4 multiplyed by vec4
        
        Returns
        ---
        Vec4
        """
        x: float= (
            v4.x * self.ax +
            v4.y * self.bx +
            v4.z * self.cx +
            v4.w * self.dx
        )

        y: float= (
            v4.x * self.ay +
            v4.y * self.by +
            v4.z * self.cy +
            v4.w * self.dy
        )

        z: float= (
            v4.x * self.az +
            v4.y * self.bz +
            v4.z * self.cz +
            v4.w * self.dz
        )

        w: float= (
            v4.x * self.aw +
            v4.y * self.bw +
            v4.z * self.cw +
            v4.w * self.dw
        )
        
        return Vec4(x, y, z, w)

    def row0(self) -> Vec4:
        """Return the first row of float values

        Returns
        ---
        Vec4
        """
        return Vec4(self.ax, self.ay, self.az, self.aw)

    def row1(self) -> Vec4:
        """Return the second row of float values
        
        Returns
        ---
        Vec4
        """
        return Vec4(self.bx, self.by, self.bz, self.bw)

    def row2(self) -> Vec4:
        """Return the third row of float values

        Returns
        ---
        Vec4
        """
        return Vec4(self.cx, self.cy, self.cz, self.cw)

    def row3(self) -> Vec4:
        """Return the fourth row of float values

        Returns
        ---
        Vec4
        """
        return Vec4(self.dx, self.dy, self.dz, self.dw)

    def col0(self) -> Vec4:
        """Return the first column of float values

        Returns
        ---
        Vec4
        """
        return Vec4(self.ax, self.bx, self.cx, self.dx)

    def col1(self) -> Vec4:
        """Return the second column of float values

        Returns
        ---
        Vec4
        """
        return Vec4(self.ay, self.by, self.cy, self.dy)

    def col2(self) -> Vec4:
        """Return the third column of float values

        Returns
        ---
        Vec4
        """
        return Vec4(self.az, self.bz, self.cz, self.dz)

    def col3(self) -> Vec4:
        """Return the fourth column of float values

        Returns
        ---
        Vec4
        """
        return Vec4(self.aw, self.bw, self.cw, self.dw)

    def at(self, row: int, col: int) -> float:
        return self[col * 4 + row]

    def determinant(self) -> float:
        """Return the determinant of self

        Returns
        ---
        float
        """
        a00: float= self.ax
        a01: float= self.ay
        a02: float= self.az
        a03: float= self.aw
        a10: float= self.bx
        a11: float= self.by
        a12: float= self.bz
        a13: float= self.bw
        a20: float= self.cx
        a21: float= self.cy
        a22: float= self.cz
        a23: float= self.cw
        a30: float= self.dx
        a31: float= self.dy
        a32: float= self.dz
        a33: float= self.dw

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
        """Return the sum of the trace values

        Returns
        ---
        float
        """
        return self.ax + self.by + self.cz + self.dw

    def array(self) -> list[float]:
        """

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
        """

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



# --- QUATERNION



class QuatError(Exception):
    '''Custom error for quaternion'''

    def __init__(self, msg: str):
        super().__init__(msg)



@dataclass(eq=False, repr=False, slots=True)
class Quaternion:
    x: float= 0.0
    y: float= 0.0
    z: float= 0.0
    w: float= 0.0

    def __getitem__(self, idx):
        match clampi(idx, 0, 3):
            case 0:
                return self.x
            case 1:
                return self.y
            case 2:
                return self.z
            case 3:
                return self.w

        raise QuatError('out of range')

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            raise QuatError('not of type Quaternion')

        return Quaternion(
            x= self.x + other.x,
            y= self.y + other.y,
            z= self.z + other.z,
            w= self.w + other.w
        )

    def __sub__(self, other):
        if not isinstance(other, Quaternion):
            raise QuatError('not of type Quaternion')

        return Quaternion(
            x= self.x - other.x,
            y= self.y - other.y,
            z= self.z - other.z,
            w= self.w - other.w
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
            x= x1 * w2 + x2 * w1 + cx,
            y= y1 * w2 + y2 * w1 + cy,
            z= z1 * w2 + z2 * w1 + cz,
            w= w1 * w2 - dt
        )

    @staticmethod
    def from_euler(x_rotation: float, y_rotation: float, z_rotation: float) -> 'Quaternion':
        """Create from euler angles

        Parameters
        ---
        x_rotation : float
        
        y_rotation : float
        
        z_rotation : float
 
        Returns
        ---
        Quaternion
        """
        rx: float= to_radians(x_rotation) * 0.5
        ry: float= to_radians(y_rotation) * 0.5
        rz: float= to_radians(z_rotation) * 0.5

        sx: float= sin(rx)
        cx: float= cos(rx)
        sy: float= sin(ry)
        cy: float= cos(ry)
        sz: float= sin(rz)
        cz: float= cos(rz)

        return Quaternion(
            x= (sx * cy * cz) + (cx * sy * sz),
            y= (cx * sy * cz) + (sx * cy * sz),
            z= (cx * cy * sz) - (sx * sy * cz),
            w= (cx * cy * cz) - (sx * sy * sz)
        )

    @staticmethod
    def from_axis(angle_deg: float, axis: Vec3) -> 'Quaternion':
        """Create a quaternion from an angle and axis of rotation

        Parameters
        ---
        angle_deg : float
            angle in degrees

        axis : Vec3
            axis of rotation

        Returns
        ---
        Quaternion

        """
        cpy: Vec3 = axis.copy()

        if not cpy.is_unit():
            cpy.to_unit()

        angle_rad: float= to_radians(angle_deg * 0.5)
        c: float= cos(angle_rad)
        s: float= sin(angle_rad)

        return Quaternion(
            x= cpy.x * s,
            y= cpy.y * s,
            z= cpy.z * s,
            w= c
        )


    @staticmethod
    def rotate_to(start: Vec3, to: Vec3) -> 'Quaternion':
        """Return rotation between two vec3's
        """
        dot: float= start.dot(to)

        if dot < -1.0 + EPSILON:
            v3: Vec3= Vec3.unit_x().cross(start)

            if v3.length_sqrt() < EPSILON:
                v3= Vec3.unit_y().cross(start)
        
            v3.to_unit()
            return Quaternion.from_axis(to_degreese(PI), v3)

        if dot > absf(-1.0 + EPSILON):
            return Quaternion(w=1.0)
        
        v3: Vec3= start.cross(to)
        return Quaternion(v3.x, v3.y, v3.z, 1.0 + dot)

    def lerp(self, to: 'Quaternion', weight: float) -> 'Quaternion':
        """Return the interpolation between two quaternions

        Parameters
        ---
        begin : Quaternion
        
        end : Quaternion

        weight : float

        Returns
        ---
        Quaternion
        """
        return Quaternion(
            x= lerp(self.x, to.x, weight),
            y= lerp(self.y, to.y, weight),
            z= lerp(self.z, to.z, weight),
            w= lerp(self.w, to.w, weight)
        )

    def nlerp(self, to: 'Quaternion', weight: float) -> 'Quaternion':
        """Return normalized lerp
        """
        return self.lerp(to, weight).unit()

    def slerp(self, to: 'Quaternion', weight: float) -> 'Quaternion':
        """Return the spherical linear interpolation between two quaternions

        Parameters
        ---
        begin : Quaternion

        end : Quaternion

        weight : float

        Returns
        ---
        Quaternion
        """
        scale0: float= 0.0
        scale1: float= 0.0
        v4: Vec4= Vec4(to.x, to.y, to.z, to.w)
        cosine: float= self.dot(to)

        if cosine < 0.0:
            cosine *= -1.0
            v4 = v4 * -1.0
        
        if (1.0 - cosine) > EPSILON:
            omega: float= arccos(cosine)
            sinom: float= sin(omega)
            scale0= sin((1.0 - weight) * omega) / sinom
            scale1= sin(weight * omega) / sinom
        else:
            scale0= 1.0 - weight
            scale1= weight

        return Quaternion(
            x= scale0 * self.x + scale1 * v4.x,
            y= scale0 * self.y + scale1 * v4.y,
            z= scale0 * self.z + scale1 * v4.z,
            w= scale0 * self.w + scale1 * v4.w
        )
 
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
        return Quaternion(
            x= self.x * by,
            y= self.y * by,
            z= self.z * by,
            w= self.w * by
        )

    def inverse(self) -> 'Quaternion':
        """Return the invserse of self

        Returns
        ---
        Quaternion
        """
        inv: float= 1.0 / self.length_sqr()
        return Quaternion(
            x= -self.x * inv,
            y= -self.y * inv,
            z= -self.z * inv,
            w= self.q * inv
        )

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
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            raise QuatError('length was zero')

        return self.scale(inv_sqrt(lsq))

    def to_unit(self) -> None:
        """Normalize the length of self

        Raises
        ---
        QuatError
            length of self was zero
        """
        lsq: float= self.length_sqr()

        if is_zero(lsq):
            raise QuatError('length was zero')

        inv: float= inv_sqrt(lsq)
        self.x *= inv
        self.y *= inv
        self.z *= inv
        self.w *= inv

    def to_euler(self) -> Vec3:
        """Return self as euler values

        Returns
        ---
        Vec3
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

        Returns
        ---
        Vec3
        """
        qt_cpy: Quaternion= self.copy()

        if qt_cpy.w > 1.0:
            qt_cpy.to_unit()

        scl: float= sqrt(1.0 - sqr(qt_cpy.w))

        if is_zero(scl):
            raise QuatError('length of quaternion was zero')

        inv: float = 1.0 / scl
        return Vec3(
            x= self.x * inv,
            y= self.y * inv,
            z= self.z * inv
        )

    def to_mat4(self) -> Mat4:
        """Create a mat4 matrix based on quaternion's current values

        Returns
        ---
        Mat4
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

        return Mat4(
            ax, ay, az, aw,
            bx, by, bz, bw,
            cx, cy, cz, cw,
            dx, dy, dz, dw
        )

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

    def get_rotation_axis_angle(self) -> float:
        """Return the current angle of self in radians

        Returns
        ---
        float
        """
        return arccos(self.w) * 2.0

    def get_rotation_axis(self) -> Vec3:
        """Return the current axis

        Returns
        ---
        Vec3
        """
        s: float= sin(self.get_axis_angle() * 0.5)
        if not is_zero(s):
            inv: float= 1.0 / s
            return Vec3(
                x= self.x * inv,
                y= self.y * inv,
                z= self.z * inv
            )

        return Vec3.unit_x()

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



# --- TRANSFORM



@dataclass(eq= False, repr= False, slots= True)
class Transform:
    position: Vec3= Vec3()
    scale: Vec3= Vec3(1.0, 1.0, 1.0)
    angle: float= 0.0
    axis: Vec3= Vec3(1.0, 1.0, 1.0)

    def get_matrix(self) -> Mat4:
        """Return transform matrix

        Returns
        ---
        glm.Mat4
        """
        
        # Quaternion.from_axis(self.angle, self.axis).to_mat4()

        return (
            Mat4.create_translation(self.position) *
            Mat4.from_axis(self.angle, self.axis) *
            Mat4.create_scaler(self.scale)
        )
    
    def get_matrix_inverse(self) -> Mat4:
        """Return transform matrix inverse

        Returns
        ---
        glm.Mat4
        """
        matrix: Mat4= self.get_matrix()
        return matrix.inverse()
