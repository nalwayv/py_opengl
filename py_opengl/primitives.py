""""""
from dataclasses import dataclass
from py_opengl import maths
from enum import Enum, auto

class Shape(Enum):
    AABB = auto()
    SPHERE = auto()
    PLAIN = auto()
    RAY = auto()

# --- AABB


@dataclass(eq=False, repr=False, slots=True)
class Aabb:
    position: maths.Vec3 = maths.Vec3()
    size: maths.Vec3 = maths.Vec3()
    shape_id: Shape = Shape.AABB

    @staticmethod
    def from_min_max(min_pt: maths.Vec3, max_pt: maths.Vec3) -> 'Aabb':
        return Aabb(
            position=(min_pt + max_pt) * 0.5,
            size=(max_pt - min_pt) * 0.5)

    def copy(self) -> 'Aabb':
        """Return a copy of self

        Returns
        ---
        Abbb
        """
        return Aabb(
            position=self.position.copy(),
            size=self.size.copy())

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        pt_min: maths.Vec3 = self.get_min()
        pt_max: maths.Vec3 = self.get_max()
        pts: list[float] = [0.0, 0.0, 0.0]

        for i in range(3):
            val: float = pt[i]
            val = maths.minf(val, pt_min[i])
            val = maths.maxf(val, pt_max[i])
            pts[i] = val

        return maths.Vec3(pts[0], pts[1], pts[2])        

    def get_min(self) -> maths.Vec3:
        pt1: maths.Vec3 = self.position + self.size
        pt2: maths.Vec3 = self.position - self.size
        
        x: float = maths.minf(pt1.x, pt2.x)
        y: float = maths.minf(pt1.y, pt2.y)
        z: float = maths.minf(pt1.z, pt2.z)

        return maths.Vec3(x, y, z)

    def get_max(self) -> maths.Vec3:
        pt1: maths.Vec3 = self.position + self.size
        pt2: maths.Vec3 = self.position - self.size
        
        x: float = maths.maxf(pt1.x, pt2.x)
        y: float = maths.maxf(pt1.y, pt2.y)
        z: float = maths.maxf(pt1.z, pt2.z)

        return maths.Vec3(x, y, z)

    def intersect_aabb(self, other: 'Aabb') -> bool:
        amin: maths.Vec3 = self.get_min()
        amax: maths.Vec3 = self.get_max()

        bmin: maths.Vec3 = other.get_min()
        bmax: maths.Vec3 = other.get_max()

        check_x: bool = amin.x <= bmax.x and amax.x >= bmin.x
        check_y: bool = amin.y <= bmax.y and amax.y >= bmin.y
        check_z: bool = amin.z <= bmax.z and amax.z >= bmin.z

        return check_x and check_y and check_z

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        amin: maths.Vec3 = self.get_min()
        amax: maths.Vec3 = self.get_max()
        if pt.x < amin.x or pt.y < amin.y or pt.z < amin.z:
            return False
        if pt.x > amax.x or pt.y > amax.y or pt.z > amax.z:
            return False
        return True

    def intersect_sphere(self, sph: 'Sphere') -> bool:
        close_pt: maths.Vec3 = self.closest_pt(sph.position)
        dis: float = (sph.position - close_pt).length_sqr()
        r2 = maths.sqr(sph.radius)
        return dis < r2

    def intersect_plain(self, plain: 'Plain') -> bool:
        len_sq: float = (
            self.size.x * maths.absf(plain.normal.x) +
            self.size.y * maths.absf(plain.normal.y) +
            self.size.z * maths.absf(plain.normal.z)
        )

        dot: float = plain.normal.dot(self.position)
        dis: float = dot - plain.direction

        return maths.absf(dis) <= len_sq


# --- SPHERE


@dataclass(eq=False, repr=False, slots=True)
class Sphere:
    position: maths.Vec3 = maths.Vec3()
    radius: float = 1.0
    shape_id: Shape = Shape.SPHERE

    def copy(self) -> 'Sphere':
        """Return a copy of self

        Returns
        ---
        Sphere
        """
        return Sphere(self.position.copy(), self.radius)

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        p = pt - self.position
        p.to_unit()
        return self.position + (p * self.radius)

    def intersect_sphere(self, other: 'Sphere') -> bool:
        dis: float = (self.position - other.position).length_sqr()
        r2: float = maths.sqr(self.radius + other.radius)
        return dis < r2

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        dis: float = (self.position - pt).length_sqr()
        r2: float = maths.sqr(self.radius)
        return dis < r2

    def intersect_aabb(self, aabb: 'Aabb') -> bool:
        close_pt: maths.Vec3 = aabb.closest_pt(self.position)
        dis: float = (self.position - close_pt).length_sqr()
        r2 = maths.sqr(self.radius)
        return dis < r2

    def intersect_plain(self, plain: 'Plain'):
        close_pt: maths.Vec3 = plain.closest_pt(self.position)
        dis: float = (self.position - close_pt).length_sqr()
        r2: float = maths.sqr(self.radius)
        return dis < r2


# --- PLAIN


class PlainError(Exception):
    '''Custom error for plain'''

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Plain:
    normal: maths.Vec3 = maths.Vec3(x=1.0)
    direction: float = 0.0
    shape_id: Shape = Shape.PLAIN

    def __post_init__(self):
        if not self.normal.is_unit():
            self.normal.to_unit()

    def copy(self) -> 'Plain':
        """Return a copy of self

        Returns
        ---
        Plain
        """
        return Plain(
            normal=self.normal.copy(),
            direction=self.direction)

    def dot(self, v4: maths.Vec4) -> float:
        x: float = self.normal.x * v4.x
        y: float = self.normal.y * v4.y
        z: float = self.normal.z * v4.z
        w: float = self.direction * v4.w
        return x + y + z + w

    def dot_coord(self, v3: maths.Vec3) -> float:
        x: float = self.normal.x * v3.x
        y: float = self.normal.y * v3.y
        z: float = self.normal.z * v3.z
        w: float = self.direction
        return x + y + z + w

    def dot_normal(self, v3: maths.Vec3) -> float:
        x: float = self.normal.x * v3.x
        y: float = self.normal.y * v3.y
        z: float = self.normal.z * v3.z
        return x + y + z

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        t: float = (self.normal.dot(pt) - self.direction) / self.normal.length_sqr()
        return pt - (self.normal * t)

    def unit(self) -> 'Plain':
        """Return a copy of self that has been normalized

        Returns
        ---
        Plain
        """
        len_sqr: float = self.normal.length_sqr()

        if maths.is_one(len_sqr):
            return self.copy()

        inv: float = maths.inv_sqrt(len_sqr)
        return Plain(
            normal=self.normal * inv,
            direction=self.direction * inv)

    def to_unit(self) -> None:
        """Normalize the length of self
        """
        len_sqr: float = self.normal.length_sqr()
        if maths.is_zero(len_sqr):
            raise PlainError('length of plain was zero')

        inv: float = maths.inv_sqrt(len_sqr)

        self.normal.x *= inv
        self.normal.y *= inv
        self.normal.z *= inv
        self.direction *= inv

    def intersect_plain(self, other: 'Plain') -> bool:
        dis: float = (self.normal.cross(other.nomal)).length_sqr()
        return not maths.is_zero(dis)

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        dot: float = pt.dot(self.normal)
        return maths.is_zero(dot - self.direction)

    def intersect_sphere(self, sph: 'Sphere') -> bool:
        close_pt = self.closest_pt(sph.position)
        len_sq = (sph.position - close_pt).length_sqr()
        r2 = maths.sqr(sph.radius)
        return len_sq < r2

    def intersect_aabb(self, aabb: 'Aabb') -> bool:
        len_sq: float = (
            aabb.size.x * maths.absf(self.normal.x) +
            aabb.size.y * maths.absf(self.normal.y) +
            aabb.size.z * maths.absf(self.normal.z))

        dot: float = self.normal.dot(aabb.position)
        dis: float = dot - self.direction

        return maths.absf(dis) <= len_sq


# --- RAY3D


@dataclass(eq=False, repr=False, slots=True)
class Ray:
    origin: maths.Vec3 = maths.Vec3()
    direction: maths.Vec3 = maths.Vec3(z=1.0)
    shape_id: Shape = Shape.RAY

    @staticmethod
    def from_points(origin: maths.Vec3, target:maths.Vec3) -> 'Ray':
        o = origin.copy()
        d = target - origin
        if not d.is_unit():
            d.to_unit()

        return Ray(origin=o, direction=d)

    def copy(self) -> 'Ray':
        o = self.origin.copy()
        d = self.direction.copy()
        return Ray(origin=o, direction=d)

    def get_hit(self, t: float) -> maths.Vec3:
        """Return point along ray

        Parameters
        ---
        t : float

        Returns
        ---
        Vec3
        """
        dir = self.direction.copy()
        if not dir.is_unit():
            dir.to_unit()
        return self.origin + (dir * t)
    
    def cast_aabb(self, aabb: Aabb) -> tuple[bool, maths.Vec3]:
        amin: maths.Vec3 = aabb.get_min()
        amax: maths.Vec3 = aabb.get_max()
        tmin: float = maths.MIN_FLOAT
        tmax: float = maths.MAX_FLOAT

        dir = self.direction.copy()
        if not dir.is_unit():
            dir.to_unit()

        for idx in range(3):
            if maths.is_zero(dir[idx]):
                if self.origin[idx] < amin[idx] or self.origin[idx] > amax[idx]:
                    return False, maths.Vec3()
            else:
                inv: float = 1.0 / dir[idx]

                t1: float = (amin[idx] - self.origin[idx]) * inv
                t2: float = (amax[idx] - self.origin[idx]) * inv

                if t1 > t2:
                    t1, t2 = t2, t1
        
                if t1 > tmin:
                    tmin = t1

                if t2 > tmax:
                    tmax = t2

                if tmin > tmax:
                    return False, maths.Vec3()
    
        return True, self.get_point(tmin)

    def cast_sphere(self, sph: Sphere) -> tuple[bool, maths.Vec3]:
        dir = self.direction.copy()
        if not dir.is_unit():
            dir.to_unit()

        a: maths.Vec3 = sph.position - self.origin
        b: float = a.dot(dir)
        c: float = a.length_sqr() - maths.sqr(sph.radius)

        if c > 0.0 and b > 0.0:
            return False, maths.Vec3()

        d: float = maths.sqr(b) - c
        if d < 0.0:
            return False, maths.Vec3()
        
        t: float = -b - maths.sqrt(d)
        if t < 0.0:
            t = 0.0

        return True, self.get_point(t)
