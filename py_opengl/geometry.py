"""Geometry
"""
from dataclasses import dataclass
from enum import Enum, auto

from py_opengl import maths


# --- IDs


class GeometryID(Enum):
    LINE= auto()
    AABB3= auto()
    SPHERE= auto()
    PLAIN= auto()
    FRUSTUM= auto()
    RAY= auto()


# --- AABB


@dataclass(eq= False, repr= True, slots= True)
class AABB3:
    """AABB using center and extents


    Aabb(center= Vec3(0, 0, 0), extents= Vec3(1, 1, 1))

    get_min == Vec3(-1, -1, -1)\n
    get_max == Vec3(1, 1, 1)
    """
    center: maths.Vec3= maths.Vec3()
    extents: maths.Vec3= maths.Vec3()

    _id: GeometryID= GeometryID.AABB3

    @staticmethod
    def from_min_max(min_pt: maths.Vec3, max_pt: maths.Vec3) -> 'AABB3':
        return AABB3(
            center= (min_pt + max_pt) * 0.5,
            extents= ((min_pt - max_pt) * 0.5).abs()
        )

    def create_combined_with(self, other: 'AABB3') -> 'AABB3':
        return AABB3.from_min_max(
            maths.Vec3.create_from_min(self.get_min(), other.get_min()),
            maths.Vec3.create_from_max(self.get_max(), other.get_max())
        )

    def combined_with(self, a: 'AABB3') -> None:
        """Set self to the union of self and a"""
        aabb= AABB3.from_min_max(
            maths.Vec3.create_from_min(self.get_min(), a.get_min()),
            maths.Vec3.create_from_max(self.get_max(), a.get_max())
        )
        self.center= aabb.center
        self.extents= aabb.extents

    def combined_from(self, a: 'AABB3', b: 'AABB3') -> None:
        """Set self to the union of a and b"""
        aabb= AABB3.from_min_max(
            maths.Vec3.create_from_min(a.get_min(), b.get_min()),
            maths.Vec3.create_from_max(a.get_max(), b.get_max())
        )
        self.center= aabb.center
        self.extents= aabb.extents

    def expand(self, by: float) -> 'AABB3':
        """Expanded by float value"""
        if by < 0.0:
            by = maths.absf(by)

        p0: maths.Vec3= self.get_min() - maths.Vec3.create_from_value(by)
        p1: maths.Vec3= self.get_max() + maths.Vec3.create_from_value(by)

        return AABB3.from_min_max(p0, p1)

    def expand_to(self, v3: maths.Vec3) -> 'AABB3':
        """Expand by vec3"""
        begin: maths.Vec3= self.copy()
        end: maths.Vec3= self.center + self.extents

        if v3.x < begin.x:
            begin.x= v3.x
        if v3.y < begin.y:
            begin.y= v3.y
        if v3.z < begin.z:
            begin.z= v3.z

        if v3.x > end.x:
            end.x= v3.x
        if v3.y > end.y:
            end.y= v3.y
        if v3.z > end.z:
            end.z= v3.z

        return AABB3(
            center= begin,
            extents= end - begin
        )

    def expanded_to(self, v3: maths.Vec3) -> None:
        begin: maths.Vec3= self.copy()
        end: maths.Vec3= self.center + self.extents

        if v3.x < begin.x:
            begin.x= v3.x
        if v3.y < begin.y:
            begin.y= v3.y
        if v3.z < begin.z:
            begin.z= v3.z

        if v3.x > end.x:
            end.x= v3.x
        if v3.y > end.y:
            end.y= v3.y
        if v3.z > end.z:
            end.z= v3.z

        self.center.set_from(begin)
        self.extents.set_from(end - begin)

    def expanded(self, by: float) -> None:
        expand= self.expand(by)
        self.center.set_from(expand.center)
        self.extents.set_from(expand.extents)

    def get_size_x(self) -> float:
        p0= self.get_min()
        p1= self.get_max()
        return p1.x - p0.x

    def get_size_y(self) -> float:
        p0= self.get_min()
        p1= self.get_max()
        return p1.y - p0.y

    def get_size_z(self) -> float:
        p0= self.get_min()
        p1= self.get_max()
        return p1.z - p0.z

    def copy(self) -> 'AABB3':
        """Return a copy of self
        """
        return AABB3(
            center= self.center.copy(),
            extents= self.extents.copy()
        )

    def set_from(self, other: 'AABB3') -> None:
        self.center.set_from(other.center)
        self.extents.set_from(other.extents)

    def perimeter(self) -> float:
        p0: maths.Vec3= self.get_min()
        p1: maths.Vec3= self.get_max()

        p2: maths.Vec3= p1 - p0

        return 4.0 * p2.sum()

    def get_area(self) -> float:
        p0= self.get_min()
        p1= self.get_max()

        return(p1 - p0).sum()

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        p0: maths.Vec3= self.get_min()
        p1: maths.Vec3= self.get_max()
        p2: maths.Vec3= maths.Vec3.zero()

        for i in range(3):
            val: float= pt.get_at(i)
            val= maths.minf(val, p0.get_at(i))
            val= maths.maxf(val, p1.get_at(i))
            p2.set_at(i, val)

        return p2

    def get_min(self) -> maths.Vec3:
        p0: maths.Vec3= self.center + self.extents
        p1: maths.Vec3= self.center - self.extents

        return maths.Vec3.create_from_min(p0, p1)

    def get_max(self) -> maths.Vec3:
        p0: maths.Vec3= self.center + self.extents
        p1: maths.Vec3= self.center - self.extents

        return maths.Vec3.create_from_max(p0, p1)

    def is_degenerate(self) -> bool:
        """Return true is self aabb is degenerate

        check if its min points equil its max points
        """
        amin= self.get_min()
        bmax= self.get_max()

        check_x= maths.is_equil(amin.x, bmax.x)
        check_y= maths.is_equil(amin.y, bmax.y)
        check_z= maths.is_equil(amin.z, bmax.z)

        return check_x and check_y and check_z

    def contains_aabb(self, other: 'AABB3') -> bool:
        amin: maths.Vec3= self.get_min()
        amax: maths.Vec3= self.get_max()
        bmin: maths.Vec3= other.get_min()
        bmax: maths.Vec3= other.get_max()

        check_x: bool= amin.x <= bmin.x and amax.x >= bmax.x
        check_y: bool= amin.y <= bmin.y and amax.y >= bmax.y
        check_z: bool= amin.z <= bmin.z and amax.z >= bmax.z

        return check_x and check_y and check_z

    def intersect_aabb(self, other: 'AABB3') -> bool:
        amin: maths.Vec3= self.get_min()
        amax: maths.Vec3= self.get_max()
        bmin: maths.Vec3= other.get_min()
        bmax: maths.Vec3= other.get_max()

        check_x: bool= amin.x <= bmax.x and amax.x >= bmin.x
        check_y: bool= amin.y <= bmax.y and amax.y >= bmin.y
        check_z: bool= amin.z <= bmax.z and amax.z >= bmin.z

        return check_x and check_y and check_z

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        amin: maths.Vec3= self.get_min()
        amax: maths.Vec3= self.get_max()

        if pt.x < amin.x or pt.y < amin.y or pt.z < amin.z:
            return False

        if pt.x > amax.x or pt.y > amax.y or pt.z > amax.z:
            return False

        return True


# --- Line


@dataclass(eq= False, repr= False, slots= True)
class Line3:
    start: maths.Vec3= maths.Vec3()
    end: maths.Vec3= maths.Vec3()

    _id: GeometryID= GeometryID.LINE

    def __hash__(self):
        return hash((self.start.x, self.start.y, self.end.x, self.end.y))

    def __eq__(self, other: 'Line3') -> bool:
        check_s= self.start.is_equil(other.start)
        check_e= self.end.is_equil(other.end)
        check_id= self._id == other._id
        return check_s and check_e and check_id

    def edge(self) -> maths.Vec3:
        return (self.end - self.start)



# --- SPHERE


@dataclass(eq= False, repr= False, slots= True)
class Sphere3:
    center: maths.Vec3= maths.Vec3()
    radius: float= 1.0
    _id: GeometryID= GeometryID.SPHERE

    def __hash__(self):
        return hash((self.radius, self.center.x, self.center.y, self.center.z))

    def __eq__(self, other: 'Sphere3') -> bool:
        check_r= maths.is_equil(self.radius, other.radius)
        check_p= self.center.is_equil(other.center)
        check_id= self._id == other._id
        return check_r and check_p and check_id

    def compute_aabb(self) -> AABB3:
        p0: maths.Vec3= self.center + maths.Vec3.create_from_value(self.radius)
        p1: maths.Vec3= self.center - maths.Vec3.create_from_value(self.radius)

        return AABB3.from_min_max(p0, p1)

    def area(self) -> float:
        """Return area
        """
        return maths.PI * maths.sqr(self.radius)

    def copy(self) -> 'Sphere3':
        """Return a copy of self
        """
        return Sphere3(self.center.copy(), self.radius)

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        point: maths.Vec3= pt - self.center
        
        if not point.is_unit():
            point.to_unit()

        return self.center + (point * self.radius)

    def furthest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        if not pt.is_unit():
            pt.to_unit()
        
        return self.center + (pt * self.radius)

    def intersect_sphere(self, other: 'Sphere3') -> bool:
        dis: float= (self.center - other.center).length_sqr()
        r2: float= maths.sqr(self.radius + other.radius)

        return dis < r2

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        dis: float= (self.center - pt).length_sqr()
        r2: float= maths.sqr(self.radius)

        return dis < r2

    def intersect_aabb(self, aabb: 'AABB3') -> bool:
        close_pt: maths.Vec3= aabb.closest_pt(self.center)
        dis: float= (self.center - close_pt).length_sqr()
        r2: float= maths.sqr(self.radius)

        return dis < r2

    def intersect_plain(self, plain: 'Plain3'):
        close_pt: maths.Vec3= plain.closest_pt(self.center)
        dis: float= (self.center - close_pt).length_sqr()
        r2: float= maths.sqr(self.radius)

        return dis < r2


# --- PLAIN


class PlainError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Plain3:
    normal: maths.Vec3= maths.Vec3(x=1.0)
    direction: float= 0.0
    _id: GeometryID= GeometryID.PLAIN

    def __hash__(self):
        return hash((self.direction, self.normal.x, self.normal.y, self.normal.z))

    def __eq__(self, other: 'Plain3') -> bool:
        check_d= maths.is_equil(self.direction, other.direction)
        check_n= self.normal.is_equil(other.normal)
        check_id= self._id == other._id
        return check_d and check_n and check_id

    def __post_init__(self):
        if not self.normal.is_unit():
            self.normal.to_unit()

    @staticmethod
    def create_from_normal_and_point(unit_v3: maths.Vec3, pt: maths.Vec3):
        if not unit_v3.is_unit():
            unit_v3.to_unit()

        n: maths.Vec3= unit_v3.copy()
        d: float= n.dot(pt)

        return Plain3(normal=n, direction=d)

    @staticmethod
    def create_from_points(
        a: maths.Vec3,
        b: maths.Vec3,
        c: maths.Vec3
    ) -> 'Plain3':
        v0: maths.Vec3= b - a
        v1: maths.Vec3= c - a

        n: maths.Vec3= v0.cross(v1)
        if not n.is_unit():
            n.to_unit()

        d: float= n.dot(a)

        return Plain3(normal= n, direction= d)

    def to_str(self) -> str:
        return f'N(X: {self.normal.x}, Y: {self.normal.y}, Z: {self.normal.z}), D({self.direction})'

    def copy(self) -> 'Plain3':
        """Return a copy of self
        """
        return Plain3(
            normal= self.normal.copy(),
            direction= self.direction
        )

    def dot(self, v4: maths.Vec4) -> float:
        x: float= self.normal.x * v4.x
        y: float= self.normal.y * v4.y
        z: float= self.normal.z * v4.z
        w: float= self.direction * v4.w

        return x + y + z + w

    def dot_normal(self, v3: maths.Vec3) -> float:
        x: float= self.normal.x * v3.x
        y: float= self.normal.y * v3.y
        z: float= self.normal.z * v3.z

        return x + y + z

    def closest_point(self, point: maths.Vec3) -> maths.Vec3:
        scale: float= (self.normal.dot(point) - self.direction) / self.normal.length_sqr()
        return point - (self.normal * scale)

    def project_point_onto_plain(self, pt: maths.Vec3) -> maths.Vec3:
        scale: float= pt.dot(self.normal) - self.direction
        return pt - (self.normal * scale)

    def unit(self) -> 'Plain3':
        """Return a copy of self with unit length
        """
        len_sqr: float= self.normal.length_sqr()

        normal: maths.Vec3= maths.Vec3.create_unit_x()
        direction: float= 0.0

        if maths.is_one(len_sqr):
            normal.set_from(self.normal)
            direction= self.direction
        else:
            inv: float= maths.inv_sqrt(len_sqr)
            normal= self.normal * inv
            direction= self.direction * inv

        return Plain3(normal, direction)

    def to_unit(self) -> None:
        """Convert to unit length
        """
        len_sqr: float= self.normal.length_sqr()
        
        if maths.is_zero(len_sqr):
            return

        inv: float= maths.inv_sqrt(len_sqr)
        self.normal.x *= inv
        self.normal.y *= inv
        self.normal.z *= inv
        self.direction *= inv

    def intersect_plain(self, other: 'Plain3') -> bool:
        dis: float= (self.normal.cross(other.nomal)).length_sqr()
        return not maths.is_zero(dis)

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        dot: float= pt.dot(self.normal)
        return maths.is_zero(dot - self.direction)

    def intersect_sphere(self, sph: 'Sphere3') -> bool:
        close_pt: maths.Vec3= self.closest_point(sph.center)
        len_sq: float= (sph.center - close_pt).length_sqr()

        return len_sq < maths.sqr(sph.radius)

    def intersect_aabb(self, aabb: 'AABB3') -> bool:
        len_sq: float = (
            aabb.extents.x * maths.absf(self.normal.x) +
            aabb.extents.y * maths.absf(self.normal.y) +
            aabb.extents.z * maths.absf(self.normal.z)
        )

        dot: float= self.normal.dot(aabb.center)
        dis: float= dot - self.direction

        return maths.absf(dis) <= len_sq


# --- FRUSTUM


@dataclass(eq= False, repr= False, slots= True)
class Frustum:
    top: Plain3= Plain3()
    bottom: Plain3= Plain3()
    left: Plain3= Plain3()
    right: Plain3= Plain3()
    near: Plain3= Plain3()
    far: Plain3= Plain3()

    _id = GeometryID.FRUSTUM


# --- RAY3D


@dataclass(eq=False, repr=False, slots=True)
class Ray3:
    origin: maths.Vec3= maths.Vec3()
    direction: maths.Vec3= maths.Vec3(z=1.0)
    
    _id: GeometryID= GeometryID.RAY

    def __post_init__(self):
        if not self.direction.is_unit():
            self.direction.to_unit()

    @staticmethod
    def from_points(origin: maths.Vec3, target: maths.Vec3) -> 'Ray3':
        o: maths.Vec3= origin.copy()
        d: maths.Vec3= target - origin

        if not d.is_unit():
            d.to_unit()

        return Ray3(
            origin= o,
            direction= d
        )

    def set_direction(self, unit_dir: maths.Vec3) -> None:
        if unit_dir.is_zero():
            return

        if not unit_dir.is_unit():
            unit_dir.to_unit()

        self.direction.x = unit_dir.x
        self.direction.y = unit_dir.y

    def copy(self) -> 'Ray3':
        return Ray3(
            origin= self.origin.copy(),
            direction= self.direction.copy()
        )

    def get_hit(self, t: float) -> maths.Vec3:
        """Return point along ray
        """
        return self.origin + (self.direction * t)

    def cast_aabb(self, aabb: AABB3) -> tuple[bool, maths.Vec3]:
        amin: maths.Vec3= aabb.get_min()
        amax: maths.Vec3= aabb.get_max()
        tmin: float= maths.MIN_FLOAT
        tmax: float= maths.MAX_FLOAT

        for idx in range(3):
            if maths.is_zero(self.direction.get_at(idx)):
                if(
                    self.origin.get_at(idx) < amin.get_at(idx) or 
                    self.origin.get_at(idx) > amax.get_at(idx)
                ):
                    return False, maths.Vec3.zero()
            else:
                inv: float= 1.0 / self.direction.get_at(idx)

                t1: float= (amin.get_at(idx) - self.origin.get_at(idx)) * inv
                t2: float= (amax.get_at(idx) - self.origin.get_at(idx)) * inv

                if t1 > t2:
                    t1, t2= t2, t1

                if t1 > tmin:
                    tmin= t1

                if t2 > tmax:
                    tmax= t2

                if tmin > tmax:
                    return False, maths.Vec3.zero()

        return True, self.get_hit(tmin)

    def cast_sphere(self, sph: Sphere3) -> tuple[bool, maths.Vec3]:
        a: maths.Vec3= sph.center - self.origin
        b: float= a.dot(self.direction)
        c: float= a.length_sqr() - maths.sqr(sph.radius)

        if c > 0.0 and b > 0.0:
            return False, maths.Vec3.zero()

        d: float= maths.sqr(b) - c
        if d < 0.0:
            return False, maths.Vec3.zero()

        t: float= -b - maths.sqrt(d)
        if t < 0.0:
            t= 0.0

        return True, self.get_hit(t)

    def cast_plain(self, pl: Plain3) -> tuple[bool, maths.Vec3]:
        nd: float= self.direction.dot(pl.normal)
        pn: float= self.origin.dot(pl.normal)

        if nd >= 0.0:
            return False, maths.Vec3.zero()

        t: float= (pl.direction - pn) / nd

        if t >= 0.0:
            return False, maths.Vec3.zero()

        return True, self.get_hit(t)
