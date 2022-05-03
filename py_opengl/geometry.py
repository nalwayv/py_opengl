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

    id: GeometryID= GeometryID.AABB3

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

    def combined_from(self, a:'AABB3', b:'AABB3') -> None:
        """Set self to the union of a and b"""
        aabb= AABB3.from_min_max(
            maths.Vec3.create_from_min(a.get_min(), b.get_min()),
            maths.Vec3.create_from_max(a.get_max(), b.get_max())
        )
        self.center= aabb.center
        self.extents= aabb.extents

    def expand(self, by: float) -> 'AABB3':
        if by < 0.0:
            by = maths.absf(by)

        p0: maths.Vec3= self.get_min() - maths.Vec3.create_from_value(by)
        p1: maths.Vec3= self.get_max() + maths.Vec3.create_from_value(by)

        return AABB3.from_min_max(p0, p1)

    def expanded(self, by: float) -> None:
        expand= self.expand(by)
        self.center.copy_from(expand.center)
        self.extents.copy_from(expand.extents)

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

        Returns
        ---
        Abbb
        """
        return AABB3(
            center= self.center.copy(),
            extents= self.extents.copy()
        )

    def copy_from(self, other: 'AABB3') -> None:
        self.center.copy_from(other.center)
        self.extents.copy_from(other.extents)

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
        p2: list[float]= [0.0, 0.0, 0.0]

        for i in range(3):
            val: float= pt[i]
            val= maths.minf(val, p0[i])
            val= maths.maxf(val, p1[i])
            p2[i]= val

        return maths.Vec3(p2[0], p2[1], p2[2])

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
        
        Returns
        ---
        bool
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

    def intersect_sphere(self, sph: 'Sphere3') -> bool:
        close_pt: maths.Vec3= self.closest_pt(sph.center)
        dis: float= (sph.center - close_pt).length_sqr()
        r2= maths.sqr(sph.radius)

        return dis < r2

    def intersect_plain(self, plain: 'Plain') -> bool:
        len_sq: float = (
            self.extents.x * maths.absf(plain.normal.x) +
            self.extents.y * maths.absf(plain.normal.y) +
            self.extents.z * maths.absf(plain.normal.z)
        )

        dot: float= plain.normal.dot(self.center)
        dis: float= dot - plain.direction

        return maths.absf(dis) <= len_sq


# --- LINE


@dataclass(eq= False, repr= False, slots= True)
class Line3:
    start: maths.Vec3= maths.Vec3()
    end: maths.Vec3= maths.Vec3()
    id: GeometryID= GeometryID.LINE

    def __hash__(self):
        return hash((self.start.x, self.start.y, self.end.x, self.end.y))
    
    def __eq__(self, other: 'Line3') -> bool:
        check_s= self.start.is_equil(other.start)
        check_e= self.end.is_equil(other.end)
        check_id= self.id == other.id
        return check_s and check_e and check_id

    def length(self) -> float:
        return self.start.distance_sqrt(self.end)


# --- SPHERE


@dataclass(eq= False, repr= False, slots= True)
class Sphere3:
    center: maths.Vec3= maths.Vec3()
    radius: float= 1.0
    id: GeometryID= GeometryID.SPHERE

    def __hash__(self):
        return hash((self.radius, self.center.x, self.center.y, self.center.z))
    
    def __eq__(self, other: 'Sphere3') -> bool:
        check_r= maths.is_equil(self.radius, other.radius)
        check_p= self.center.is_equil(other.center)
        check_id= self.id == other.id
        return check_r and check_p and check_id

    def compute_aabb(self) -> AABB3:
        p0: maths.Vec3= self.center + maths.Vec3.create_from_value(self.radius)
        p1: maths.Vec3= self.center - maths.Vec3.create_from_value(self.radius)

        return AABB3.from_min_max(p0, p1)

    def area(self) -> float:
        """Return area

        Returns
        ---
        float
        """
        return maths.PI * maths.sqr(self.radius)

    def copy(self) -> 'Sphere3':
        """Return a copy of self

        Returns
        ---
        Sphere
        """
        return Sphere3(self.center.copy(), self.radius)

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        point: maths.Vec3= pt - self.center
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

    def intersect_plain(self, plain: 'Plain'):
        close_pt: maths.Vec3= plain.closest_pt(self.center)
        dis: float= (self.center - close_pt).length_sqr()
        r2: float= maths.sqr(self.radius)

        return dis < r2


# --- PLAIN


class PlainError(Exception):
    """Custom error for plain"""

    def __init__(self, msg: str):
        super().__init__(msg)


@dataclass(eq=False, repr=False, slots=True)
class Plain:
    normal: maths.Vec3= maths.Vec3(x=1.0)
    direction: float= 0.0
    id: GeometryID= GeometryID.PLAIN

    def __hash__(self):
        return hash((self.direction, self.normal.x, self.normal.y, self.normal.z))
    
    def __eq__(self, other: 'Plain') -> bool:
        check_d= maths.is_equil(self.direction, other.direction)
        check_n= self.normal.is_equil(other.normal)
        check_id= self.id == other.id
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

        return Plain(normal=n, direction=d)

    @staticmethod
    def create_from_points(a: maths.Vec3, b: maths.Vec3, c: maths.Vec3) -> 'Plain':
        v0: maths.Vec3= b - a
        v1: maths.Vec3= c - a
        
        n: maths.Vec3= v0.cross(v1)
        if not n.is_unit():
            n.to_unit()

        d: float= n.dot(a)
        
        return Plain(normal= n, direction= d)

    def copy(self) -> 'Plain':
        """Return a copy of self

        Returns
        ---
        Plain
        """
        return Plain(
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

    def unit(self) -> 'Plain':
        """Return a copy of self that has been normalized

        Returns
        ---
        Plain
        """
        len_sqr: float= self.normal.length_sqr()

        normal: maths.Vec3= maths.Vec3.create_unit_x()
        direction: float= 0.0

        if maths.is_one(len_sqr):
            normal.x= self.normal.x
            normal.y= self.normal.y
            normal.z= self.normal.z
            direction= self.direction
        else:
            inv: float= maths.inv_sqrt(len_sqr)

            normal= self.normal * inv
            direction= self.direction * inv

        return Plain(normal, direction)

    def to_unit(self) -> None:
        """Normalize the length of self
        """
        len_sqr: float= self.normal.length_sqr()
        if maths.is_zero(len_sqr):
            raise PlainError('length of plain was zero')

        inv: float= maths.inv_sqrt(len_sqr)

        self.normal.x *= inv
        self.normal.y *= inv
        self.normal.z *= inv
        self.direction *= inv

    def intersect_plain(self, other: 'Plain') -> bool:
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


# --- RAY3D


@dataclass(eq=False, repr=False, slots=True)
class Ray3:
    origin: maths.Vec3= maths.Vec3()
    direction: maths.Vec3= maths.Vec3(z=1.0)
    id: GeometryID= GeometryID.RAY

    def __post_init__(self):
        if not self.direction.is_unit():
            self.direction.to_unit()

    @staticmethod
    def from_points(origin: maths.Vec3, target:maths.Vec3) -> 'Ray3':
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

        Parameters
        ---
        t : float

        Returns
        ---
        Vec3
        """
        return self.origin + (self.direction * t)
    
    def cast_aabb(self, aabb: AABB3) -> tuple[bool, maths.Vec3]:
        amin: maths.Vec3= aabb.get_min()
        amax: maths.Vec3= aabb.get_max()
        tmin: float= maths.MIN_FLOAT
        tmax: float= maths.MAX_FLOAT

        for idx in range(3):
            if maths.is_zero(self.direction[idx]):
                if self.origin[idx] < amin[idx] or self.origin[idx] > amax[idx]:
                    return False, maths.Vec3.zero()
            else:
                inv: float= 1.0 / self.direction[idx]

                t1: float= (amin[idx] - self.origin[idx]) * inv
                t2: float= (amax[idx] - self.origin[idx]) * inv

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

    def cast_plain(self, pl: Plain) -> tuple[bool, maths.Vec3]:
        nd: float= self.direction.dot(pl.normal)
        pn: float= self.origin.dot(pl.normal)
 
        if nd >= 0.0:
            return False, maths.Vec3.zero()
        
        t: float= (pl.direction - pn) / nd
        
        if t >= 0.0:
            return False, maths.Vec3.zero()

        return True, self.get_hit(t)