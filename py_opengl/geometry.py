"""Geometry
"""
from audioop import cross
from pydoc import plain
from re import L
from typing import Final
from enum import Enum, auto

from py_opengl import maths


# --- IDs


class GeometryType(Enum):
    AABB3= auto()
    LINE= auto()
    TRIANGLE= auto()
    SPHERE= auto()
    PLAIN= auto()
    RAY= auto()
    FRUSTUM= auto()


# --- AABB


class AABB3:
    """AABB using center and extents


    Aabb(center= Vec3(0, 0, 0), extents= Vec3(1, 1, 1))

    get_min == Vec3(-1, -1, -1)\n
    get_max == Vec3(1, 1, 1)
    """

    __slots__= ('center', 'extents', 'TYPE')

    def __init__(
        self,
        center: maths.Vec3= maths.Vec3(),
        extents: maths.Vec3= maths.Vec3(),
    ) -> None:
        self.center: maths.Vec3= center
        self.extents: maths.Vec3= extents
        self.TYPE: GeometryType= GeometryType.AABB3

    def __eq__(self, other: 'AABB3') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.center.is_equil(other.center) and
                self.extents.is_equil(other.extents) and
                self.TYPE == other.TYPE
            ):
                return True
        return False

    def __str__(self) -> str:
        pmin= self.get_min()
        pmax= self.get_max()
        return f'[MIN: {pmin}, MAX: {pmax}]'

    @staticmethod
    def create_from_min_max(min_pt: maths.Vec3, max_pt: maths.Vec3) -> 'AABB3':
        return AABB3(
            center= (min_pt + max_pt) * 0.5,
            extents= (min_pt - max_pt) * 0.5
        )

    @staticmethod
    def create_combined_from(a: 'AABB3', b: 'AABB3') -> 'AABB3':
        return AABB3.create_from_min_max(
            maths.Vec3.create_from_min(a.get_min(), b.get_min()),
            maths.Vec3.create_from_max(a.get_max(), b.get_max())
        )

    def combined_with(self, a: 'AABB3') -> None:
        """Set self to the union of self and a"""
        aabb= AABB3.create_from_min_max(
            maths.Vec3.create_from_min(self.get_min(), a.get_min()),
            maths.Vec3.create_from_max(self.get_max(), a.get_max())
        )
        self.center.set_from(aabb.center)
        self.extents.set_from(aabb.extents)

    def combine_with(self, a: 'AABB3') -> 'AABB3':
        """Set self to the union of self and a"""
        return AABB3.create_from_min_max(
            maths.Vec3.create_from_min(self.get_min(), a.get_min()),
            maths.Vec3.create_from_max(self.get_max(), a.get_max())
        )

    def combined_from(self, a: 'AABB3', b: 'AABB3') -> None:
        """Set self to the union of a and b"""
        aabb= AABB3.create_from_min_max(
            maths.Vec3.create_from_min(a.get_min(), b.get_min()),
            maths.Vec3.create_from_max(a.get_max(), b.get_max())
        )
        self.center.set_from(aabb.center)
        self.extents.set_from(aabb.extents)

    def expand(self, by: float) -> 'AABB3':
        """Expanded by float value"""
        if by < 0.0:
            by = maths.absf(by)

        p0: maths.Vec3= self.get_min() - maths.Vec3.create_from_value(by)
        p1: maths.Vec3= self.get_max() + maths.Vec3.create_from_value(by)

        return AABB3.create_from_min_max(p0, p1)

    def expanded(self, by: float) -> None:
        expand= self.expand(by)
        self.center.set_from(expand.center)
        self.extents.set_from(expand.extents)

    def translate(self, v3: maths.Vec3) -> None:
        """Translate by
        """
        self.center.set_from(self.center + v3)

    def copy(self) -> 'AABB3':
        """Return a copy of self
        """
        return AABB3(
            center= self.center.copy(),
            extents= self.extents.copy()
        )

    def perimeter(self) -> float:
        pmin: maths.Vec3= self.get_min()
        pmax: maths.Vec3= self.get_max()

        return 4.0 * (pmax - pmin).sum()

    def get_area(self) -> float:
        pmin= self.get_min()
        pmax= self.get_max()

        return(pmax - pmin).sum()

    def closest_pt(self, pt: maths.Vec3) -> maths.Vec3:
        pmin: maths.Vec3= self.get_min()
        pmax: maths.Vec3= self.get_max()
        pt: maths.Vec3= maths.Vec3.zero()

        for i in range(3):
            val: float= pt.get_at(i)
            val= maths.minf(val, pmin.get_at(i))
            val= maths.maxf(val, pmax.get_at(i))
            pt.set_at(i, val)

        return pt

    def get_min(self) -> maths.Vec3:
        p0: maths.Vec3= self.center + self.extents
        p1: maths.Vec3= self.center - self.extents

        return maths.Vec3.create_from_min(p0, p1)

    def get_max(self) -> maths.Vec3:
        p0: maths.Vec3= self.center + self.extents
        p1: maths.Vec3= self.center - self.extents

        return maths.Vec3.create_from_max(p0, p1)

    def is_degenerate(self) -> bool:
        """Return true if self aabb is degenerate

        check if its min points equil its max points
        """
        pmin= self.get_min()
        pmax= self.get_max()

        check_x= maths.is_equil(pmin.x, pmax.x)
        check_y= maths.is_equil(pmin.y, pmax.y)
        check_z= maths.is_equil(pmin.z, pmax.z)

        return check_x and check_y and check_z

    def get_corners(self) -> list[maths.Vec3]:
        ptmin: maths.Vec3= self.get_min()
        ptmax: maths.Vec3= self.get_max()

        corners: list[maths.Vec3]= [
            maths.Vec3(ptmin.x, ptmin.y, ptmin.z),
            maths.Vec3(ptmax.x, ptmin.y, ptmin.z),
            maths.Vec3(ptmax.x, ptmax.y, ptmin.z),
            maths.Vec3(ptmin.x, ptmax.y, ptmin.z),
            maths.Vec3(ptmax.x, ptmin.y, ptmax.z),
            maths.Vec3(ptmax.x, ptmin.y, ptmax.z),
            maths.Vec3(ptmax.x, ptmax.y, ptmax.z),
            maths.Vec3(ptmin.x, ptmax.y, ptmax.z)
        ]
        return corners

    def contains_aabb(self, other: 'AABB3') -> bool:
        amin: maths.Vec3= self.get_min()
        amax: maths.Vec3= self.get_max()
        bmin: maths.Vec3= other.get_min()
        bmax: maths.Vec3= other.get_max()

        return (
            (amin.x <= bmin.x and amax.x >= bmax.x) and 
            (amin.y <= bmin.y and amax.y >= bmax.y) and 
            (amin.z <= bmin.z and amax.z >= bmax.z)
        )

    def intersect_aabb(self, other: 'AABB3') -> bool:
        amin: maths.Vec3= self.get_min()
        amax: maths.Vec3= self.get_max()
        bmin: maths.Vec3= other.get_min()
        bmax: maths.Vec3= other.get_max()
        
        return (
            (amin.x <= bmax.x and amax.x >= bmin.x) and
            (amin.y <= bmax.y and amax.y >= bmin.y) and
            (amin.z <= bmax.z and amax.z >= bmin.z)
        )

    def intersect_pt(self, pt: maths.Vec3) -> bool:
        pmin: maths.Vec3= self.get_min()
        pmax: maths.Vec3= self.get_max()

        if pt.x < pmin.x or pt.y < pmin.y or pt.z < pmin.z:
            return False

        if pt.x > pmax.x or pt.y > pmax.y or pt.z > pmax.z:
            return False

        return True


# --- Line


class Line3:

    __slots__= ('start', 'end', 'TYPE')

    def __init__(
        self,
        start: maths.Vec3= maths.Vec3(),
        end: maths.Vec3= maths.Vec3()
    ) -> None:
        self.start: maths.Vec3= start
        self.end: maths.Vec3= end
        self.TYPE: GeometryType= GeometryType.LINE

    def __eq__(self, other: 'Line3') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.start.is_equil(other.start) and
                self.end.is_equil(other.end) and
                self.TYPE == other.TYPE
            ):
                return True
        return False

    def __str__(self) -> str:
        return f'[START: {self.start}, END: {self.end}]'

    def edge(self) -> maths.Vec3:
        return (self.end - self.start)


# --- Triangle


class Triangle3:

    __slots__= ('p0', 'p1', 'p2', 'TYPE')

    def __init__(
        self,
        p0: maths.Vec3,
        p1: maths.Vec3,
        p2: maths.Vec3
    ) -> None:
        self.p0= p0.copy()
        self.p1= p1.copy()
        self.p2= p2.copy()
        self.TYPE: GeometryType= GeometryType.TRIANGLE

    def __eq__(self, other: 'Triangle3') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.p0.is_equil(other.p0) and
                self.p1.is_equil(other.p1) and
                self.p2.is_equil(other.p2) and
                self.TYPE == other.TYPE
            ):
                return True
        return False

    def __str__(self) -> str:
        return f'[P0: {self.p0}, P1: {self.p1}, P2: {self.p2}]'

    def intersect_pt(self, pt: maths.Vec3):
        bc= maths.Vec3.barycentric(
            self.p0,
            self.p1,
            self.p2,
            pt
        )
        # u: float= bc.x
        v: float= bc.y
        w: float= bc.z
        return (
            v >= 0.0 and
            w >= 0.0 and
            (v + w) <= 1.0
        )


# --- SPHERE


class Sphere3:

    __slots__= ('center', 'radius', 'TYPE')

    def __init__(
        self,
        center: maths.Vec3= maths.Vec3(),
        radius: float= 1.0
    ) -> None:
        self.center: maths.Vec3= center
        self.radius: float= radius
        self.TYPE: GeometryType= GeometryType.SPHERE

    def __eq__(self, other: 'Sphere3') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.center.is_equil(other.center) and
                maths.is_equil(self.radius, other.radius) and
                self.TYPE == other.TYPE
            ):
                return True
        return False

    def __str__(self) -> str:
        return f'[CENTER: {self.center}, RADIUS: {self.radius}]'

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

    def intersect_plain(self, plain: 'Plane'):
        close_pt: maths.Vec3= plain.closest_pt(self.center)
        dis: float= (self.center - close_pt).length_sqr()
        r2: float= maths.sqr(self.radius)

        return dis < r2


# --- PLANE


class Plane:

    __slots__= ('normal', 'd', 'TYPE')

    def __init__(self, norm: maths.Vec3=maths.Vec3(), d: float= 0.0) -> None:
        self.normal: maths.Vec3= norm
        self.d: float= d
        self.TYPE: GeometryType= GeometryType.PLAIN

    def __eq__(self, other: 'Plane') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.normal.is_equil(other.normal) and
                maths.is_equil(self.d, other.d) and
                self.TYPE == other.TYPE 
            ):
                return True
        return False

    def __str__(self) -> str:
        return f'[N: {self.normal}, D: {self.d:.4f}]'

    @staticmethod
    def create_from_v4(v4: maths.Vec4) -> 'Plane':
        return Plane(v4.xyz(), v4.w)

    @staticmethod
    def create_from_pts(a: maths.Vec3, b: maths.Vec3, c: maths.Vec3) -> 'Plane':
        ab: maths.Vec3= b - a
        ac: maths.Vec3= c - a
        n: maths.Vec3= ab.cross(ac)

        if not n.is_unit():
            n.to_unit()

        d: float= -n.dot(a)
        return Plane(n, d)

    @staticmethod
    def create_from_unit(unit_v3: maths.Vec3, pt: maths.Vec3):
        return Plane(unit_v3, unit_v3.dot(pt))

    @staticmethod
    def create_intersection_pt(a: 'Plane', b: 'Plane', c: 'Plane') -> maths.Vec3:
        """
        """
        na: maths.Vec3= a.normal
        nb: maths.Vec3= b.normal
        nc: maths.Vec3= c.normal
        den: float= -na.cross(nb).dot(nc)

        if maths.is_zero(den):
            return maths.Vec3.zero()

        #TODO() scaling by d causes bad scalinf

        inv: float= 1.0 / den
        # p0: maths.Vec3= nb.cross(nc) * a.d
        # p1: maths.Vec3= nc.cross(na) * b.d
        # p2: maths.Vec3= na.cross(nb) * c.d

        p0: maths.Vec3= nb.cross(nc)
        p1: maths.Vec3= nc.cross(na)
        p2: maths.Vec3= na.cross(nb)

        return (p0 + p1 + p2) * inv

    def xyzd(self)->maths.Vec4:
        return maths.Vec4(self.normal.x, self.normal.y, self.normal.z, self.d)
    
    def dot(self, v4: maths.Vec4) -> float:
        """
        """
        return self.to_vec4().dot(v4)

    def dot_normal(self, v3: maths.Vec3) -> float:
        """
        """
        return self.normal.dot(v3)

    def classify_pt(self, v3: maths.Vec3) -> int:
        """Check what side of plane v3 fall on
        
        results:
            <0 
        
            >0

            0
        """
        result: float= self.dot_normal(v3) + self.d
        return int(result)

    def classify_ab3(self, ab3: AABB3) -> int:
        """Check what side of plane ab3 fall on
        
        results:
            <0 
        
            >0

            0
        """
        pmin: maths.Vec3= ab3.get_min()
        pmax: maths.Vec3= ab3.get_min()

        dmin: float= 0.0
        dmax: float= 0.0

        if self.normal.x > 0.0:
            dmin += self.normal.x * pmin.x
            dmax += self.normal.x * pmax.x
        else:
            dmin += self.normal.x * pmax.x
            dmax += self.normal.x * pmin.x

        if self.normal.y > 0.0:
            dmin += self.normal.y * pmin.y
            dmax += self.normal.y * pmax.y
        else:
            dmin += self.normal.y * pmax.y
            dmax += self.normal.y * pmin.y

        if self.normal.z > 0.0:
            dmin += self.normal.z * pmin.z
            dmax += self.normal.z * pmax.z
        else:
            dmin += self.normal.z * pmax.z
            dmax += self.normal.z * pmin.z

        if dmin >= self.d:
            return 1
        if dmax <= self.d:
            return -1
        return 0

    def unit(self) -> 'Plane':
        """Return a copy of self with unit length
        """
        if maths.is_one(self.normal.length_sqr()):
            return
        ls= self.normal.length_sqrt()

        if maths.is_zero(ls):
            return Plane()

        inv: float = 1.0 / ls
        n= self.normal * inv
        d= self.d * inv if self.d >= 1.0 else self.d
        return Plane(n, d)

    def to_unit(self) -> None:
        """Convert to unit length
        """
        ls: float= self.normal.length_sqrt()
        if maths.is_zero(ls):
            return

        inv: float = 1.0 / ls
        self.normal.x *= inv
        self.normal.y *= inv
        self.normal.z *= inv
        self.d *= inv

# ---


class Frustum:
    
    __slots__= ('planes', 'TYPE' )

    def __init__(self) -> None:
        self.planes: list[Plane]= [Plane()] * 6
        self.TYPE: GeometryType= GeometryType.FRUSTUM

    @staticmethod
    def create_from_matrix(vp_matrix: maths.Mat4) -> 'Frustum':
        """Create frustum from view projection matrix
        """
        result: Frustum= Frustum()

        near: maths.Vec4= vp_matrix.col3() + vp_matrix.col2()
        far: maths.Vec4= vp_matrix.col3() - vp_matrix.col2()
        left: maths.Vec4= vp_matrix.col3() + vp_matrix.col0()
        right: maths.Vec4= vp_matrix.col3() - vp_matrix.col0()
        top: maths.Vec4= vp_matrix.col3() - vp_matrix.col1()
        bottom: maths.Vec4= vp_matrix.col3() + vp_matrix.col1()

        result.planes[0]= Plane.create_from_v4(near)
        result.planes[1]= Plane.create_from_v4(far)
        result.planes[2]= Plane.create_from_v4(left)
        result.planes[3]= Plane.create_from_v4(right)
        result.planes[4]= Plane.create_from_v4(top)
        result.planes[5]= Plane.create_from_v4(bottom)
        
        for pl in result.planes:
            pl.to_unit()

        return result

    def get_near(self) -> Plane:
        """Return near Plane
        """
        return self.planes[0]

    def get_far(self) -> Plane:
        """Return far Plane
        """
        return self.planes[1]

    def get_left(self) -> Plane:
        """Return left Plane
        """
        return self.planes[2]

    def get_right(self) -> Plane:
        """Return right Plane
        """
        return self.planes[3]
    
    def get_top(self) -> Plane:
        """Return top Plane
        """
        return self.planes[4]
    
    def get_bottom(self) -> Plane:
        """Return bottom Plane
        """
        return self.planes[5]

    def get_corners(self) -> list[maths.Vec3]:
        """Return corners of camera frustum

        [ nbl, nbr, ntl, ntr, fbl, fbr, ftl, ftr ]
        """
        n: Final[int]= 0
        f: Final[int]= 1
        l: Final[int]= 2
        r: Final[int]= 3
        t: Final[int]= 4
        b: Final[int]= 5

        corners: list[maths.Vec3]= [
            Plane.create_intersection_pt(
                self.planes[n], self.planes[b], self.planes[l]
            ),
            Plane.create_intersection_pt(
                self.planes[n], self.planes[b], self.planes[r]
            ),
            Plane.create_intersection_pt(
                self.planes[n], self.planes[t], self.planes[l]
            ),
            Plane.create_intersection_pt(
                self.planes[n], self.planes[t], self.planes[r]
            ),
            Plane.create_intersection_pt(
                self.planes[f], self.planes[b], self.planes[l]
            ),
            Plane.create_intersection_pt(
                self.planes[f], self.planes[b], self.planes[r]
            ),
            Plane.create_intersection_pt(
                self.planes[f], self.planes[t], self.planes[l]
            ),
            Plane.create_intersection_pt(
                self.planes[f], self.planes[t], self.planes[r]
            )
        ]

        return corners


# --- RAY3D


class Ray3:

    __slots__= ('origin', 'direction', 'TYPE')

    def __init__(
        self,
        origin: maths.Vec3= maths.Vec3(),
        direction: maths.Vec3= maths.Vec3()
    ) -> None:
        self.origin: maths.Vec3= origin
        self.direction: maths.Vec3= direction
        self.TYPE: GeometryType= GeometryType.RAY

        if not self.direction.is_unit():
            self.direction.to_unit()

    def __eq__(self, other: 'Ray3') -> bool:
        if isinstance(other, self.__class__):
            if(
                self.origin.is_equil(other.origin) and
                self.direction.is_equil(other.direction) and
                self.TYPE == other.TYPE 
            ):
                return True
        return False

    def __str__(self) -> str:
        return f'[O: {self.origin}, D: {self.direction}]'

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

    def cast_aabb(self, aabb: AABB3) -> float:
        pmin: maths.Vec3= aabb.get_min()
        pmax: maths.Vec3= aabb.get_max()
        tmin: float= maths.MIN_FLOAT
        tmax: float= maths.MAX_FLOAT

        for idx in range(3):
            if maths.is_zero(self.direction.get_at(idx)):
                if(
                    self.origin.get_at(idx) < pmin.get_at(idx) or 
                    self.origin.get_at(idx) > pmax.get_at(idx)
                ):
                    return -1.0
            else:
                inv: float= 1.0 / self.direction.get_at(idx)

                t1: float= (pmin.get_at(idx) - self.origin.get_at(idx)) * inv
                t2: float= (pmax.get_at(idx) - self.origin.get_at(idx)) * inv

                if t1 > t2:
                    t1, t2= maths.swapf(t1, t2)

                if t1 > tmin:
                    tmin= t1

                if t2 > tmax:
                    tmax= t2

                if tmin > tmax:
                    return -1.0

        return tmin

    def cast_sphere(self, sph: Sphere3) -> float:
        a: maths.Vec3= sph.center - self.origin
        b: float= a.dot(self.direction)
        c: float= a.length_sqr() - maths.sqr(sph.radius)

        if c > 0.0 and b > 0.0:
            return -1.0

        d: float= maths.sqr(b) - c
        if d < 0.0:
            return -1.0

        t: float= -b - maths.sqrt(d)
        if t < 0.0:
            t= 0.0

        return t

    def cast_plane(self, pl: Plane) -> float:
        nd: float= self.direction.dot(pl.normal)
        pn: float= self.origin.dot(pl.normal)

        if nd >= 0.0:
            return -1.0

        t: float= (pl.d - pn) / nd

        if t >= 0.0:
            return t

        return -1.0
